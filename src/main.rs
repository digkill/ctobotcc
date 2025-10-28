use anyhow::{bail, Result};
use axum::{
    routing::{get},
    Router,
};
use clap::{Parser, Subcommand};
use sqlx::mysql::{MySqlConnectOptions, MySqlSslMode};
use sqlx::pool::PoolOptions;
use sqlx::{MySql, Pool};
use std::{env, str::FromStr};
use tracing::{info};
use tracing_subscriber::{prelude::*, EnvFilter};

mod commands;
mod db;
mod handlers;
mod openai;
mod qa;
mod tamtam;
mod context;

use handlers::webhook;
use qa::KnowledgeBase;

/* ===================== CLI ===================== */

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Option<Commands>,
}

#[derive(Subcommand, Debug)]
enum Commands {
    /// Create a new mailbox
    CreateMailbox {
        /// Local part (left of @) OR full email if you prefer
        #[arg(short, long)]
        email: String,

        /// Password for the new mailbox
        #[arg(short, long)]
        password: String,

        /// Domain where the mailbox will be created
        #[arg(short = 'd', long = "domain")]
        domain: String,
    },
    /// List all mailboxes for a domain
    ListMailboxes {
        /// Domain name
        #[arg(short = 'd', long = "domain")]
        domain: String,
    },
    /// Drop (delete) mailbox
    DropMailbox {
        /// Full email address
        #[arg(short, long)]
        email: String,
    },
    /// Ask CodeClassGPT with DB context
    Cc {
        /// Question text
        #[arg(trailing_var_arg = true)]
        question: Vec<String>,
    },
}


/* ===================== entry ===================== */

#[tokio::main]
async fn main() -> Result<()> {
    // env: .env из CWD, затем — рядом с Cargo.toml
    let _ = dotenvy::dotenv();
    if std::env::var("DATABASE_URL").is_err() {
        if let Ok(path) = std::env::var("CARGO_MANIFEST_DIR") {
            let env_path = std::path::Path::new(&path).join(".env");
            dotenvy::from_path(&env_path).ok();
        }
    }

    // Initialize logging FIRST.
    let file_appender = tracing_appender::rolling::daily(".", "ctobot.log");
    let (non_blocking_appender, _guard) = tracing_appender::non_blocking(file_appender);
    tracing_subscriber::registry()
        .with(EnvFilter::new("info"))
        .with(tracing_subscriber::fmt::layer().with_writer(std::io::stdout))
        .with(tracing_subscriber::fmt::layer().with_writer(non_blocking_appender))
        .init();

    let cli = Cli::parse();

    if let Some(command) = cli.command {
        match command {
            Commands::CreateMailbox { email, password, domain } => {
                // Accept either local part or full email; normalize to local part
                let local = if let Some((l, d)) = email.split_once('@') {
                    if d != domain { bail!("Email domain and --domain mismatch"); }
                    l.to_string()
                } else {
                    email
                };
                if !commands::mail::allowed_domain(&domain) {
                    bail!("Domain not allowed. Allowed domains: code-class.ru, uchi.team");
                }

                println!("Creating mailbox {}@{}...", local, domain);
                match commands::mail::beget_create_mailbox(&domain, &local, &password).await {
                    Ok(true) => println!("Mailbox created successfully."),
                    Ok(false) => println!("Failed to create mailbox (API returned false)."),
                    Err(e) => println!("Error creating mailbox: {}", e),
                }
            }
            Commands::ListMailboxes { domain } => {
                println!("Listing mailboxes for domain {}...", &domain);
                match commands::mail::beget_list_mailboxes(&domain).await {
                    Ok(list) => {
                        if list.is_empty() {
                            println!("No mailboxes found.");
                        } else {
                            for m in list { println!("{}", m); }
                        }
                    }
                    Err(e) => println!("Error listing mailboxes: {}", e),
                }
            }
            Commands::DropMailbox { email } => {
                let (local, domain) = match email.split_once('@') {
                    Some((l, d)) => (l.to_string(), d.to_string()),
                    None => bail!("Invalid email format. Use user@domain.com"),
                };
                if !commands::mail::allowed_domain(&domain) {
                    bail!("Domain not allowed. Allowed domains: code-class.ru, uchi.team");
                }
                println!("Dropping mailbox {}...", email);
                match commands::mail::beget_drop_mailbox(&domain, &local).await {
                    Ok(true) => println!("Mailbox dropped."),
                    Ok(false) => println!("Failed to drop mailbox (API returned false)."),
                    Err(e) => println!("Error dropping mailbox: {}", e),
                }
            }
            Commands::Cc { question } => {
                let q = question.join(" ").trim().to_string();
                if q.is_empty() { bail!("Usage: cc <question>"); }

                // Build pools and KB once for CLI
                let pool_rw = build_pool_rw().await?;
                let pool_codeclass_ro = build_pool_codeclass_ro().await?;
                let kb = qa::load_kb_json(&env::var("KNOWLEDGE_JSON").unwrap_or_else(|_| "knowledge.json".to_string()));
                let app_state = AppState { pool_rw: pool_rw.clone(), pool_codeclass_ro: pool_codeclass_ro.clone(), kb };

                // Persona — CodeClassGPT
                let system_prompt = openai::codeclassgpt_prompt();
                let history_max: i64 = env::var("HISTORY_MAX").ok().and_then(|s| s.parse().ok()).unwrap_or(12);

                // For CLI, use a fixed hist_key
                let hist_key: i64 = 0;

                // 0) Q&A shortcuts (codeclass.RO → JSON)
                if let Some(answer) = qa::try_answer_from_codeclass_qa(&app_state, &q).await
                    .or_else(|| qa::try_answer_from_json_qa(&app_state, &q))
                {
                    println!("{}", answer);
                    return Ok(());
                }

                // 1) История (ctoseo.RW)
                let mut messages = vec![openai::OpenAIMessage { role: "system".into(), content: system_prompt }];
                if let Ok(h) = db::ctoseo::load_history(&pool_rw, hist_key, history_max).await { messages.extend(h); }
                messages.push(openai::OpenAIMessage { role: "user".into(), content: q.clone() });

                // 2) Факты (ctoseo.RW)
                if let Ok(facts) = db::ctoseo::load_facts(&pool_rw, hist_key).await {
                    if !facts.is_empty() {
                        let joined = facts.join("\n- ");
                        messages.insert(1, openai::OpenAIMessage { role: "system".into(), content: format!("Дополнительные факты (БД ctoseo):\n- {}", joined) });
                    }
                }

                // 2.5) Динамический контекст из codeclass
                if let Some(ctx) = context::build_dynamic_context(&app_state, &q).await {
                    messages.insert(1, openai::OpenAIMessage { role: "system".into(), content: ctx });
                }

                // 3) Подсказки
                if let Some(hints) = qa::weak_hints_from_codeclass_and_json(&app_state, &q).await {
                    messages.insert(1, openai::OpenAIMessage { role: "system".into(), content: format!("Подсказки из Q&A:\n{}", hints) });
                }

                // 4) Модель
                let answer = match openai::ask_openai(messages).await {
                    Ok(a) if !a.trim().is_empty() => a,
                    Ok(_) => "…".to_string(),
                    Err(e) => { eprintln!("OpenAI error: {e:?}"); "CodeClassGPT: временно не могу ответить. Попробуйте ещё раз.".to_string() }
                };
                println!("{}", answer);
            }
        }
        return Ok(());
    }

    println!("CWD: {}", std::env::current_dir()?.display());

    /* ---------- MySQL pools (TLS REQUIRED to avoid 1835) ---------- */
    // ctoseo (RW)
    let db_rw_url = std::env::var("DATABASE_URL").expect("DATABASE_URL not set (ctoseo)");
    let rw_opts = MySqlConnectOptions::from_str(&db_rw_url)
        .expect("bad DATABASE_URL")
        .ssl_mode(MySqlSslMode::Required);
    let pool_rw = PoolOptions::<MySql>::new()
        .max_connections(10)
        .acquire_timeout(std::time::Duration::from_secs(10))
        .connect_with(rw_opts)
        .await?;
    info!("Connected to MySQL (ctoseo, RW)");

    // codeclass (RO)
    let db_ro_url = std::env::var("CODECLASS_DATABASE_URL_RO")
        .expect("CODECLASS_DATABASE_URL_RO not set (codeclass, RO)");
    let ro_opts = MySqlConnectOptions::from_str(&db_ro_url)
        .expect("bad CODECLASS_DATABASE_URL_RO")
        .ssl_mode(MySqlSslMode::Required);
    let pool_codeclass_ro = PoolOptions::<MySql>::new()
        .max_connections(10)
        .acquire_timeout(std::time::Duration::from_secs(10))
        .connect_with(ro_opts)
        .await?;
    info!("Connected to MySQL (codeclass, RO)");

    // JSON Q&A
    let kb =
        qa::load_kb_json(&env::var("KNOWLEDGE_JSON").unwrap_or_else(|_| "knowledge.json".to_string()));

    let app = Router::new()
        .route("/tamtam/webhook", axum::routing::post(webhook))
        .route("/healthz", get(|| async { "ok" }))
        .with_state(AppState {
            pool_rw,
            pool_codeclass_ro,
            kb,
        });

    let port: u16 = env::var("PORT")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(3011);
    let addr = std::net::SocketAddr::from(([0, 0, 0, 0], port));
    info!("Listening on {addr}");
    axum::serve(tokio::net::TcpListener::bind(addr).await?, app).await?;
    Ok(())
}

#[derive(Clone)]
pub struct AppState {
    pub pool_rw: Pool<MySql>,           // ctoseo (RW)
    pub pool_codeclass_ro: Pool<MySql>, // codeclass (RO)
    pub kb: KnowledgeBase,
}

// Helpers to build pools for CLI
async fn build_pool_rw() -> Result<Pool<MySql>> {
    let db_rw_url = std::env::var("DATABASE_URL").expect("DATABASE_URL not set (ctoseo)");
    let rw_opts = MySqlConnectOptions::from_str(&db_rw_url)
        .expect("bad DATABASE_URL")
        .ssl_mode(MySqlSslMode::Required);
    let pool_rw = PoolOptions::<MySql>::new()
        .max_connections(5)
        .acquire_timeout(std::time::Duration::from_secs(10))
        .connect_with(rw_opts)
        .await?;
    Ok(pool_rw)
}

async fn build_pool_codeclass_ro() -> Result<Pool<MySql>> {
    let db_ro_url = std::env::var("CODECLASS_DATABASE_URL_RO").expect("CODECLASS_DATABASE_URL_RO not set (codeclass, RO)");
    let ro_opts = MySqlConnectOptions::from_str(&db_ro_url)
        .expect("bad CODECLASS_DATABASE_URL_RO")
        .ssl_mode(MySqlSslMode::Required);
    let pool_ro = PoolOptions::<MySql>::new()
        .max_connections(5)
        .acquire_timeout(std::time::Duration::from_secs(10))
        .connect_with(ro_opts)
        .await?;
    Ok(pool_ro)
}
