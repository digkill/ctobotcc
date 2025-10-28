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
