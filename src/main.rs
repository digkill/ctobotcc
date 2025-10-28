use anyhow::Result;
use axum::{
    routing::{get},
    Router,
};
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

/* ===================== entry ===================== */

#[tokio::main]
async fn main() -> Result<()> {
    // env: .env из CWD, затем — рядом с Cargo.toml
    let _ = dotenvy::dotenv();
    if std::env::var("DATABASE_URL").is_err() {
        let manifest_env = format!("{}/.env", env!("CARGO_MANIFEST_DIR"));
        let _ = dotenvy::from_filename(&manifest_env);
    }

    let file_appender = tracing_appender::rolling::daily(".", "ctobot.log");
    let (non_blocking_appender, _guard) = tracing_appender::non_blocking(file_appender);

    tracing_subscriber::registry()
        .with(EnvFilter::new("info"))
        .with(tracing_subscriber::fmt::layer().with_writer(std::io::stdout))
        .with(tracing_subscriber::fmt::layer().with_writer(non_blocking_appender))
        .init();

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
