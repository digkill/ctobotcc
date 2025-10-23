use axum::{routing::post, Json, Router};
use axum::http::StatusCode;
use serde::{Deserialize, Serialize};
use std::env;
use tracing::{error, info};
use sqlx::{MySql, Pool};
use chrono::Utc;
use anyhow::Result;
use sqlx::Row;

#[tokio::main]
async fn main() -> Result<()> {
    // Попробуем стандартную загрузку
    let _ = dotenvy::dotenv();

    // Если переменной нет — попробуем .env рядом с Cargo.toml (MANIFEST_DIR)
    if std::env::var("DATABASE_URL").is_err() {
        let manifest_env = format!("{}/.env", env!("CARGO_MANIFEST_DIR"));
        let _ = dotenvy::from_filename(&manifest_env);
    }

    // Лог текущей директории для отладки
    if let Ok(cwd) = std::env::current_dir() {
        tracing_subscriber::fmt().with_env_filter("info").compact().init();
        println!("CWD: {}", cwd.display());
    } else {
        tracing_subscriber::fmt().with_env_filter("info").compact().init();
    }

    let db_url = std::env::var("DATABASE_URL")
        .expect("DATABASE_URL not set (проверь .env в корне проекта или $env:DATABASE_URL)");

    let pool = Pool::<MySql>::connect(&db_url).await?;
    tracing::info!("Connected to MySQL");

    //sqlx::migrate!("./migrations").run(&pool).await?;
    //tracing::info!("Migrations applied");

    let app = Router::new()
        .route("/tamtam/webhook", post(webhook))
        .with_state(AppState { pool });

    let port: u16 = env::var("PORT").ok().and_then(|s| s.parse().ok()).unwrap_or(3000);
    let addr = std::net::SocketAddr::from(([0,0,0,0], port));
    info!("Listening on {addr}");
    axum::serve(tokio::net::TcpListener::bind(addr).await?, app).await?;
    Ok(())
}

#[derive(Clone)]
struct AppState {
    pool: Pool<MySql>,
}

/* ------------ TamTam payloads (минимально нужные поля) ------------- */

#[derive(Debug, Deserialize)]
struct TamTamUpdate {
    message: Option<TamTamMessageWrapper>,
}

#[derive(Debug, Deserialize)]
struct TamTamMessageWrapper {
    #[serde(default)]
    body: TamTamBody,
    #[serde(default)]
    recipient: TamTamRecipient,
    #[serde(default)]
    chat_id: Option<i64>,
    #[serde(default)]
    chat: Option<TamTamChat>,
}

#[derive(Debug, Deserialize, Default)]
struct TamTamBody {
    #[serde(default)]
    text: String,
}

#[derive(Debug, Deserialize, Default)]
struct TamTamRecipient {
    #[serde(default)]
    chat_id: Option<i64>,
}

#[derive(Debug, Deserialize, Default)]
struct TamTamChat {
    #[serde(default)]
    chat_id: Option<i64>,
}

/* -------------------- OpenAI minimal types ------------------------- */

#[derive(Serialize)]
struct OpenAIChatRequest {
    model: String,
    messages: Vec<OpenAIMessage>,
    temperature: f32,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
struct OpenAIMessage {
    role: String,
    content: String,
}

#[derive(Deserialize, Debug)]
struct OpenAIChatResponse {
    choices: Vec<OpenAIChoice>,
}

#[derive(Deserialize, Debug)]
struct OpenAIChoice {
    message: OpenAIMessage,
}

/* ---------------------- Webhook handler ---------------------------- */

async fn webhook(
    axum::extract::State(state): axum::extract::State<AppState>,
    Json(update): Json<TamTamUpdate>,
) -> StatusCode {
    // Быстрый 200 — TamTam ретраит при долгих ответах
    tokio::spawn(handle_update(state, update));
    StatusCode::OK
}

async fn handle_update(state: AppState, update: TamTamUpdate) {
    if update.message.is_none() { return; }
    let msg = update.message.unwrap();

    let text = msg.body.text.trim().to_string();
    if text.is_empty() { return; }

    // chat_id достаем из нескольких мест (в TamTam бывают разные формы)
    let chat_id = msg.recipient.chat_id
        .or(msg.chat_id)
        .or(msg.chat.and_then(|c| c.chat_id))
        .unwrap_or_default();

    // 1) Собираем историю и факты из БД
    let history_max: i64 = env::var("HISTORY_MAX").ok().and_then(|s| s.parse().ok()).unwrap_or(12);
    let system_prompt = env::var("SYSTEM_PROMPT")
        .unwrap_or_else(|_| "You are a helpful assistant.".to_string());

    let mut messages = vec![
        OpenAIMessage { role: "system".into(), content: system_prompt }
    ];

    if let Ok(h) = load_history(&state.pool, chat_id, history_max).await {
        messages.extend(h);
    }

    // Текущий запрос пользователя
    messages.push(OpenAIMessage { role: "user".into(), content: text.clone() });

    // Подмешаем факты (глобальные + персональные) отдельным system-сообщением
    if let Ok(facts) = load_facts(&state.pool, chat_id).await {
        if !facts.is_empty() {
            let joined = facts.join("\n- ");
            messages.insert(1, OpenAIMessage {
                role: "system".into(),
                content: format!("Дополнительные факты (для справки):\n- {}", joined),
            });
        }
    }

    // 2) Вызов OpenAI
    let answer = match ask_openai(messages.clone()).await {
        Ok(a) if !a.trim().is_empty() => a,
        Ok(_) => "…".to_string(),
        Err(e) => {
            error!("OpenAI error: {e:?}");
            "Упс, сейчас не могу ответить. Попробуйте ещё раз.".to_string()
        }
    };

    // 3) Сохраняем в историю и шлем ответ
    if let Err(e) = save_history_batch(&state.pool, chat_id, &[
        OpenAIMessage { role: "user".into(), content: text },
        OpenAIMessage { role: "assistant".into(), content: answer.clone() },
    ]).await {
        error!("save_history_batch error: {e:?}");
    }

    if let Err(e) = send_tamtam(chat_id, &answer).await {
        error!("send_tamtam error: {e:?}");
    }
}

/* ---------------------- DB helpers ---------------------- */

async fn load_history(pool: &Pool<MySql>, chat_id: i64, limit: i64) -> Result<Vec<OpenAIMessage>> {
    let rows = sqlx::query(
        r#"
        SELECT role, content
        FROM chat_history
        WHERE chat_id = ?
        ORDER BY created_at DESC
        LIMIT ?
        "#
    )
        .bind(chat_id)
        .bind(limit)
        .fetch_all(pool).await?;

    let mut items: Vec<OpenAIMessage> = rows.into_iter().rev().map(|row| {
        // вытащим поля вручную
        let role: String = row.try_get("role").unwrap_or_default();
        let content: String = row.try_get("content").unwrap_or_default();
        OpenAIMessage { role, content }
    }).collect();

    items.retain(|m| matches!(m.role.as_str(), "system" | "user" | "assistant"));
    Ok(items)
}


async fn save_history_batch(pool: &Pool<MySql>, chat_id: i64, msgs: &[OpenAIMessage]) -> Result<()> {
    let mut tx = pool.begin().await?;
    for m in msgs {
        sqlx::query(
            r#"
            INSERT INTO chat_history (chat_id, role, content, created_at)
            VALUES (?, ?, ?, ?)
            "#
        )
            .bind(chat_id)
            .bind(&m.role)
            .bind(&m.content)
            .bind(Utc::now().naive_utc())
            .execute(&mut *tx)
            .await?;
    }
    tx.commit().await?;
    Ok(())
}

async fn load_facts(pool: &Pool<MySql>, chat_id: i64) -> Result<Vec<String>> {
    let rows = sqlx::query(
        r#"
        SELECT content FROM facts
        WHERE (chat_id IS NULL) OR (chat_id = ?)
        ORDER BY created_at DESC
        LIMIT 20
        "#
    )
        .bind(chat_id)
        .fetch_all(pool).await?;

    Ok(rows.into_iter().map(|row| row.try_get::<String, _>("content").unwrap_or_default()).collect())
}

/* ---------------- TamTam send ---------------- */

#[derive(Serialize)]
struct TTRecipient { chat_id: i64 }
#[derive(Serialize)]
struct TTMessage { text: String }
#[derive(Serialize)]
struct TTBody { recipient: TTRecipient, message: TTMessage }

async fn send_tamtam(chat_id: i64, text: &str) -> Result<()> {
    let token = env::var("TT_BOT_TOKEN").expect("TT_BOT_TOKEN not set");
    let url = format!(
        "https://botapi.tamtam.chat/messages?access_token={}",
        urlencoding::encode(&token)
    );

    let body = TTBody {
        recipient: TTRecipient { chat_id },
        message: TTMessage { text: text.to_string() },
    };

    let client = reqwest::Client::new();
    let res = client.post(url).json(&body).send().await?;
    if !res.status().is_success() {
        let t = res.text().await.unwrap_or_default();
        error!("TamTam send failed: {}", t);
    }
    Ok(())
}

/* ---------------- OpenAI call (фикс borrow) ---------------- */

async fn ask_openai(messages: Vec<OpenAIMessage>) -> Result<String> {
    let api_key = env::var("OPENAI_API_KEY").expect("OPENAI_API_KEY not set");
    let model = env::var("OPENAI_MODEL").unwrap_or_else(|_| "gpt-4o-mini".into());

    let req = OpenAIChatRequest { model, messages, temperature: 0.7 };

    let client = reqwest::Client::new();
    let res = client
        .post("https://api.openai.com/v1/chat/completions")
        .bearer_auth(api_key)
        .json(&req)
        .send()
        .await?;

    let status = res.status();
    if !status.is_success() {
        let body = res.text().await.unwrap_or_default(); // text(self) потребляет res
        anyhow::bail!("OpenAI HTTP {}: {}", status, body);
    }

    let data: OpenAIChatResponse = res.json().await?;
    let answer = data.choices.get(0)
        .map(|c| c.message.content.clone())
        .unwrap_or_else(|| "…".to_string());

    Ok(answer)
}

