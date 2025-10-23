use axum::{routing::get, routing::post, Json, Router};
use axum::http::StatusCode;
use serde::{Deserialize, Serialize};
use std::env;
use tracing::{error, info};
use sqlx::{MySql, Pool, Row};
use chrono::Utc;
use anyhow::Result;

// ======================= entry =======================

#[tokio::main]
async fn main() -> Result<()> {
    // 1) .env — из CWD
    let _ = dotenvy::dotenv();
    // 2) .env — рядом с Cargo.toml (если не нашли)
    if std::env::var("DATABASE_URL").is_err() {
        let manifest_env = format!("{}/.env", env!("CARGO_MANIFEST_DIR"));
        let _ = dotenvy::from_filename(&manifest_env);
    }

    tracing_subscriber::fmt().with_env_filter("info").compact().init();
    println!("CWD: {}", std::env::current_dir()?.display());

    // DB
    let db_url = std::env::var("DATABASE_URL")
        .expect("DATABASE_URL not set (проверь .env или $env:DATABASE_URL)");
    let pool = Pool::<MySql>::connect(&db_url).await?;
    info!("Connected to MySQL");

    // (Опционально) если используешь миграции sqlx::migrate! — раскомментируй:
    // sqlx::migrate!("./migrations").run(&pool).await?;
    // info!("Migrations applied");

    // Загрузка JSON-базы знаний в память
    let kb = load_kb_json(
        &env::var("KNOWLEDGE_JSON").unwrap_or_else(|_| "knowledge.json".to_string())
    );

    let app = Router::new()
        .route("/tamtam/webhook", post(webhook))
        .route("/healthz", get(|| async { "ok" }))
        .with_state(AppState { pool, kb });

    let port: u16 = env::var("PORT").ok().and_then(|s| s.parse().ok()).unwrap_or(3000);
    let addr = std::net::SocketAddr::from(([0,0,0,0], port));
    info!("Listening on {addr}");
    axum::serve(tokio::net::TcpListener::bind(addr).await?, app).await?;
    Ok(())
}

#[derive(Clone)]
struct AppState {
    pool: Pool<MySql>,
    kb: KnowledgeBase, // in-memory JSON KB
}

// ================= TamTam payloads ====================

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
    #[serde(default)]
    sender: Option<TamTamSender>,
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

#[derive(Debug, Deserialize, Default)]
struct TamTamSender {
    #[serde(default)]
    user_id: Option<i64>,
}

enum TTRecipientKind { Chat(i64), User(i64) }

fn pick_recipient(msg: &TamTamMessageWrapper) -> Option<TTRecipientKind> {
    if let Some(id) = msg.recipient.chat_id
        .or(msg.chat_id)
        .or(msg.chat.as_ref().and_then(|c| c.chat_id)) {
        return Some(TTRecipientKind::Chat(id));
    }
    if let Some(uid) = msg.sender.as_ref().and_then(|s| s.user_id) {
        return Some(TTRecipientKind::User(uid));
    }
    None
}

// ================= OpenAI types =======================

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

// ================= webhook ============================

async fn webhook(
    axum::extract::State(state): axum::extract::State<AppState>,
    Json(update): Json<TamTamUpdate>,
) -> StatusCode {
    tokio::spawn(handle_update(state, update));
    StatusCode::OK
}

async fn handle_update(state: AppState, update: TamTamUpdate) {
    let Some(msg) = update.message else { return; };
    let text = msg.body.text.trim().to_string();
    if text.is_empty() { return; }

    let Some(recipient) = pick_recipient(&msg) else {
        error!("No recipient found in update: {:?}", msg);
        return;
    };

    // История/системный промпт
    let history_max: i64 = env::var("HISTORY_MAX").ok().and_then(|s| s.parse().ok()).unwrap_or(12);
    let system_prompt = env::var("SYSTEM_PROMPT")
        .unwrap_or_else(|_| "You are a helpful assistant.".to_string());

    let mut messages = vec![ OpenAIMessage { role: "system".into(), content: system_prompt } ];

    // История из БД (по chat_id для чатов / по user_id для лички — будем хранить раздельно)
    let history_key = history_key_for(&recipient);
    if let Ok(h) = load_history(&state.pool, history_key, history_max).await {
        messages.extend(h);
    }

    // Вопрос пользователя
    messages.push(OpenAIMessage { role: "user".into(), content: text.clone() });

    // Факты из БД
    if let Ok(facts) = load_facts(&state.pool, history_key).await {
        if !facts.is_empty() {
            let joined = facts.join("\n- ");
            messages.insert(1, OpenAIMessage {
                role: "system".into(),
                content: format!("Дополнительные факты из БД:\n- {}", joined),
            });
        }
    }

    // Факты из JSON KB
    if !state.kb.items.is_empty() {
        let joined = state.kb.items.join("\n- ");
        messages.insert(1, OpenAIMessage {
            role: "system".into(),
            content: format!("Дополнительные факты из knowledge.json:\n- {}", joined),
        });
    }

    // Вызов OpenAI
    let answer = match ask_openai(messages.clone()).await {
        Ok(a) if !a.trim().is_empty() => a,
        Ok(_) => "…".to_string(),
        Err(e) => {
            error!("OpenAI error: {e:?}");
            "Упс, сейчас не могу ответить. Попробуйте ещё раз.".to_string()
        }
    };

    // Сохраняем историю
    if let Err(e) = save_history_batch(&state.pool, history_key, &[
        OpenAIMessage { role: "user".into(), content: text },
        OpenAIMessage { role: "assistant".into(), content: answer.clone() },
    ]).await {
        error!("save_history_batch error: {e:?}");
    }

    // Отправляем ответ
    if let Err(e) = send_tamtam(recipient, &answer).await {
        error!("send_tamtam error: {e:?}");
    }
}

// ================= DB helpers =========================

fn history_key_for(r: &TTRecipientKind) -> i64 {
    match *r {
        TTRecipientKind::Chat(id) => id,
        TTRecipientKind::User(uid) => -uid, // Лайфхак: лички храним как отрицательные ключи, чтобы не путать с chat_id
    }
}

async fn load_history(pool: &Pool<MySql>, key: i64, limit: i64) -> Result<Vec<OpenAIMessage>> {
    let rows = sqlx::query(
        r#"
        SELECT role, content
        FROM chat_history
        WHERE chat_id = ?
        ORDER BY created_at DESC
        LIMIT ?
        "#
    )
        .bind(key)
        .bind(limit)
        .fetch_all(pool).await?;

    let mut items: Vec<OpenAIMessage> = rows.into_iter().rev().map(|row| {
        let role: String = row.try_get("role").unwrap_or_default();
        let content: String = row.try_get("content").unwrap_or_default();
        OpenAIMessage { role, content }
    }).collect();

    items.retain(|m| matches!(m.role.as_str(), "system" | "user" | "assistant"));
    Ok(items)
}

async fn save_history_batch(pool: &Pool<MySql>, key: i64, msgs: &[OpenAIMessage]) -> Result<()> {
    let mut tx = pool.begin().await?;
    for m in msgs {
        sqlx::query(
            r#"
            INSERT INTO chat_history (chat_id, role, content, created_at)
            VALUES (?, ?, ?, ?)
            "#
        )
            .bind(key)
            .bind(&m.role)
            .bind(&m.content)
            .bind(Utc::now().naive_utc())
            .execute(&mut *tx)
            .await?;
    }
    tx.commit().await?;
    Ok(())
}

async fn load_facts(pool: &Pool<MySql>, key: i64) -> Result<Vec<String>> {
    let rows = sqlx::query(
        r#"
        SELECT content FROM facts
        WHERE (chat_id IS NULL) OR (chat_id = ?)
        ORDER BY created_at DESC
        LIMIT 50
        "#
    )
        .bind(key)
        .fetch_all(pool).await?;

    Ok(rows.into_iter().map(|row| row.try_get::<String, _>("content").unwrap_or_default()).collect())
}

// ================= TamTam send ========================

use serde_json::json;

async fn send_tamtam(recipient: TTRecipientKind, text: &str) -> Result<()> {
    let token = env::var("TT_BOT_TOKEN").expect("TT_BOT_TOKEN not set");
    let url = format!(
        "https://botapi.tamtam.chat/messages?access_token={}",
        urlencoding::encode(&token)
    );

    let recipient_json = match recipient {
        TTRecipientKind::Chat(chat_id) => json!({ "chat_id": chat_id }),
        TTRecipientKind::User(user_id) => json!({ "user_id": user_id }),
    };

    let body = json!({
        "recipient": recipient_json,
        "message": { "text": text }
    });

    let client = reqwest::Client::new();
    let res = client.post(url).json(&body).send().await?;
    if !res.status().is_success() {
        let t = res.text().await.unwrap_or_default();
        error!("TamTam send failed: {}", t);
    }
    Ok(())
}

// ================= OpenAI call ========================

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

// ============== JSON knowledge base ===================

#[derive(Clone, Default)]
struct KnowledgeBase {
    items: Vec<String>,
}

#[derive(Deserialize)]
struct KnowledgeJson {
    #[serde(default)]
    facts: Vec<String>,
}

fn load_kb_json(path: &str) -> KnowledgeBase {
    match std::fs::read_to_string(path) {
        Ok(s) => {
            match serde_json::from_str::<KnowledgeJson>(&s) {
                Ok(k) => {
                    info!("Loaded knowledge.json: {} facts", k.facts.len());
                    KnowledgeBase { items: k.facts }
                }
                Err(e) => {
                    error!("knowledge.json parse error: {e:?}");
                    KnowledgeBase::default()
                }
            }
        }
        Err(e) => {
            info!("knowledge.json not found ({}): {}", path, e);
            KnowledgeBase::default()
        }
    }
}
