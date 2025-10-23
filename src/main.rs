use anyhow::Result;
use axum::http::StatusCode;
use axum::{
    Json, Router,
    routing::{get, post},
};
use chrono::Utc;
use regex::Regex;
use serde::{Deserialize, Serialize};
use sqlx::{MySql, Pool, Row};
use std::env;
use tracing::{error, info};

#[tokio::main]
async fn main() -> Result<()> {
    // env: пытаемся из CWD и из каталога проекта
    let _ = dotenvy::dotenv();
    if std::env::var("DATABASE_URL").is_err() {
        let manifest_env = format!("{}/.env", env!("CARGO_MANIFEST_DIR"));
        let _ = dotenvy::from_filename(&manifest_env);
    }

    tracing_subscriber::fmt()
        .with_env_filter("info")
        .compact()
        .init();
    println!("CWD: {}", std::env::current_dir()?.display());

    // БД
    let db_url = std::env::var("DATABASE_URL").expect("DATABASE_URL not set");
    let pool = Pool::<MySql>::connect(&db_url).await?;
    info!("Connected to MySQL");

    // (по желанию) автопрогон миграций:
    // sqlx::migrate!("./migrations").run(&pool).await?;
    // info!("Migrations applied");

    // JSON Q&A
    let kb =
        load_kb_json(&env::var("KNOWLEDGE_JSON").unwrap_or_else(|_| "knowledge.json".to_string()));

    let app = Router::new()
        .route("/tamtam/webhook", post(webhook))
        .route("/healthz", get(|| async { "ok" }))
        .with_state(AppState { pool, kb });

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
struct AppState {
    pool: Pool<MySql>,
    kb: KnowledgeBase,
}

/* ===================== TamTam payloads ===================== */

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
    #[serde(default)]
    user_id: Option<i64>,
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

#[derive(Debug, Clone, Copy)]
enum TTRecipientKind {
    Chat(i64),
    User(i64),
}

fn pick_recipient(msg: &TamTamMessageWrapper) -> Option<TTRecipientKind> {
    // приоритет: явный chat_id → явный user_id → sender.user_id
    if let Some(id) = msg
        .recipient
        .chat_id
        .or(msg.chat_id)
        .or(msg.chat.as_ref().and_then(|c| c.chat_id))
    {
        return Some(TTRecipientKind::Chat(id));
    }
    if let Some(uid) = msg.recipient.user_id {
        return Some(TTRecipientKind::User(uid));
    }
    if let Some(uid) = msg.sender.as_ref().and_then(|s| s.user_id) {
        return Some(TTRecipientKind::User(uid));
    }
    None
}

/* ===================== OpenAI types ======================== */

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

/* ====================== Webhook ============================ */

async fn webhook(
    axum::extract::State(state): axum::extract::State<AppState>,
    Json(update): Json<TamTamUpdate>,
) -> StatusCode {
    tokio::spawn(handle_update(state, update));
    StatusCode::OK
}

async fn handle_update(state: AppState, update: TamTamUpdate) {
    info!(
        "ids in update: recipient.chat_id={:?} recipient.user_id={:?} chat_id={:?} sender.user_id={:?}",
        msg.recipient.chat_id,
        msg.recipient.user_id,
        msg.chat_id,
        msg.sender.as_ref().and_then(|s| s.user_id),
    );
    let Some(msg) = update.message else {
        return;
    };
    let raw_text = msg.body.text.clone();
    let text = raw_text.trim().to_string();
    if text.is_empty() {
        return;
    }

    let Some(recipient) = pick_recipient(&msg) else {
        error!("No recipient found in update: {:?}", msg);
        return;
    };

    // ключ истории: чаты — chat_id, личка — -user_id
    let hist_key = match recipient {
        TTRecipientKind::Chat(id) => id,
        TTRecipientKind::User(uid) => -uid,
    };

    let system_prompt =
        env::var("SYSTEM_PROMPT").unwrap_or_else(|_| "You are a helpful assistant.".to_string());
    let history_max: i64 = env::var("HISTORY_MAX")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(12);

    // 0) Попытка ответить из Q&A (DB + JSON)
    if let Some(answer) = try_answer_from_qa(&state, &text).await {
        // Сохраним и ответим без OpenAI
        let _ = save_history_batch(
            &state.pool,
            hist_key,
            &[
                OpenAIMessage {
                    role: "user".into(),
                    content: text.clone(),
                },
                OpenAIMessage {
                    role: "assistant".into(),
                    content: answer.clone(),
                },
            ],
        )
        .await;
        let _ = send_tamtam(recipient, &answer).await;
        return;
    }

    // 1) Сбор истории
    let mut messages = vec![OpenAIMessage {
        role: "system".into(),
        content: system_prompt,
    }];
    if let Ok(h) = load_history(&state.pool, hist_key, history_max).await {
        messages.extend(h);
    }
    messages.push(OpenAIMessage {
        role: "user".into(),
        content: text.clone(),
    });

    // 2) Факты из БД
    if let Ok(facts) = load_facts(&state.pool, hist_key).await {
        if !facts.is_empty() {
            let joined = facts.join("\n- ");
            messages.insert(
                1,
                OpenAIMessage {
                    role: "system".into(),
                    content: format!("Дополнительные факты (БД):\n- {}", joined),
                },
            );
        }
    }

    // 3) Слабые совпадения из Q&A подмешиваем как контекст
    if let Some(hints) = weak_hints_from_qa(&state, &text).await {
        messages.insert(
            1,
            OpenAIMessage {
                role: "system".into(),
                content: format!("Подсказки из Q&A:\n{}", hints),
            },
        );
    }

    // 4) Модель
    let answer = match ask_openai(messages.clone()).await {
        Ok(a) if !a.trim().is_empty() => a,
        Ok(_) => "…".to_string(),
        Err(e) => {
            error!("OpenAI error: {e:?}");
            "Упс, сейчас не могу ответить. Попробуйте ещё раз.".to_string()
        }
    };

    // 5) Сохраняем и отправляем
    if let Err(e) = save_history_batch(
        &state.pool,
        hist_key,
        &[
            OpenAIMessage {
                role: "user".into(),
                content: text,
            },
            OpenAIMessage {
                role: "assistant".into(),
                content: answer.clone(),
            },
        ],
    )
    .await
    {
        error!("save_history_batch error: {e:?}");
    }
    if let Err(e) = send_tamtam(recipient, &answer).await {
        error!("send_tamtam error: {e:?}");
    }
}

/* ====================== DB helpers ======================== */

async fn load_history(pool: &Pool<MySql>, key: i64, limit: i64) -> Result<Vec<OpenAIMessage>> {
    let rows = sqlx::query(
        r#"
        SELECT role, content
        FROM chat_history
        WHERE chat_id = ?
        ORDER BY created_at DESC
        LIMIT ?
        "#,
    )
    .bind(key)
    .bind(limit)
    .fetch_all(pool)
    .await?;

    let mut items: Vec<OpenAIMessage> = rows
        .into_iter()
        .rev()
        .map(|row| {
            let role: String = row.try_get("role").unwrap_or_default();
            let content: String = row.try_get("content").unwrap_or_default();
            OpenAIMessage { role, content }
        })
        .collect();

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
            "#,
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
        "#,
    )
    .bind(key)
    .fetch_all(pool)
    .await?;

    Ok(rows
        .into_iter()
        .map(|row| row.try_get::<String, _>("content").unwrap_or_default())
        .collect())
}

/* ================== Q&A knowledge (DB + JSON) ============= */

#[derive(Clone, Default)]
struct KnowledgeBase {
    // плоский JSON Q&A
    qa: Vec<QaPair>,
}

#[derive(Clone, Deserialize, Serialize)]
struct QaPair {
    q: String,
    a: String,
    #[serde(default)]
    tags: Option<String>,
}

#[derive(Deserialize)]
struct KnowledgeJson {
    #[serde(default)]
    faq: Vec<QaPair>,
}

fn normalize(s: &str) -> String {
    let s = s.to_lowercase();
    let re = Regex::new(r"[^\p{L}\p{Nd}\s]").unwrap(); // только буквы/цифры/пробел
    re.replace_all(&s, " ")
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
}

fn jw(a: &str, b: &str) -> f64 {
    strsim::jaro_winkler(a, b) as f64
}

fn token_overlap(a: &str, b: &str) -> f64 {
    let set_a: std::collections::HashSet<_> = a.split_whitespace().collect();
    let set_b: std::collections::HashSet<_> = b.split_whitespace().collect();
    let inter = set_a.intersection(&set_b).count() as f64;
    let union = set_a.union(&set_b).count() as f64;
    if union == 0.0 { 0.0 } else { inter / union }
}

fn score(q: &str, cand: &str) -> f64 {
    // гибрид: Jaro–Winkler + Jaccard по токенам
    let n_q = normalize(q);
    let n_c = normalize(cand);
    0.65 * jw(&n_q, &n_c) + 0.35 * token_overlap(&n_q, &n_c)
}

fn load_kb_json(path: &str) -> KnowledgeBase {
    match std::fs::read_to_string(path) {
        Ok(s) => match serde_json::from_str::<KnowledgeJson>(&s) {
            Ok(k) => {
                info!("Loaded knowledge.json: {} facts", k.faq.len());
                KnowledgeBase { qa: k.faq }
            }
            Err(e) => {
                error!("knowledge.json parse error: {e:?}");
                KnowledgeBase::default()
            }
        },
        Err(e) => {
            info!("knowledge.json not found ({}): {}", path, e);
            KnowledgeBase::default()
        }
    }
}

async fn try_answer_from_qa(state: &AppState, query: &str) -> Option<String> {
    // 1) DB Q&A
    let mut best: Option<(f64, String)> = None;

    if let Ok(rows) =
        sqlx::query("SELECT question, answer FROM qa_pairs ORDER BY id DESC LIMIT 200")
            .fetch_all(&state.pool)
            .await
    {
        for row in rows {
            let q: String = row.try_get("question").unwrap_or_default();
            let a: String = row.try_get("answer").unwrap_or_default();
            let s = score(query, &q);
            if s >= 0.88 {
                return Some(a); // уверенное совпадение — отдаём сразу
            }
            if s >= 0.70 {
                if let Some((best_s, _)) = best.as_ref() {
                    if s > *best_s {
                        best = Some((s, a));
                    }
                } else {
                    best = Some((s, a));
                }
            }
        }
    }

    // 2) JSON Q&A
    for pair in &state.kb.qa {
        let s = score(query, &pair.q);
        if s >= 0.88 {
            return Some(pair.a.clone());
        }
        if s >= 0.70 {
            if let Some((best_s, _)) = best.as_ref() {
                if s > *best_s {
                    best = Some((s, pair.a.clone()));
                }
            } else {
                best = Some((s, pair.a.clone()));
            }
        }
    }

    // если лучший кандидат очень хороший (>=0.82) — можно тоже вернуть сразу
    if let Some((s, a)) = best {
        if s >= 0.82 {
            return Some(a);
        }
    }

    None
}

async fn weak_hints_from_qa(state: &AppState, query: &str) -> Option<String> {
    // вернём 1–3 намёка средней уверенности, чтобы помочь LLM
    let mut candidates: Vec<(f64, String, String)> = Vec::new();

    if let Ok(rows) =
        sqlx::query("SELECT question, answer FROM qa_pairs ORDER BY id DESC LIMIT 200")
            .fetch_all(&state.pool)
            .await
    {
        for row in rows {
            let q: String = row.try_get("question").unwrap_or_default();
            let a: String = row.try_get("answer").unwrap_or_default();
            let s = score(query, &q);
            if s >= 0.60 && s < 0.88 {
                candidates.push((s, q, a));
            }
        }
    }

    for pair in &state.kb.qa {
        let s = score(query, &pair.q);
        if s >= 0.60 && s < 0.88 {
            candidates.push((s, pair.q.clone(), pair.a.clone()));
        }
    }

    if candidates.is_empty() {
        return None;
    }

    candidates.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
    let take = std::cmp::min(3, candidates.len());
    let mut out = String::new();
    for (_, q, a) in candidates.into_iter().take(take) {
        out.push_str(&format!("- Вопрос: {}\n  Ответ: {}\n", q, a));
    }
    Some(out)
}

/* ===================== TamTam send ========================= */

use serde_json::json;

async fn send_tamtam(recipient: TTRecipientKind, text: &str) -> Result<()> {
    let token = std::env::var("TT_BOT_TOKEN").expect("TT_BOT_TOKEN not set");

    // Базовый URL
    let mut url = format!(
        "https://botapi.tamtam.chat/messages?access_token={}",
        urlencoding::encode(&token)
    );

    // ВАЖНО: user_id / chat_id идут в QUERY, не в JSON
    match recipient {
        TTRecipientKind::Chat(chat_id) => {
            url.push_str(&format!("&chat_id={}", chat_id));
        }
        TTRecipientKind::User(user_id) => {
            url.push_str(&format!("&user_id={}", user_id));
        }
    }

    // Тело — только содержимое сообщения (без recipient)
    let body = json!({ "text": text });

    let client = reqwest::Client::new();
    let res = client.post(url).json(&body).send().await?;
    if !res.status().is_success() {
        let t = res.text().await.unwrap_or_default();
        error!("TamTam send failed: {}", t);
    }
    Ok(())
}

/* ===================== OpenAI call ========================= */

async fn ask_openai(messages: Vec<OpenAIMessage>) -> Result<String> {
    let api_key = env::var("OPENAI_API_KEY").expect("OPENAI_API_KEY not set");
    let model = env::var("OPENAI_MODEL").unwrap_or_else(|_| "gpt-4o-mini".into());

    let req = OpenAIChatRequest {
        model,
        messages,
        temperature: 0.7,
    };

    let client = reqwest::Client::new();
    let res = client
        .post("https://api.openai.com/v1/chat/completions")
        .bearer_auth(api_key)
        .json(&req)
        .send()
        .await?;

    let status = res.status();
    if !status.is_success() {
        let body = res.text().await.unwrap_or_default();
        anyhow::bail!("OpenAI HTTP {}: {}", status, body);
    }

    let data: OpenAIChatResponse = res.json().await?;
    let answer = data
        .choices
        .get(0)
        .map(|c| c.message.content.clone())
        .unwrap_or_else(|| "…".to_string());

    Ok(answer)
}
