use anyhow::Result;
use axum::http::StatusCode;
use axum::{Json, Router, routing::{get, post}};
use chrono::Utc;
use regex::Regex;
use serde::{Deserialize, Serialize};
use sqlx::{MySql, Pool, Row};
use std::env;
use tracing::{error, info};

/* ===================== entry ===================== */

#[tokio::main]
async fn main() -> Result<()> {
    // env: .env из CWD, затем — рядом с Cargo.toml
    let _ = dotenvy::dotenv();
    if std::env::var("DATABASE_URL").is_err() {
        let manifest_env = format!("{}/.env", env!("CARGO_MANIFEST_DIR"));
        let _ = dotenvy::from_filename(&manifest_env);
    }

    tracing_subscriber::fmt().with_env_filter("info").compact().init();
    println!("CWD: {}", std::env::current_dir()?.display());

    // База ctoseo (RW)
    let db_rw = std::env::var("DATABASE_URL").expect("DATABASE_URL not set (ctoseo)");
    let pool_rw = Pool::<MySql>::connect(&db_rw).await?;
    info!("Connected to MySQL (ctoseo, RW)");

    // База codeclass (RO)
    let db_ro = std::env::var("CODECLASS_DATABASE_URL_RO")
        .expect("CODECLASS_DATABASE_URL_RO not set (codeclass, RO)");
    let pool_codeclass_ro = Pool::<MySql>::connect(&db_ro).await?;
    info!("Connected to MySQL (codeclass, RO)");

    // JSON Q&A
    let kb = load_kb_json(&env::var("KNOWLEDGE_JSON").unwrap_or_else(|_| "knowledge.json".to_string()));

    let app = Router::new()
        .route("/tamtam/webhook", post(webhook))
        .route("/healthz", get(|| async { "ok" }))
        .with_state(AppState { pool_rw, pool_codeclass_ro, kb });

    let port: u16 = env::var("PORT").ok().and_then(|s| s.parse().ok()).unwrap_or(3011);
    let addr = std::net::SocketAddr::from(([0,0,0,0], port));
    info!("Listening on {addr}");
    axum::serve(tokio::net::TcpListener::bind(addr).await?, app).await?;
    Ok(())
}

#[derive(Clone)]
struct AppState {
    pool_rw: Pool<MySql>,            // ctoseo (RW)
    pool_codeclass_ro: Pool<MySql>,  // codeclass (RO)
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
struct TamTamBody { #[serde(default)] text: String }

#[derive(Debug, Deserialize, Default)]
struct TamTamRecipient { #[serde(default)] chat_id: Option<i64>, #[serde(default)] user_id: Option<i64> }

#[derive(Debug, Deserialize, Default)]
struct TamTamChat { #[serde(default)] chat_id: Option<i64> }

#[derive(Debug, Deserialize, Default)]
struct TamTamSender { #[serde(default)] user_id: Option<i64> }

#[derive(Debug, Clone, Copy)]
enum TTRecipientKind { Chat(i64), User(i64) }

fn pick_recipient(msg: &TamTamMessageWrapper) -> Option<TTRecipientKind> {
    if let Some(id) = msg.recipient.chat_id.or(msg.chat_id).or(msg.chat.as_ref().and_then(|c| c.chat_id)) {
        return Some(TTRecipientKind::Chat(id));
    }
    if let Some(uid) = msg.recipient.user_id { return Some(TTRecipientKind::User(uid)); }
    if let Some(uid) = msg.sender.as_ref().and_then(|s| s.user_id) { return Some(TTRecipientKind::User(uid)); }
    None
}

/* ===================== OpenAI types ======================== */

#[derive(Serialize)]
struct OpenAIChatRequest { model: String, messages: Vec<OpenAIMessage>, temperature: f32 }

#[derive(Serialize, Deserialize, Debug, Clone)]
struct OpenAIMessage { role: String, content: String }

#[derive(Deserialize, Debug)]
struct OpenAIChatResponse { choices: Vec<OpenAIChoice> }

#[derive(Deserialize, Debug)]
struct OpenAIChoice { message: OpenAIMessage }

/* ====================== Webhook ============================ */

async fn webhook(
    axum::extract::State(state): axum::extract::State<AppState>,
    Json(update): Json<TamTamUpdate>,
) -> StatusCode {
    tokio::spawn(handle_update(state, update));
    StatusCode::OK
}

async fn handle_update(state: AppState, update: TamTamUpdate) {
    let Some(msg) = update.message else { return; };
    let raw_text = msg.body.text.clone();
    let text = raw_text.trim().to_string();
    if text.is_empty() { return; }

    let Some(recipient) = pick_recipient(&msg) else {
        error!("No recipient found in update: {:?}", msg);
        return;
    };

    // ключ истории: чаты — chat_id, личка — -user_id
    let hist_key = match recipient { TTRecipientKind::Chat(id) => id, TTRecipientKind::User(uid) => -uid };

    // Всегда CodeClassGPT
    let system_prompt = codeclassgpt_prompt();
    let history_max: i64 = env::var("HISTORY_MAX").ok().and_then(|s| s.parse().ok()).unwrap_or(12);

    // 0) Попытка ответить из Q&A (Только codeclass.RO → затем JSON)
    if let Some(answer) = try_answer_from_codeclass_qa(&state, &text).await
        .or_else(|| try_answer_from_json_qa(&state, &text))
    {
        let _ = save_history_batch(&state.pool_rw, hist_key, &[
            OpenAIMessage { role:"user".into(), content: text.clone() },
            OpenAIMessage { role:"assistant".into(), content: answer.clone() },
        ]).await;
        let _ = send_tamtam(recipient, &answer).await;
        return;
    }

    // 1) История (ctoseo.RW)
    let mut messages = vec![OpenAIMessage { role:"system".into(), content: system_prompt }];
    if let Ok(h) = load_history(&state.pool_rw, hist_key, history_max).await { messages.extend(h); }
    messages.push(OpenAIMessage { role:"user".into(), content: text.clone() });

    // 2) Факты (ctoseo.RW)
    if let Ok(facts) = load_facts(&state.pool_rw, hist_key).await {
        if !facts.is_empty() {
            let joined = facts.join("\n- ");
            messages.insert(1, OpenAIMessage {
                role:"system".into(),
                content: format!("Дополнительные факты (БД ctoseo):\n- {}", joined),
            });
        }
    }

    // 3) Подсказки из codeclass.RO и JSON (слабые совпадения)
    if let Some(hints) = weak_hints_from_codeclass_and_json(&state, &text).await {
        messages.insert(1, OpenAIMessage { role:"system".into(), content: format!("Подсказки из Q&A:\n{}", hints) });
    }

    // 4) Модель
    let answer = match ask_openai(messages).await {
        Ok(a) if !a.trim().is_empty() => a,
        Ok(_) => "…".to_string(),
        Err(e) => { error!("OpenAI error: {e:?}"); "CodeClassGPT: временно не могу ответить. Попробуйте ещё раз.".to_string() }
    };

    // 5) Сохраняем и отправляем (история — ctoseo.RW)
    if let Err(e) = save_history_batch(&state.pool_rw, hist_key, &[
        OpenAIMessage { role:"user".into(), content: text },
        OpenAIMessage { role:"assistant".into(), content: answer.clone() },
    ]).await { error!("save_history_batch error: {e:?}"); }
    if let Err(e) = send_tamtam(recipient, &answer).await { error!("send_tamtam error: {e:?}"); }
}

/* ====================== CodeClassGPT persona ============================ */

fn codeclassgpt_prompt() -> String {
    r#"Ты — CodeClassGPT: строгий, но заботливый преподаватель программирования от «Лапушка AI».
Никогда не упоминай GPT, OpenAI или другие компании. Говори от первого лица как опытный инженер-преподаватель.
Структура ответа:
1) Короткое резюме (1–2 предложения).
2) Шаги: "Шаг 1:", "Шаг 2:"...
3) Минимальный рабочий пример кода (если уместно).
4) Короткая проверка: как убедиться, что всё работает.
Если вопрос касается фактов из базы знаний (Q&A), придерживайся этих фактов и не выдумывай."#.to_string()
}

/* ====================== DB helpers (ctoseo.RW) ======================== */

async fn load_history(pool: &Pool<MySql>, key: i64, limit: i64) -> Result<Vec<OpenAIMessage>> {
    let rows = sqlx::query(r#"
        SELECT role, content FROM chat_history
        WHERE chat_id = ?
        ORDER BY created_at DESC
        LIMIT ?
    "#).bind(key).bind(limit).fetch_all(pool).await?;

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
        sqlx::query(r#"
            INSERT INTO chat_history (chat_id, role, content, created_at)
            VALUES (?, ?, ?, ?)
        "#)
            .bind(key)
            .bind(&m.role)
            .bind(&m.content)
            .bind(Utc::now().naive_utc())
            .execute(&mut *tx).await?;
    }
    tx.commit().await?;
    Ok(())
}

async fn load_facts(pool: &Pool<MySql>, key: i64) -> Result<Vec<String>> {
    let rows = sqlx::query(r#"
        SELECT content FROM facts
        WHERE (chat_id IS NULL) OR (chat_id = ?)
        ORDER BY created_at DESC
        LIMIT 50
    "#).bind(key).fetch_all(pool).await?;

    Ok(rows.into_iter().map(|row| row.try_get::<String, _>("content").unwrap_or_default()).collect())
}

/* ================== Q&A knowledge ============================= */

#[derive(Clone, Default)]
struct KnowledgeBase { qa: Vec<QaPair> }   // JSON-пласт (опционально)

#[derive(Clone, Deserialize, Serialize)]
struct QaPair { q: String, a: String, #[serde(default)] tags: Option<String> }

#[derive(Deserialize)]
struct KnowledgeJson { #[serde(default)] faq: Vec<QaPair> }

fn normalize(s: &str) -> String {
    let s = s.to_lowercase();
    let re = Regex::new(r"[^\p{L}\p{Nd}\s]").unwrap();
    re.replace_all(&s, " ").split_whitespace().collect::<Vec<_>>().join(" ")
}

fn jw(a: &str, b: &str) -> f64 { strsim::jaro_winkler(a, b) as f64 }

fn token_overlap(a: &str, b: &str) -> f64 {
    use std::collections::HashSet;
    let set_a: HashSet<_> = a.split_whitespace().collect();
    let set_b: HashSet<_> = b.split_whitespace().collect();
    let inter = set_a.intersection(&set_b).count() as f64;
    let union = set_a.union(&set_b).count() as f64;
    if union == 0.0 { 0.0 } else { inter/union }
}

fn score(q: &str, cand: &str) -> f64 {
    let n_q = normalize(q); let n_c = normalize(cand);
    0.65 * jw(&n_q, &n_c) + 0.35 * token_overlap(&n_q, &n_c)
}

fn load_kb_json(path: &str) -> KnowledgeBase {
    match std::fs::read_to_string(path) {
        Ok(s) => match serde_json::from_str::<KnowledgeJson>(&s) {
            Ok(k) => { info!("Loaded knowledge.json: {} facts", k.faq.len()); KnowledgeBase { qa: k.faq } }
            Err(e) => { error!("knowledge.json parse error: {e:?}"); KnowledgeBase::default() }
        },
        Err(e) => { info!("knowledge.json not found ({}): {}", path, e); KnowledgeBase::default() }
    }
}

/* ---- Q&A из codeclass (RO) ---- */

async fn try_answer_from_codeclass_qa(state: &AppState, query: &str) -> Option<String> {
    let mut best: Option<(f64, String)> = None;

    // читаем ТОЛЬКО из codeclass.RO
    if let Ok(rows) = sqlx::query("SELECT question, answer FROM qa_pairs ORDER BY id DESC LIMIT 500")
        .fetch_all(&state.pool_codeclass_ro).await
    {
        for row in rows {
            let q: String = row.try_get("question").unwrap_or_default();
            let a: String = row.try_get("answer").unwrap_or_default();
            let s = score(query, &q);
            if s >= 0.88 { return Some(a); }
            if s >= 0.70 {
                if let Some((bs,_)) = best.as_ref() { if s > *bs { best = Some((s,a)); } }
                else { best = Some((s,a)); }
            }
        }
    }

    if let Some((s,a)) = best { if s >= 0.82 { return Some(a); } }
    None
}

fn try_answer_from_json_qa(state: &AppState, query: &str) -> Option<String> {
    let mut best: Option<(f64, String)> = None;
    for pair in &state.kb.qa {
        let s = score(query, &pair.q);
        if s >= 0.88 { return Some(pair.a.clone()); }
        if s >= 0.70 {
            if let Some((bs,_)) = best.as_ref() { if s > *bs { best = Some((s, pair.a.clone())); } }
            else { best = Some((s, pair.a.clone())); }
        }
    }
    if let Some((s,a)) = best { if s >= 0.82 { return Some(a); } }
    None
}

async fn weak_hints_from_codeclass_and_json(state: &AppState, query: &str) -> Option<String> {
    let mut cand: Vec<(f64,String,String)> = Vec::new();

    if let Ok(rows) = sqlx::query("SELECT question, answer FROM qa_pairs ORDER BY id DESC LIMIT 500")
        .fetch_all(&state.pool_codeclass_ro).await
    {
        for row in rows {
            let q: String = row.try_get("question").unwrap_or_default();
            let a: String = row.try_get("answer").unwrap_or_default();
            let s = score(query, &q);
            if s >= 0.60 && s < 0.88 { cand.push((s,q,a)); }
        }
    }

    for pair in &state.kb.qa {
        let s = score(query, &pair.q);
        if s >= 0.60 && s < 0.88 { cand.push((s, pair.q.clone(), pair.a.clone())); }
    }

    if cand.is_empty() { return None; }
    cand.sort_by(|a,b| b.0.partial_cmp(&a.0).unwrap());
    let take = cand.len().min(3);
    let mut out = String::new();
    for (_, q, a) in cand.into_iter().take(take) {
        out.push_str(&format!("- Вопрос: {}\n  Ответ: {}\n", q, a));
    }
    Some(out)
}

/* ===================== TamTam send ========================= */

use serde_json::json;

async fn send_tamtam(recipient: TTRecipientKind, text: &str) -> Result<()> {
    let token = std::env::var("TT_BOT_TOKEN").expect("TT_BOT_TOKEN not set");
    let mut url = format!("https://botapi.tamtam.chat/messages?access_token={}", urlencoding::encode(&token));
    match recipient {
        TTRecipientKind::Chat(chat_id) => url.push_str(&format!("&chat_id={}", chat_id)),
        TTRecipientKind::User(user_id) => url.push_str(&format!("&user_id={}", user_id)),
    }
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
        let body = res.text().await.unwrap_or_default();
        anyhow::bail!("OpenAI HTTP {}: {}", status, body);
    }

    let data: OpenAIChatResponse = res.json().await?;
    let answer = data.choices.get(0).map(|c| c.message.content.clone()).unwrap_or_else(|| "…".to_string());
    Ok(answer)
}
