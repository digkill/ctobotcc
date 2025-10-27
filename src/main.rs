use anyhow::Result;
use axum::http::StatusCode;
use axum::{
    Json, Router,
    routing::{get, post},
};
use chrono::Utc;
use regex::Regex;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use sqlx::pool::PoolOptions;
use sqlx::{
    MySql, Pool, Row,
    mysql::{MySqlConnectOptions, MySqlSslMode},
};
use std::{env, str::FromStr};
use tracing::{error, info};
use rand::{distributions::Alphanumeric, Rng};

/* ===================== entry ===================== */

#[tokio::main]
async fn main() -> Result<()> {
    // env: .env из CWD, затем — рядом с Cargo.toml
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
        load_kb_json(&env::var("KNOWLEDGE_JSON").unwrap_or_else(|_| "knowledge.json".to_string()));

    let app = Router::new()
        .route("/tamtam/webhook", post(webhook))
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
struct AppState {
    pool_rw: Pool<MySql>,           // ctoseo (RW)
    pool_codeclass_ro: Pool<MySql>, // codeclass (RO)
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

    // === Правило: если команда начинается с '/', то НЕ использовать OpenAI ===
    if text.starts_with('/') {
        // 1) Почтовые команды
        if let Some(reply) = handle_mail_commands(&text).await {
            let _ = save_history_batch(
                &state.pool_rw,
                hist_key,
                &[
                    OpenAIMessage { role: "user".into(), content: raw_text.clone() },
                    OpenAIMessage { role: "assistant".into(), content: reply.clone() },
                ],
            ).await;
            let _ = send_tamtam(recipient, &reply).await;
            return;
        }

        // 2) Быстрые RO-команды (БД codeclass)
        if let Some(reply) = handle_ro_db_queries(&state, &text).await {
            let _ = save_history_batch(
                &state.pool_rw,
                hist_key,
                &[
                    OpenAIMessage { role: "user".into(), content: raw_text.clone() },
                    OpenAIMessage { role: "assistant".into(), content: reply.clone() },
                ],
            ).await;
            let _ = send_tamtam(recipient, &reply).await;
            return;
        }

        // 3) Неизвестная команда — короткая подсказка без OpenAI
        let reply = "Неизвестная команда. Доступно: /mail create|passwd|list, /user, /admin, /courses, /pricing, /schedule, /lessons, /enrollments, /orders, /invoices, /partner_payments, /loan, /feedback";
        let _ = save_history_batch(
            &state.pool_rw,
            hist_key,
            &[
                OpenAIMessage { role: "user".into(), content: raw_text.clone() },
                OpenAIMessage { role: "assistant".into(), content: reply.to_string() },
            ],
        ).await;
        let _ = send_tamtam(recipient, reply).await;
        return;
    }

    // === Почта Beget — перехватываем сразу ===
    if let Some(reply) = handle_mail_commands(&text).await {
        let _ = save_history_batch(
            &state.pool_rw,
            hist_key,
            &[
                OpenAIMessage { role: "user".into(), content: raw_text },
                OpenAIMessage { role: "assistant".into(), content: reply.clone() },
            ],
        ).await;
        let _ = send_tamtam(recipient, &reply).await;
        return;
    }

    // === Быстрые селекты из codeclass (RO) — до Q&A/LLM ===
    if let Some(reply) = handle_ro_db_queries(&state, &text).await {
        let _ = save_history_batch(
            &state.pool_rw,
            hist_key,
            &[
                OpenAIMessage {
                    role: "user".into(),
                    content: text.clone(),
                },
                OpenAIMessage {
                    role: "assistant".into(),
                    content: reply.clone(),
                },
            ],
        )
            .await;
        let _ = send_tamtam(recipient, &reply).await;
        return;
    }


    // Persona — CodeClassGPT
    let system_prompt = codeclassgpt_prompt();
    let history_max: i64 = env::var("HISTORY_MAX")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(12);

    // 0) Попытка ответить из Q&A (codeclass.RO → JSON)
    if let Some(answer) = try_answer_from_codeclass_qa(&state, &text)
        .await
        .or_else(|| try_answer_from_json_qa(&state, &text))
    {
        let _ = save_history_batch(
            &state.pool_rw,
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

    // 1) История (ctoseo.RW)
    let mut messages = vec![OpenAIMessage {
        role: "system".into(),
        content: system_prompt,
    }];
    if let Ok(h) = load_history(&state.pool_rw, hist_key, history_max).await {
        messages.extend(h);
    }
    messages.push(OpenAIMessage {
        role: "user".into(),
        content: text.clone(),
    });

    // 2) Факты (ctoseo.RW)
    if let Ok(facts) = load_facts(&state.pool_rw, hist_key).await {
        if !facts.is_empty() {
            let joined = facts.join("\n- ");
            messages.insert(
                1,
                OpenAIMessage {
                    role: "system".into(),
                    content: format!("Дополнительные факты (БД ctoseo):\n- {}", joined),
                },
            );
        }
    }

    // 3) Подсказки из codeclass.RO и JSON (слабые совпадения)
    if let Some(hints) = weak_hints_from_codeclass_and_json(&state, &text).await {
        messages.insert(
            1,
            OpenAIMessage {
                role: "system".into(),
                content: format!("Подсказки из Q&A:\n{}", hints),
            },
        );
    }

    // 4) Модель
    let answer = match ask_openai(messages).await {
        Ok(a) if !a.trim().is_empty() => a,
        Ok(_) => "…".to_string(),
        Err(e) => {
            error!("OpenAI error: {e:?}");
            "CodeClassGPT: временно не могу ответить. Попробуйте ещё раз.".to_string()
        }
    };

    // 5) Сохраняем и отправляем (история — ctoseo.RW)
    if let Err(e) = save_history_batch(
        &state.pool_rw,
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

/* ====================== CodeClassGPT persona ============================ */

fn codeclassgpt_prompt() -> String {
    r#"Ты — CodeClassGPT: стратегичный и опытный CTO-ментор, помогающий топ-менеджерам принимать технологические решения.
Никогда не упоминай GPT, OpenAI или другие компании. Говори от первого лица как практикующий технический директор с опытом управления командами, архитектурой и продуктом.

Структура ответа:
1) Краткий инсайт (1–2 предложения, фокус на сути решения или управленческом выводе).
2) План действий: "Этап 1:", "Этап 2:"... — с приоритетами и рисками.
3) Пример практического применения (сценарий, метрики, архитектурная схема или код, если уместно).
4) Критерии успеха — как оценить, что решение работает (метрики, эффекты, фидбэк команды).

Если вопрос связан с инженерными практиками, процессами или стратегией, опирайся на проверенные управленческие подходы и реальные кейсы без домыслов."#.to_string()
}

/* ====================== DB helpers (ctoseo.RW) ======================== */

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

/* ================== Q&A knowledge (JSON) ============================= */

#[derive(Clone, Default)]
struct KnowledgeBase {
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
    let re = Regex::new(r"[^\p{L}\p{Nd}\s]").unwrap();
    re.replace_all(&s, " ")
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
}

fn jw(a: &str, b: &str) -> f64 {
    strsim::jaro_winkler(a, b) as f64
}

fn token_overlap(a: &str, b: &str) -> f64 {
    use std::collections::HashSet;
    let set_a: HashSet<_> = a.split_whitespace().collect();
    let set_b: HashSet<_> = b.split_whitespace().collect();
    let inter = set_a.intersection(&set_b).count() as f64;
    let union = set_a.union(&set_b).count() as f64;
    if union == 0.0 { 0.0 } else { inter / union }
}

fn score(q: &str, cand: &str) -> f64 {
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

/* ---- Q&A из codeclass (RO) ---- */

async fn try_answer_from_codeclass_qa(state: &AppState, query: &str) -> Option<String> {
    let mut best: Option<(f64, String)> = None;

    if let Ok(rows) =
        sqlx::query("SELECT question, answer FROM qa_pairs ORDER BY id DESC LIMIT 500")
            .fetch_all(&state.pool_codeclass_ro)
            .await
    {
        for row in rows {
            let q: String = row.try_get("question").unwrap_or_default();
            let a: String = row.try_get("answer").unwrap_or_default();
            let s = score(query, &q);
            if s >= 0.88 {
                return Some(a);
            }
            if s >= 0.70 {
                if let Some((bs, _)) = best.as_ref() {
                    if s > *bs {
                        best = Some((s, a));
                    }
                } else {
                    best = Some((s, a));
                }
            }
        }
    }

    if let Some((s, a)) = best {
        if s >= 0.82 {
            return Some(a);
        }
    }
    None
}

fn try_answer_from_json_qa(state: &AppState, query: &str) -> Option<String> {
    let mut best: Option<(f64, String)> = None;
    for pair in &state.kb.qa {
        let s = score(query, &pair.q);
        if s >= 0.88 {
            return Some(pair.a.clone());
        }
        if s >= 0.70 {
            if let Some((bs, _)) = best.as_ref() {
                if s > *bs {
                    best = Some((s, pair.a.clone()));
                }
            } else {
                best = Some((s, pair.a.clone()));
            }
        }
    }
    if let Some((s, a)) = best {
        if s >= 0.82 {
            return Some(a);
        }
    }
    None
}

/* ===================== Beget Mail ============================= */

const BEGET_API_BASE: &str = "https://api.beget.com/api/mail";
const WEBMAIL_URL: &str = "https://web.beget.email/";

fn allowed_domain(d: &str) -> bool {
    matches!(d, "code-class.ru" | "uchi.team")
}

async fn beget_call(method: &str, input: Value) -> Result<Value> {
    let login = match env::var("BEGET_LOGIN") {
        Ok(v) => v,
        Err(_) => anyhow::bail!("BEGET_LOGIN не задан. Добавь BEGET_LOGIN и BEGET_PASSWD в .env"),
    };
    let passwd = match env::var("BEGET_PASSWD") {
        Ok(v) => v,
        Err(_) => anyhow::bail!("BEGET_PASSWD не задан. Добавь BEGET_LOGIN и BEGET_PASSWD в .env"),
    };
    let input_s = serde_json::to_string(&input)?;
    let url = format!(
        "{base}/{method}?login={login}&passwd={passwd}&input_format=json&output_format=json&input_data={data}",
        base = BEGET_API_BASE,
        method = method,
        login = urlencoding::encode(&login),
        passwd = urlencoding::encode(&passwd),
        data = urlencoding::encode(&input_s),
    );
    let client = reqwest::Client::new();
    let res = client.get(url).send().await?;
    let status = res.status();
    let text = res.text().await.unwrap_or_default();
    if !status.is_success() {
        anyhow::bail!(format!("Beget HTTP {}: {}", status, text));
    }
    let val: Value = serde_json::from_str(&text).unwrap_or(Value::Bool(text.trim() == "true"));
    Ok(val)
}

async fn beget_create_mailbox(domain: &str, mailbox: &str, password: &str) -> Result<bool> {
    let v = beget_call(
        "createMailbox",
        serde_json::json!({"domain": domain, "mailbox": mailbox, "mailbox_password": password}),
    )
    .await?;
    Ok(match v { Value::Bool(b) => b, _ => false })
}

async fn beget_change_mailbox_password(domain: &str, mailbox: &str, password: &str) -> Result<bool> {
    let v = beget_call(
        "changeMailboxPassword",
        serde_json::json!({"domain": domain, "mailbox": mailbox, "mailbox_password": password}),
    )
    .await?;
    Ok(match v { Value::Bool(b) => b, _ => false })
}

fn generate_password() -> String {
    // 14 символов: буквы/цифры — максимально совместимо
    rand::thread_rng()
        .sample_iter(&Alphanumeric)
        .take(14)
        .map(char::from)
        .collect()
}

fn format_mac_setup(email: &str, password: &str) -> String {
    format!(
        "Почта создана: {email}\nПароль: {password}\n\nВход в веб-почту: {web}\n\nНастройка в Mail (macOS):\n- Входящая IMAP: imap.beget.com — порт 993 (SSL) или 143 (STARTTLS)\n- Входящая POP3: pop3.beget.com — порт 995 (SSL) или 110 (STARTTLS)\n- Исходящая SMTP: smtp.beget.com — порт 465 (SSL) или 2525 (STARTTLS/без шифр.)\n- Аутентификация: обычный пароль\n\nЛогин везде: {email}",
        email = email,
        password = password,
        web = WEBMAIL_URL,
    )
}

enum MailCommand {
    Create { email: String, password: Option<String> },
    Passwd { email: String, password: String },
    List { domain: String },
    Help,
}

fn parse_mail_command(text: &str) -> Option<MailCommand> {
    let t = text.trim();
    if !t.starts_with("/mail") { return None; }
    let parts: Vec<&str> = t.split_whitespace().collect();
    if parts.len() == 1 { return Some(MailCommand::Help); }
    match parts.get(1).copied().unwrap_or("") {
        "create" => {
            // /mail create <email> [password]
            if let Some(email) = parts.get(2) {
                let pass = parts.get(3).map(|s| s.to_string());
                return Some(MailCommand::Create { email: email.to_string(), password: pass });
            }
        }
        "passwd" | "password" => {
            // /mail passwd <email> <newpass>
            if let (Some(email), Some(pw)) = (parts.get(2), parts.get(3)) {
                return Some(MailCommand::Passwd { email: email.to_string(), password: pw.to_string() });
            }
        }
        "list" => {
            // /mail list <domain>
            if let Some(dom) = parts.get(2) {
                return Some(MailCommand::List { domain: dom.to_string() });
            }
        }
        _ => return Some(MailCommand::Help),
    }
    Some(MailCommand::Help)
}

fn parse_mail_natural(text: &str) -> Option<MailCommand> {
    let t = text.trim();
    // create: "создай/создать/сделай ... почту/ящик <email> [пароль <pw>]"
    let re_create = Regex::new(r"(?i)(создай|создать|сделай).*(почту|ящик)[^\S\r\n]+([\w.+-]+@[\w.-]+)(?:[^\S\r\n]+парол[ьяи]*[^\S\r\n]+(\S+))?").ok()?;
    if let Some(c) = re_create.captures(t) {
        let email = c.get(3).map(|m| m.as_str().to_string())?;
        let password = c.get(4).map(|m| m.as_str().to_string());
        return Some(MailCommand::Create { email, password });
    }

    // passwd: "поменять/сменить пароль <email> <новый>"
    let re_pass = Regex::new(r"(?i)(поменять|сменить).*(парол[ьяи]).*?([\w.+-]+@[\w.-]+)[^\S\r\n]+(\S+)").ok()?;
    if let Some(c) = re_pass.captures(t) {
        let email = c.get(3).map(|m| m.as_str().to_string())?;
        let password = c.get(4).map(|m| m.as_str().to_string())?;
        return Some(MailCommand::Passwd { email, password });
    }

    // list: "список (почты|ящиков) <domain>"
    let re_list = Regex::new(r"(?i)список.*(почты|ящиков)[^\S\r\n]+([\w.-]+)").ok()?;
    if let Some(c) = re_list.captures(t) {
        let domain = c.get(2).map(|m| m.as_str().to_string())?;
        return Some(MailCommand::List { domain });
    }
    None
}

async fn handle_mail_commands(text: &str) -> Option<String> {
    let cmd = parse_mail_command(text).or_else(|| parse_mail_natural(text))?;
    match cmd {
        MailCommand::Help => Some("Команды почты:\n/mail create <email> [password]\n/mail passwd <email> <new_password>\n/mail list <domain> — домены: code-class.ru, uchi.team".into()),
        MailCommand::List { domain } => {
            if !allowed_domain(&domain) {
                return Some("Разрешены домены: code-class.ru, uchi.team".into());
            }
            let v = beget_call("getMailboxList", serde_json::json!({"domain": domain})).await.ok()?;
            if let Some(arr) = v.as_array() {
                if arr.is_empty() { return Some("Список пуст".into()); }
                let mut out = String::new();
                for it in arr {
                    let mb = it.get("mailbox").and_then(|v| v.as_str()).unwrap_or("");
                    let dm = it.get("domain").and_then(|v| v.as_str()).unwrap_or("");
                    out.push_str(&format!("{}@{}\n", mb, dm));
                }
                Some(out)
            } else {
                Some("Не удалось получить список".into())
            }
        }
        MailCommand::Create { email, password } => {
            let (local, domain) = match email.split_once('@') {
                Some((l, d)) => (l.to_string(), d.to_string()),
                None => return Some("Укажи email: имя@code-class.ru или имя@uchi.team".into()),
            };
            if !allowed_domain(&domain) {
                return Some("Разрешены домены: code-class.ru, uchi.team".into());
            }
            let pw = password.unwrap_or_else(generate_password);
            match beget_create_mailbox(&domain, &local, &pw).await {
                Ok(true) => Some(format_mac_setup(&format!("{}@{}", local, domain), &pw)),
                Ok(false) => Some("Не удалось создать ящик (Beget вернул false)".into()),
                Err(e) => Some(format!("Ошибка Beget: {e}")),
            }
        }
        MailCommand::Passwd { email, password } => {
            let (local, domain) = match email.split_once('@') {
                Some((l, d)) => (l.to_string(), d.to_string()),
                None => return Some("Укажи email: имя@code-class.ru или имя@uchi.team".into()),
            };
            if !allowed_domain(&domain) {
                return Some("Разрешены домены: code-class.ru, uchi.team".into());
            }
            match beget_change_mailbox_password(&domain, &local, &password).await {
                Ok(true) => Some(format!("Пароль обновлён для {}. Вход: {}", email, WEBMAIL_URL)),
                Ok(false) => Some("Не удалось сменить пароль (Beget вернул false)".into()),
                Err(e) => Some(format!("Ошибка Beget: {e}")),
            }
        }
    }
}

async fn weak_hints_from_codeclass_and_json(state: &AppState, query: &str) -> Option<String> {
    let mut cand: Vec<(f64, String, String)> = Vec::new();

    if let Ok(rows) =
        sqlx::query("SELECT question, answer FROM qa_pairs ORDER BY id DESC LIMIT 500")
            .fetch_all(&state.pool_codeclass_ro)
            .await
    {
        for row in rows {
            let q: String = row.try_get("question").unwrap_or_default();
            let a: String = row.try_get("answer").unwrap_or_default();
            let s = score(query, &q);
            if s >= 0.60 && s < 0.88 {
                cand.push((s, q, a));
            }
        }
    }

    for pair in &state.kb.qa {
        let s = score(query, &pair.q);
        if s >= 0.60 && s < 0.88 {
            cand.push((s, pair.q.clone(), pair.a.clone()));
        }
    }

    if cand.is_empty() {
        return None;
    }
    cand.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
    let take = cand.len().min(3);
    let mut out = String::new();
    for (_, q, a) in cand.into_iter().take(take) {
        out.push_str(&format!("- Вопрос: {}\n  Ответ: {}\n", q, a));
    }
    Some(out)
}

/* ===================== codeclass (RO) quick queries ===================== */

#[derive(Debug)]
enum RoIntent {
    UserBy(String),
    AdminBy(String),
    CourseFind(Option<String>),
    PricingFor(Option<String>),
    ScheduleFor {
        course: Option<String>,
        date: Option<String>,
    },
    LessonsFor {
        course: Option<String>,
        date: Option<String>,
    },
    EnrollmentsFor(String),
    OrdersFor(String),
    InvoicesFor(String),
    PartnerPaymentsFor(String),
    LoanAppsFor(String),
    LessonFeedbackFor {
        user: Option<String>,
        lesson_id: Option<i64>,
    },
}

fn parse_ro_intent(text: &str) -> Option<RoIntent> {
    let t = text.trim().to_lowercase();
    let parts: Vec<&str> = t.split_whitespace().collect();

    // команды
    if parts.first().copied() == Some("/user") && parts.len() >= 2 {
        return Some(RoIntent::UserBy(parts[1..].join(" ")));
    }
    if parts.first().copied() == Some("/admin") && parts.len() >= 2 {
        return Some(RoIntent::AdminBy(parts[1..].join(" ")));
    }
    if parts.first().copied() == Some("/courses") {
        return Some(RoIntent::CourseFind(parts.get(1).map(|s| s.to_string())));
    }
    if parts.first().copied() == Some("/pricing") {
        return Some(RoIntent::PricingFor(parts.get(1).map(|s| s.to_string())));
    }
    if parts.first().copied() == Some("/schedule") {
        let course = parts.get(1).map(|s| s.to_string());
        let date = parts.get(2).map(|s| s.to_string());
        return Some(RoIntent::ScheduleFor { course, date });
    }
    if parts.first().copied() == Some("/lessons") {
        let course = parts.get(1).map(|s| s.to_string());
        let date = parts.get(2).map(|s| s.to_string());
        return Some(RoIntent::LessonsFor { course, date });
    }
    if parts.first().copied() == Some("/enrollments") && parts.len() >= 2 {
        return Some(RoIntent::EnrollmentsFor(parts[1..].join(" ")));
    }
    if parts.first().copied() == Some("/orders") && parts.len() >= 2 {
        return Some(RoIntent::OrdersFor(parts[1..].join(" ")));
    }
    if parts.first().copied() == Some("/invoices") && parts.len() >= 2 {
        return Some(RoIntent::InvoicesFor(parts[1..].join(" ")));
    }
    if parts.first().copied() == Some("/partner_payments") && parts.len() >= 2 {
        return Some(RoIntent::PartnerPaymentsFor(parts[1..].join(" ")));
    }
    if parts.first().copied() == Some("/loan") && parts.len() >= 2 {
        return Some(RoIntent::LoanAppsFor(parts[1..].join(" ")));
    }
    if parts.first().copied() == Some("/feedback") {
        if parts.get(1) == Some(&"lesson") {
            if let Some(id) = parts.get(2).and_then(|s| s.parse::<i64>().ok()) {
                return Some(RoIntent::LessonFeedbackFor {
                    user: None,
                    lesson_id: Some(id),
                });
            }
        } else if parts.get(1) == Some(&"user") && parts.len() >= 3 {
            return Some(RoIntent::LessonFeedbackFor {
                user: Some(parts[2..].join(" ")),
                lesson_id: None,
            });
        }
    }

    // естественные фразы
    let re_user = Regex::new(r"(?i)(посмотри|найди).*(ученика|пользователя)\s+(.+)$").unwrap();
    if let Some(caps) = re_user.captures(&t) {
        if let Some(q) = caps.get(3) {
            let q = q.as_str().trim().to_string();
            if !q.is_empty() {
                return Some(RoIntent::UserBy(q));
            }
        }
    }
    if t.starts_with("расписание на ") {
        let date = t.replace("расписание на ", "").trim().to_string();
        return Some(RoIntent::ScheduleFor {
            course: None,
            date: Some(date),
        });
    }
    if t.starts_with("цены") || t.contains("прайс") {
        return Some(RoIntent::PricingFor(None));
    }
    None
}

async fn handle_ro_db_queries(state: &AppState, text: &str) -> Option<String> {
    let intent = parse_ro_intent(text)?;
    let out = match intent {
        RoIntent::UserBy(q) => query_user(&state.pool_codeclass_ro, &q).await,
        RoIntent::AdminBy(q) => query_admin(&state.pool_codeclass_ro, &q).await,
        RoIntent::CourseFind(q) => query_courses(&state.pool_codeclass_ro, q.as_deref()).await,
        RoIntent::PricingFor(q) => query_pricing(&state.pool_codeclass_ro, q.as_deref()).await,
        RoIntent::ScheduleFor { course, date } => {
            query_schedule(&state.pool_codeclass_ro, course.as_deref(), date.as_deref()).await
        }
        RoIntent::LessonsFor { course, date } => {
            query_lessons(&state.pool_codeclass_ro, course.as_deref(), date.as_deref()).await
        }
        RoIntent::EnrollmentsFor(q) => query_enrollments(&state.pool_codeclass_ro, &q).await,
        RoIntent::OrdersFor(q) => query_orders(&state.pool_codeclass_ro, &q).await,
        RoIntent::InvoicesFor(q) => query_invoices(&state.pool_codeclass_ro, &q).await,
        RoIntent::PartnerPaymentsFor(q) => {
            query_partner_payments(&state.pool_codeclass_ro, &q).await
        }
        RoIntent::LoanAppsFor(q) => query_loan_apps(&state.pool_codeclass_ro, &q).await,
        RoIntent::LessonFeedbackFor { user, lesson_id } => {
            query_lesson_feedback(&state.pool_codeclass_ro, user.as_deref(), lesson_id).await
        }
    }
        .unwrap_or_else(|e| format!("Ошибка запроса: {e:?}"));

    Some(if out.is_empty() {
        "Ничего не найдено.".into()
    } else {
        out
    })
}

/* ===================== Конкретные SELECT-запросы (RO) ===================== */

async fn query_user(pool: &Pool<MySql>, q: &str) -> Result<String> {
    // users: name, last_name, username, email
    let rows = sqlx::query(
        r#"
        SELECT CAST(id AS SIGNED) AS id,
               CONCAT_WS(' ', name, last_name) AS full_name,
               username, email
        FROM users
        WHERE email LIKE ? OR username LIKE ?
           OR name LIKE ? OR last_name LIKE ?
        ORDER BY id DESC
        LIMIT 10
        "#,
    )
        .bind(format!("%{}%", q))
        .bind(format!("%{}%", q))
        .bind(format!("%{}%", q))
        .bind(format!("%{}%", q))
        .fetch_all(pool)
        .await?;

    let mut out = String::new();
    for r in rows {
        let id: i64 = r.try_get("id")?;
        let name: String = r.try_get("full_name").unwrap_or_default();
        let username: String = r.try_get("username").unwrap_or_default();
        let email: String = r.try_get("email").unwrap_or_default();
        out.push_str(&format!("ID:{id} • {name} • @{username} • {email}\n"));
    }
    Ok(out)
}

async fn query_admin(pool: &Pool<MySql>, q: &str) -> Result<String> {
    let rows = sqlx::query(
        r#"
        SELECT CAST(id AS SIGNED) AS id, name, username, email
        FROM admins
        WHERE email LIKE ? OR name LIKE ? OR username LIKE ?
        ORDER BY id DESC
        LIMIT 10
        "#,
    )
        .bind(format!("%{}%", q))
        .bind(format!("%{}%", q))
        .bind(format!("%{}%", q))
        .fetch_all(pool)
        .await?;

    let mut out = String::new();
    for r in rows {
        let id: i64 = r.try_get("id")?;
        let name: String = r.try_get("name").unwrap_or_default();
        let username: String = r.try_get("username").unwrap_or_default();
        let email: String = r.try_get("email").unwrap_or_default();
        out.push_str(&format!("ID:{id} • {name} • @{username} • {email}\n"));
    }
    Ok(out)
}

async fn query_courses(pool: &Pool<MySql>, q: Option<&str>) -> Result<String> {
    let rows = if let Some(k) = q {
        sqlx::query(
            r#"
            SELECT CAST(id AS SIGNED) AS id, title
            FROM courses
            WHERE title LIKE ?
            ORDER BY id DESC
            LIMIT 10
            "#,
        )
            .bind(format!("%{}%", k))
            .fetch_all(pool)
            .await?
    } else {
        sqlx::query(r#"SELECT CAST(id AS SIGNED) AS id, title FROM courses ORDER BY id DESC LIMIT 10"#)
            .fetch_all(pool)
            .await?
    };
    let mut out = String::new();
    for r in rows {
        let id: i64 = r.try_get("id")?;
        let title: String = r.try_get("title")?;
        out.push_str(&format!("ID:{id} • {title}\n"));
    }
    Ok(out)
}

async fn query_pricing(pool: &Pool<MySql>, course_like: Option<&str>) -> Result<String> {
    // pricing: title, count_lesson, price + join courses.title
    let rows = if let Some(k) = course_like {
        sqlx::query(
            r#"
            SELECT c.title AS course,
                   p.title AS plan_title,
                   p.count_lesson,
                   p.price
            FROM pricing p
            LEFT JOIN courses c ON c.id = p.course_id
            WHERE c.title LIKE ?
            ORDER BY p.course_id DESC, p.price ASC
            LIMIT 10
            "#,
        )
            .bind(format!("%{}%", k))
            .fetch_all(pool)
            .await?
    } else {
        sqlx::query(
            r#"
            SELECT c.title AS course,
                   p.title AS plan_title,
                   p.count_lesson,
                   p.price
            FROM pricing p
            LEFT JOIN courses c ON c.id = p.course_id
            ORDER BY p.course_id DESC, p.price ASC
            LIMIT 10
            "#,
        )
            .fetch_all(pool)
            .await?
    };

    let mut out = String::new();
    for r in rows {
        let course: String = r.try_get("course").unwrap_or_default();
        let plan: String = r.try_get("plan_title").unwrap_or_default();
        let count_lesson: i64 = r.try_get("count_lesson").unwrap_or(0);
        let price: i64 = r.try_get("price").unwrap_or(0);
        out.push_str(&format!(
            "{course}: {plan} — {count_lesson} занятий • {price} ₽\n"
        ));
    }
    Ok(out)
}

async fn query_schedule(
    pool: &Pool<MySql>,
    course_like: Option<&str>,
    date: Option<&str>,
) -> Result<String> {
    // schedules → groups → courses, дата: COALESCE(start_at, date_start)
    let mut q = String::from(
        r#"
        SELECT CAST(s.id AS SIGNED) AS id,
               c.title AS course,
               DATE_FORMAT(COALESCE(s.start_at, s.date_start), '%Y-%m-%d %H:%i') AS dt,
               g.title AS group_name
        FROM schedules s
        LEFT JOIN groups g ON g.id = s.group_id
        LEFT JOIN courses c ON c.id = g.course_id
        WHERE 1=1
        "#,
    );
    let mut binds: Vec<String> = vec![];
    if let Some(k) = course_like {
        q.push_str(" AND (c.title LIKE ?)");
        binds.push(format!("%{}%", k));
    }
    if let Some(d) = date {
        q.push_str(" AND DATE(COALESCE(s.start_at, s.date_start)) = ?");
        binds.push(d.to_string());
    }
    q.push_str(" ORDER BY COALESCE(s.start_at, s.date_start) ASC LIMIT 10");

    let mut query = sqlx::query(&q);
    for b in binds {
        query = query.bind(b);
    }
    let rows = query.fetch_all(pool).await?;

    let mut out = String::new();
    for r in rows {
        let id: i64 = r.try_get("id")?;
        let course: String = r.try_get("course").unwrap_or_default();
        let dt: String = r.try_get("dt").unwrap_or_default();
        let group: String = r.try_get("group_name").unwrap_or_default();
        out.push_str(&format!("#{id} • {dt} • {course} • {group}\n"));
    }
    Ok(out)
}

async fn query_lessons(
    pool: &Pool<MySql>,
    course_like: Option<&str>,
    date: Option<&str>,
) -> Result<String> {
    // lessons → course_lesson → courses
    let mut q = String::from(
        r#"
        SELECT CAST(l.id AS SIGNED) AS id,
               l.title AS lesson,
               c.title AS course,
               DATE_FORMAT(l.created_at, '%Y-%m-%d %H:%i') AS dt
        FROM lessons l
        LEFT JOIN course_lesson cl ON cl.lesson_id = l.id
        LEFT JOIN courses c ON c.id = cl.course_id
        WHERE 1=1
        "#,
    );
    let mut binds: Vec<String> = vec![];
    if let Some(k) = course_like {
        q.push_str(" AND (c.title LIKE ?)");
        binds.push(format!("%{}%", k));
    }
    if let Some(d) = date {
        // если в lessons есть starts_at — подменить на него здесь
        q.push_str(" AND DATE(l.created_at) = ?");
        binds.push(d.to_string());
    }
    q.push_str(" ORDER BY l.created_at ASC LIMIT 10");

    let mut query = sqlx::query(&q);
    for b in binds {
        query = query.bind(b);
    }
    let rows = query.fetch_all(pool).await?;

    let mut out = String::new();
    for r in rows {
        let id: i64 = r.try_get("id")?;
        let course: String = r.try_get("course").unwrap_or_default();
        let lesson: String = r.try_get("lesson").unwrap_or_default();
        let dt: String = r.try_get("dt").unwrap_or_default();
        out.push_str(&format!("#{id} • {dt} • {course} — {lesson}\n"));
    }
    Ok(out)
}

async fn find_user_id(pool: &Pool<MySql>, q: &str) -> Result<Option<i64>> {
    let row = sqlx::query(
        r#"
        SELECT CAST(id AS SIGNED) AS id
        FROM users
        WHERE email LIKE ? OR phone LIKE ? OR username LIKE ?
           OR name LIKE ? OR last_name LIKE ?
        ORDER BY id DESC LIMIT 1
        "#,
    )
        .bind(format!("%{}%", q))
        .bind(format!("%{}%", q))
        .bind(format!("%{}%", q))
        .bind(format!("%{}%", q))
        .bind(format!("%{}%", q))
        .fetch_optional(pool)
        .await?;

    Ok(row.and_then(|r| r.try_get::<i64, _>("id").ok()))
}

async fn query_enrollments(pool: &Pool<MySql>, user_q: &str) -> Result<String> {
    if let Some(uid) = find_user_id(pool, user_q).await? {
        let rows = sqlx::query(
            r#"
            SELECT CAST(e.id AS SIGNED) AS id,
                   c.title AS course,
                   e.status
            FROM enrollments e
            LEFT JOIN courses c ON c.id = e.course_id
            WHERE e.user_id = ?
            ORDER BY e.id DESC
            LIMIT 10
            "#,
        )
            .bind(uid)
            .fetch_all(pool)
            .await?;

        let mut out = format!("Записи для user_id={uid}:\n");
        for r in rows {
            let id: i64 = r.try_get("id")?;
            let course: String = r.try_get("course").unwrap_or_default();
            let status: String = r.try_get("status").unwrap_or_default();
            out.push_str(&format!("#{id} • {course} • {status}\n"));
        }
        return Ok(out);
    }
    Ok("Пользователь не найден.".into())
}

async fn query_orders(pool: &Pool<MySql>, user_q: &str) -> Result<String> {
    if let Some(uid) = find_user_id(pool, user_q).await? {
        let rows = sqlx::query(
            r#"
            SELECT CAST(o.id AS SIGNED) AS id,
                   o.total_amount,
                   o.status,
                   DATE_FORMAT(o.created_at, '%Y-%m-%d') AS dt
            FROM `order` o
            WHERE EXISTS (
                SELECT 1 FROM invoices i
                WHERE i.user_id = ? AND i.id = o.invoice_id
            )
            ORDER BY o.id DESC
            LIMIT 10
            "#,
        )
            .bind(uid)
            .fetch_all(pool)
            .await?;

        let mut out = format!("Заказы для user_id={uid} (через invoices):\n");
        for r in rows {
            let id: i64 = r.try_get("id")?;
            let total: f64 = r.try_get("total_amount").unwrap_or(0.0);
            let status: i64 = r.try_get("status").unwrap_or(0);
            let dt: String = r.try_get("dt").unwrap_or_default();
            out.push_str(&format!("#{id} • {dt} • {total} • status={status}\n"));
        }
        return Ok(out);
    }
    Ok("Пользователь не найден.".into())
}

async fn query_invoices(pool: &Pool<MySql>, user_q: &str) -> Result<String> {
    if let Some(uid) = find_user_id(pool, user_q).await? {
        let rows = sqlx::query(
            r#"
            SELECT CAST(id AS SIGNED) AS id,
                   pay_amount,
                   status,
                   DATE_FORMAT(created_at, '%Y-%m-%d') AS dt
            FROM invoices
            WHERE user_id = ?
            ORDER BY id DESC
            LIMIT 10
            "#,
        )
            .bind(uid)
            .fetch_all(pool)
            .await?;

        let mut out = format!("Счета для user_id={uid}:\n");
        for r in rows {
            let id: i64 = r.try_get("id")?;
            let amount: f64 = r.try_get("pay_amount").unwrap_or(0.0);
            let status: i64 = r.try_get("status").unwrap_or(0);
            let dt: String = r.try_get("dt").unwrap_or_default();
            out.push_str(&format!("#{id} • {dt} • {amount} • status:{status}\n"));
        }
        return Ok(out);
    }
    Ok("Пользователь не найден.".into())
}

async fn query_partner_payments(pool: &Pool<MySql>, _user_q: &str) -> Result<String> {
    // В таблице payments_partners нет user_id — отдаем последние записи
    let rows = sqlx::query(
        r#"
        SELECT CAST(id AS SIGNED) AS id, name, total_payable,
               DATE_FORMAT(period_from, '%Y-%m-%d') AS dfrom,
               DATE_FORMAT(period_to, '%Y-%m-%d')   AS dto
        FROM payments_partners
        ORDER BY period_to DESC
        LIMIT 10
        "#,
    )
        .fetch_all(pool)
        .await?;

    let mut out = String::from("Последние партнёрские выплаты:\n");
    for r in rows {
        let id: i64 = r.try_get("id")?;
        let name: String = r.try_get("name").unwrap_or_default();
        let total: f64 = r.try_get("total_payable").unwrap_or(0.0);
        let dfrom: String = r.try_get("dfrom").unwrap_or_default();
        let dto: String = r.try_get("dto").unwrap_or_default();
        out.push_str(&format!("#{id} • {dfrom}..{dto} • {name} • {total}\n"));
    }
    Ok(out)
}

pub async fn query_loan_apps(pool: &Pool<MySql>, user_q: &str) -> Result<String> {
    let q = user_q.trim();
    if q.is_empty() {
        return Ok("Укажи запрос: имя/телефон/email/номер заказа или id.".into());
    }

    // Если это чисто число — считаем, что хотят точный поиск по id.
    let numeric_id = q.parse::<i64>().ok();

    let rows = if let Some(id) = numeric_id {
        sqlx::query(
            r#"
            SELECT
              CAST(id AS SIGNED)            AS id,
              CAST(franchise_id AS SIGNED)  AS franchise_id,
              CAST(school_id AS SIGNED)     AS school_id,
              first_name, last_name, middle_name,
              client_phone, client_email,
              order_id, tinkoff_order_id, link,
              is_course, is_test,
              status, amount,
              DATE_FORMAT(created_at, '%Y-%m-%d %H:%i') AS dt,
              DATE_FORMAT(confirm_date, '%Y-%m-%d %H:%i') AS cdt
            FROM loan_application
            WHERE id = ?
            ORDER BY created_at DESC
            LIMIT 20
            "#,
        )
            .bind(id)
            .fetch_all(pool)
            .await?
    } else {
        // Текстовый поиск по основным полям
        sqlx::query(
            r#"
            SELECT
              CAST(id AS SIGNED)            AS id,
              CAST(franchise_id AS SIGNED)  AS franchise_id,
              CAST(school_id AS SIGNED)     AS school_id,
              first_name, last_name, middle_name,
              client_phone, client_email,
              order_id, tinkoff_order_id, link,
              is_course, is_test,
              status, amount,
              DATE_FORMAT(created_at, '%Y-%m-%d %H:%i') AS dt,
              DATE_FORMAT(confirm_date, '%Y-%m-%d %H:%i') AS cdt
            FROM loan_application
            WHERE
                first_name       LIKE CONCAT('%', ?, '%')
             OR last_name        LIKE CONCAT('%', ?, '%')
             OR middle_name      LIKE CONCAT('%', ?, '%')
             OR client_phone     LIKE CONCAT('%', ?, '%')
             OR client_email     LIKE CONCAT('%', ?, '%')
             OR order_id         LIKE CONCAT('%', ?, '%')
             OR tinkoff_order_id LIKE CONCAT('%', ?, '%')
            ORDER BY created_at DESC
            LIMIT 20
            "#,
        )
            .bind(q) // first_name
            .bind(q) // last_name
            .bind(q) // middle_name
            .bind(q) // client_phone
            .bind(q) // client_email
            .bind(q) // order_id
            .bind(q) // tinkoff_order_id
            .fetch_all(pool)
            .await?
    };

    if rows.is_empty() {
        return Ok("Ничего не нашёл по запросу.".into());
    }

    let mut out = String::new();
    for r in rows {
        let id: i64 = r.try_get("id")?;
        let fid: Option<i64> = r.try_get("franchise_id").ok();
        let sid: Option<i64> = r.try_get("school_id").ok();

        let first_name: String = r.try_get("first_name").unwrap_or_default();
        let last_name: String = r.try_get("last_name").unwrap_or_default();
        let middle_name: String = r.try_get("middle_name").unwrap_or_default();

        let phone: String = r.try_get("client_phone").unwrap_or_default();
        let email: String = r.try_get("client_email").unwrap_or_default();

        let order_id: String = r.try_get("order_id").unwrap_or_default();
        let tinkoff_order_id: String = r.try_get("tinkoff_order_id").unwrap_or_default();
        let link: String = r.try_get("link").unwrap_or_default();

        let is_course: i64 = r.try_get("is_course").unwrap_or(0);
        let is_test: i64 = r.try_get("is_test").unwrap_or(0);

        let status: i64 = r.try_get("status").unwrap_or(0);
        let amount: f64 = r.try_get("amount").unwrap_or(0.0);

        let dt: String = r.try_get("dt").unwrap_or_default();
        let cdt: String = r.try_get("cdt").unwrap_or_default();

        let fio = [
            last_name.as_str(),
            first_name.as_str(),
            middle_name.as_str(),
        ]
            .iter()
            .filter(|s| !s.is_empty())
            .cloned()
            .collect::<Vec<_>>()
            .join(" ");

        out.push_str(&format!(
            "#{id} • {dt} • {fio}\n\
             └ phone:{phone} • email:{email}\n\
             └ amount:{amount} • status:{status} • course:{is_course} • test:{is_test}\n\
             └ order:{order_id} • tinkoff:{tinkoff_order_id}\n"
        ));
        if let Some(fid) = fid {
            out.push_str(&format!("└ franchise:{fid}"));
            if let Some(sid) = sid {
                out.push_str(&format!(" • school:{sid}"));
            }
            out.push('\n');
        }
        if !link.is_empty() {
            out.push_str(&format!("└ link:{link}\n"));
        }
        if !cdt.is_empty() {
            out.push_str(&format!("└ confirmed:{cdt}\n"));
        }
        out.push('\n');
    }

    Ok(out)
}

async fn query_lesson_feedback(
    pool: &Pool<MySql>,
    user_q: Option<&str>,
    lesson_id: Option<i64>,
) -> Result<String> {
    if let Some(id) = lesson_id {
        let rows = sqlx::query(
            r#"
            SELECT CAST(id AS SIGNED) AS id,
                   CAST(lesson_id AS SIGNED) AS lesson_id,
                   CAST(user_id  AS SIGNED)  AS user_id,
                   rating,
                   LEFT(comment, 140) AS c,
                   DATE_FORMAT(created_at, '%Y-%m-%d') AS dt
            FROM lesson_feedback
            WHERE lesson_id = ?
            ORDER BY created_at DESC
            LIMIT 10
            "#,
        )
            .bind(id)
            .fetch_all(pool)
            .await?;

        let mut out = format!("Фидбек по уроку #{id}:\n");
        for r in rows {
            let fid: i64 = r.try_get("id")?;
            let uid: i64 = r.try_get("user_id")?;
            let rating: i64 = r.try_get("rating").unwrap_or(0);
            let c: String = r.try_get("c").unwrap_or_default();
            let dt: String = r.try_get("dt").unwrap_or_default();
            out.push_str(&format!("#{fid} • {dt} • user:{uid} • {rating}/5 • {c}\n"));
        }
        return Ok(out);
    } else if let Some(uq) = user_q {
        if let Some(uid) = find_user_id(pool, uq).await? {
            let rows = sqlx::query(
                r#"
                SELECT CAST(id AS SIGNED) AS id,
                       CAST(lesson_id AS SIGNED) AS lesson_id,
                       rating,
                       LEFT(comment, 140) AS c,
                       DATE_FORMAT(created_at, '%Y-%m-%d') AS dt
                FROM lesson_feedback
                WHERE user_id = ?
                ORDER BY created_at DESC
                LIMIT 10
                "#,
            )
                .bind(uid)
                .fetch_all(pool)
                .await?;

            let mut out = format!("Фидбек пользователя user_id={uid}:\n");
            for r in rows {
                let fid: i64 = r.try_get("id")?;
                let lid: i64 = r.try_get("lesson_id")?;
                let rating: i64 = r.try_get("rating").unwrap_or(0);
                let c: String = r.try_get("c").unwrap_or_default();
                let dt: String = r.try_get("dt").unwrap_or_default();
                out.push_str(&format!(
                    "#{fid} • {dt} • lesson:{lid} • {rating}/5 • {c}\n"
                ));
            }
            return Ok(out);
        }
        return Ok("Пользователь не найден.".into());
    }
    Ok("Укажи /feedback user <запрос> или /feedback lesson <id>.".into())
}

/* ===================== TamTam send ========================= */

use serde_json::json;

async fn send_tamtam(recipient: TTRecipientKind, text: &str) -> Result<()> {
    let token = std::env::var("TT_BOT_TOKEN").expect("TT_BOT_TOKEN not set");
    let mut url = format!(
        "https://botapi.tamtam.chat/messages?access_token={}",
        urlencoding::encode(&token)
    );
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
