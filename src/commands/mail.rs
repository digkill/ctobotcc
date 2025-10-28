use anyhow::Result;
use rand::{distributions::Alphanumeric, Rng};
use regex::Regex;
use serde::Deserialize;
use serde_json::Value;
use std::env;
use tracing::info;

/* ===================== Beget Mail ============================= */

const BEGET_API_BASE: &str = "https://api.beget.com/api/mail";
const WEBMAIL_URL: &str = "https://web.beget.email/";

pub fn allowed_domain(d: &str) -> bool {
    matches!(d, "code-class.ru" | "uchi.team")
}

#[derive(Debug, Deserialize)]
struct BegetEnvelope {
    status: String,
    #[serde(default)]
    answer: Value,
    #[serde(default)]
    error_text: Option<String>,
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
    // Log request without exposing credentials
    info!("Beget API request: method={} input={}", method, input_s);
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
    info!("Beget API response: {}", text);
    if !status.is_success() {
        anyhow::bail!(format!("Beget HTTP {}: {}", status, text));
    }

    // Case 1: JSON with status envelope
    if let Ok(envelope) = serde_json::from_str::<BegetEnvelope>(&text) {
        if envelope.status == "error" {
            let err = envelope
                .error_text
                .unwrap_or_else(|| "Unknown Beget error".to_string());
            return Err(anyhow::anyhow!(err));
        }
        // Some endpoints put nested status in `answer` → detect nested error
        if let Some(obj) = envelope.answer.as_object() {
            if let Some(st) = obj.get("status").and_then(|v| v.as_str()) {
                if st.eq_ignore_ascii_case("error") {
                    // Try to collect error_texts
                    let msg = obj
                        .get("errors")
                        .and_then(|v| v.as_array())
                        .map(|arr| {
                            let mut parts: Vec<String> = Vec::new();
                            for it in arr {
                                if let Some(t) = it.get("error_text").and_then(|x| x.as_str()) {
                                    parts.push(t.to_string());
                                }
                            }
                            if parts.is_empty() { "Beget nested error".to_string() } else { parts.join(", ") }
                        })
                        .unwrap_or_else(|| "Beget nested error".to_string());
                    return Err(anyhow::anyhow!(msg));
                }
            }
        }
        // Success, return the answer field
        return Ok(envelope.answer);
    }

    // Case 2: Non-JSON "true"
    if text.trim() == "true" {
        return Ok(Value::Bool(true));
    }

    // Case 3: Raw JSON value (for old API versions?)
    if let Ok(val) = serde_json::from_str::<Value>(&text) {
        return Ok(val);
    }

    anyhow::bail!(format!("Unexpected Beget API response: {}", text))
}

pub async fn beget_create_mailbox(domain: &str, mailbox: &str, password: &str) -> Result<bool> {
    let v = beget_call(
        "createMailbox",
        serde_json::json!({"domain": domain, "mailbox": mailbox, "mailbox_password": password}),
    )
    .await?;

    if let Some(b) = v.as_bool() {
        return Ok(b);
    }
    if let Some(obj) = v.as_object() {
        if let Some(st) = obj.get("status").and_then(|v| v.as_str()) {
            if st.eq_ignore_ascii_case("success") {
                return Ok(true);
            }
            if st.eq_ignore_ascii_case("error") {
                // Shouldn't reach here because beget_call would have errored already,
                // but keep a safe fallback.
                let msg: String = obj
                    .get("errors")
                    .and_then(|v| v.as_array())
                    .and_then(|arr| arr.get(0))
                    .and_then(|e| e.get("error_text"))
                    .and_then(|v| v.as_str())
                    .map(|s| s.to_string())
                    .unwrap_or_else(|| "Failed to create mailbox".to_string());
                return Err(anyhow::anyhow!(msg));
            }
        }
    }

    Ok(false)
}

pub async fn beget_drop_mailbox(domain: &str, mailbox: &str) -> Result<bool> {
    let v = beget_call(
        "dropMailbox",
        serde_json::json!({"domain": domain, "mailbox": mailbox}),
    )
    .await?;

    if let Some(b) = v.as_bool() {
        return Ok(b);
    }
    if let Some(obj) = v.as_object() {
        if let Some(st) = obj.get("status").and_then(|v| v.as_str()) {
            if st.eq_ignore_ascii_case("success") {
                return Ok(true);
            }
            if st.eq_ignore_ascii_case("error") {
                let msg: String = obj
                    .get("errors")
                    .and_then(|v| v.as_array())
                    .and_then(|arr| arr.get(0))
                    .and_then(|e| e.get("error_text"))
                    .and_then(|v| v.as_str())
                    .map(|s| s.to_string())
                    .unwrap_or_else(|| "Failed to drop mailbox".to_string());
                return Err(anyhow::anyhow!(msg));
            }
        }
    }
    Ok(false)
}

pub async fn beget_list_mailboxes(domain: &str) -> Result<Vec<String>> {
    let v = beget_call(
        "getMailboxList",
        serde_json::json!({"domain": domain}),
    )
    .await?;

    if let Some(arr) = v.as_array() {
        let mut out: Vec<String> = Vec::new();
        for it in arr {
            let mb = it.get("mailbox").and_then(|v| v.as_str()).unwrap_or("");
            let dm = it.get("domain").and_then(|v| v.as_str()).unwrap_or(domain);
            if !mb.is_empty() {
                out.push(format!("{}@{}", mb, dm));
            }
        }
        return Ok(out);
    }

    // Some responses may wrap into {status, answer:[...]}
    if let Some(obj) = v.as_object() {
        if let Some(ans) = obj.get("answer").and_then(|x| x.as_array()) {
            let mut out: Vec<String> = Vec::new();
            for it in ans {
                let mb = it.get("mailbox").and_then(|v| v.as_str()).unwrap_or("");
                let dm = it.get("domain").and_then(|v| v.as_str()).unwrap_or(domain);
                if !mb.is_empty() {
                    out.push(format!("{}@{}", mb, dm));
                }
            }
            return Ok(out);
        }
    }

    Ok(Vec::new())
}

async fn beget_change_mailbox_password(
    domain: &str,
    mailbox: &str,
    password: &str,
) -> Result<bool> {
    let v = beget_call(
        "changeMailboxPassword",
        serde_json::json!({"domain": domain, "mailbox": mailbox, "mailbox_password": password}),
    )
    .await?;
    Ok(match v {
        Value::Bool(b) => b,
        _ => false,
    })
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
    Create {
        email: String,
        password: Option<String>,
    },
    Passwd {
        email: String,
        password: String,
    },
    List {
        domain: String,
    },
    Drop {
        email: String,
    },
    Help,
}

fn parse_mail_command(text: &str) -> Option<MailCommand> {
    let t = text.trim();
    if !t.starts_with("/mail") {
        return None;
    }
    let parts: Vec<&str> = t.split_whitespace().collect();
    if parts.len() == 1 {
        return Some(MailCommand::Help);
    }
    match parts.get(1).copied().unwrap_or("") {
        "create" => {
            // /mail create <email> [password]
            if let Some(email) = parts.get(2) {
                let pass = parts.get(3).map(|s| s.to_string());
                return Some(MailCommand::Create {
                    email: email.to_string(),
                    password: pass,
                });
            }
        }
        "passwd" | "password" => {
            // /mail passwd <email> <newpass>
            if let (Some(email), Some(pw)) = (parts.get(2), parts.get(3)) {
                return Some(MailCommand::Passwd {
                    email: email.to_string(),
                    password: pw.to_string(),
                });
            }
        }
        "list" => {
            // /mail list <domain>
            if let Some(dom) = parts.get(2) {
                return Some(MailCommand::List {
                    domain: dom.to_string(),
                });
            }
        }
        "drop" | "delete" => {
            // /mail drop <email>
            if let Some(email) = parts.get(2) {
                return Some(MailCommand::Drop {
                    email: email.to_string(),
                });
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
    let re_pass =
        Regex::new(r"(?i)(поменять|сменить).*(парол[ьяи]).*?([\w.+-]+@[\w.-]+)[^\S\r\n]+(\S+)")
            .ok()?;
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

    // drop: "удали(ть)/удалить ... почту/ящик <email>"
    let re_drop = Regex::new(r"(?i)(удали|удалить).*(почту|ящик)[^\S\r\n]+([\w.+-]+@[\w.-]+)").ok()?;
    if let Some(c) = re_drop.captures(t) {
        let email = c.get(3).map(|m| m.as_str().to_string())?;
        return Some(MailCommand::Drop { email });
    }

    None
}

pub async fn handle_mail_commands(text: &str) -> Option<String> {
    let cmd = parse_mail_command(text).or_else(|| parse_mail_natural(text))?;
    match cmd {
        MailCommand::Help => Some("Команды почты:\n/mail create <email> [password]\n/mail passwd <email> <new_password>\n/mail list <domain>\n/mail drop <email> — домены: code-class.ru, uchi.team".into()),
        MailCommand::List { domain } => {
            if !allowed_domain(&domain) {
                return Some("Разрешены домены: code-class.ru, uchi.team".into());
            }
            match beget_call("getMailboxList", serde_json::json!({"domain": domain})).await {
                Ok(v) => {
                    if let Some(arr) = v.as_array() {
                        if arr.is_empty() {
                            return Some("Список пуст".into());
                        }
                        let mut out = String::new();
                        for it in arr {
                            let mb = it.get("mailbox").and_then(|v| v.as_str()).unwrap_or("");
                            let dm = it.get("domain").and_then(|v| v.as_str()).unwrap_or("");
                            out.push_str(&format!("{}@{}\n", mb, dm));
                        }
                        Some(out)
                    } else {
                        Some("Не удалось получить список (неверный формат ответа)".into())
                    }
                }
                Err(e) => Some(format!("Ошибка Beget: {e}")),
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
        MailCommand::Drop { email } => {
            let (local, domain) = match email.split_once('@') {
                Some((l, d)) => (l.to_string(), d.to_string()),
                None => return Some("Укажи email: имя@code-class.ru или имя@uchi.team".into()),
            };
            if !allowed_domain(&domain) {
                return Some("Разрешены домены: code-class.ru, uchi.team".into());
            }
            match beget_drop_mailbox(&domain, &local).await {
                Ok(true) => Some(format!("Почтовый ящик {} удалён.", email)),
                Ok(false) => Some("Не удалось удалить ящик (Beget вернул false)".into()),
                Err(e) => Some(format!("Ошибка Beget: {e}")),
            }
        }
    }
}
