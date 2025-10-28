use anyhow::{anyhow, Result};
use serde_json::json;
use tracing::{error, info};
use super::models::TTRecipientKind;

/* ===================== TamTam send ========================= */

pub async fn send_tamtam(recipient: TTRecipientKind, text: &str) -> Result<()> {
    let token = std::env::var("TT_BOT_TOKEN").map_err(|_| anyhow!("TT_BOT_TOKEN not set"))?;
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
    info!("TamTam send request: target={:?}", recipient);
    let res = client.post(url).json(&body).send().await?;
    let status = res.status();
    let t = res.text().await.unwrap_or_default();
    if !status.is_success() {
        error!("TamTam send failed: HTTP {} body: {}", status, t);
        return Err(anyhow!(format!("TamTam HTTP {}", status)));
    }
    Ok(())
}
