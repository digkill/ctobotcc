use serde::Deserialize;

/* ===================== TamTam payloads ===================== */

#[derive(Debug, Deserialize)]
pub struct TamTamUpdate {
    pub message: Option<TamTamMessageWrapper>,
}

#[derive(Debug, Deserialize)]
pub struct TamTamMessageWrapper {
    #[serde(default)]
    pub body: TamTamBody,
    #[serde(default)]
    pub recipient: TamTamRecipient,
    #[serde(default)]
    pub chat_id: Option<i64>,
    #[serde(default)]
    pub chat: Option<TamTamChat>,
    #[serde(default)]
    pub sender: Option<TamTamSender>,
}

#[derive(Debug, Deserialize, Default)]
pub struct TamTamBody {
    #[serde(default)]
    pub text: String,
}

#[derive(Debug, Deserialize, Default)]
pub struct TamTamRecipient {
    #[serde(default)]
    pub chat_id: Option<i64>,
    #[serde(default)]
    pub user_id: Option<i64>,
}

#[derive(Debug, Deserialize, Default)]
pub struct TamTamChat {
    #[serde(default)]
    pub chat_id: Option<i64>,
}

#[derive(Debug, Deserialize, Default)]
pub struct TamTamSender {
    #[serde(default)]
    pub user_id: Option<i64>,
}

#[derive(Debug, Clone, Copy)]
pub enum TTRecipientKind {
    Chat(i64),
    User(i64),
}

pub fn pick_recipient(msg: &TamTamMessageWrapper) -> Option<TTRecipientKind> {
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
