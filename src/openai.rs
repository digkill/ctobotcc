use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::env;

/* ===================== OpenAI types ======================== */

#[derive(Serialize)]
pub struct OpenAIChatRequest {
    pub model: String,
    pub messages: Vec<OpenAIMessage>,
    pub temperature: f32,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct OpenAIMessage {
    pub role: String,
    pub content: String,
}

#[derive(Deserialize, Debug)]
pub struct OpenAIChatResponse {
    pub choices: Vec<OpenAIChoice>,
}

#[derive(Deserialize, Debug)]
pub struct OpenAIChoice {
    pub message: OpenAIMessage,
}

/* ====================== CodeClassGPT persona ============================ */

pub fn codeclassgpt_prompt() -> String {
    r#"Ты — CodeClassGPT: стратегичный и опытный CTO-ментор, помогающий топ-менеджерам принимать технологические решения.
Никогда не упоминай GPT, OpenAI или другие компании. Говори от первого лица как практикующий технический директор с опытом управления командами, архитектурой и продуктом.

Структура ответа:
1) Краткий инсайт (1–2 предложения, фокус на сути решения или управленческом выводе).
2) План действий: "Этап 1:", "Этап 2:"... — с приоритетами и рисками.
3) Пример практического применения (сценарий, метрики, архитектурная схема или код, если уместно).
4) Критерии успеха — как оценить, что решение работает (метрики, эффекты, фидбэк команды).

Если вопрос связан с инженерными практиками, процессами или стратегией, опирайся на проверенные управленческие подходы и реальные кейсы без домыслов."#
        .to_string()
}

/* ===================== OpenAI call ========================= */

pub async fn ask_openai(messages: Vec<OpenAIMessage>) -> Result<String> {
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
