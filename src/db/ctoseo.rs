use crate::openai::OpenAIMessage;
use anyhow::Result;
use chrono::Utc;
use sqlx::{MySql, Pool, Row};

/* ====================== DB helpers (ctoseo.RW) ======================== */

pub async fn load_history(pool: &Pool<MySql>, key: i64, limit: i64) -> Result<Vec<OpenAIMessage>> {
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

pub async fn save_history_batch(pool: &Pool<MySql>, key: i64, msgs: &[OpenAIMessage]) -> Result<()> {
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

pub async fn load_facts(pool: &Pool<MySql>, key: i64) -> Result<Vec<String>> {
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
