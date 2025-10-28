use crate::commands::mail::handle_mail_commands;
use crate::commands::ro_db::handle_ro_db_queries;
use crate::db::ctoseo::{load_facts, load_history, save_history_batch};
use crate::openai::{ask_openai, codeclassgpt_prompt, OpenAIMessage};
use crate::qa::{try_answer_from_codeclass_qa, try_answer_from_json_qa, weak_hints_from_codeclass_and_json};
use crate::tamtam::{client::send_tamtam, models::{pick_recipient, TamTamUpdate, TTRecipientKind}};
use crate::AppState;
use axum::{http::StatusCode, Json};
use std::env;
use tracing::error;

/* ====================== Webhook ============================ */

pub async fn webhook(
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

    // === Команды (явные, со слешем) и естественный язык ===
    if let Some(reply) = handle_mail_commands(&text).await {
        let _ = save_history_batch(
            &state.pool_rw,
            hist_key,
            &[
                OpenAIMessage {
                    role: "user".into(),
                    content: raw_text.clone(),
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
    if let Some(reply) = handle_ro_db_queries(&state, &text).await {
        let _ = save_history_batch(
            &state.pool_rw,
            hist_key,
            &[
                OpenAIMessage {
                    role: "user".into(),
                    content: raw_text.clone(),
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

    // === Неизвестная явная команда (/) — не идём в OpenAI ===
    if text.starts_with('/') {
        let reply = "Неизвестная команда. Доступно: /mail create|passwd|list, /user, /admin, /courses, /pricing, /schedule, /lessons, /enrollments, /orders, /invoices, /partner_payments, /loan, /feedback";
        let _ = save_history_batch(
            &state.pool_rw,
            hist_key,
            &[
                OpenAIMessage {
                    role: "user".into(),
                    content: raw_text,
                },
                OpenAIMessage {
                    role: "assistant".into(),
                    content: reply.to_string(),
                },
            ],
        )
        .await;
        let _ = send_tamtam(recipient, reply).await;
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
