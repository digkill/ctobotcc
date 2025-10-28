use crate::db::codeclass;
use crate::AppState;
use anyhow::Result;

// Build dynamic DB context snippets based on the question content.
// Returns a formatted string to inject as a system message.
pub async fn build_dynamic_context(state: &AppState, question: &str) -> Option<String> {
    let q_lower = question.to_lowercase();
    let mut sections: Vec<String> = Vec::new();

    // Helper to safely push non-empty results
    async fn push_if_ok<Fut>(sections: &mut Vec<String>, title: &str, fut: Fut)
    where
        Fut: std::future::Future<Output = Result<String>>,
    {
        if let Ok(txt) = fut.await {
            let t = txt.trim();
            if !t.is_empty() {
                sections.push(format!("{}\n{}", title, t));
            }
        }
    }

    // Very simple heuristic keywords (RU/EN)
    if q_lower.contains("user") || q_lower.contains("польз") || q_lower.contains("ученик") {
        let key = extract_key(question);
        push_if_ok(&mut sections, "БД codeclass: пользователи", codeclass::query_user(&state.pool_codeclass_ro, &key)).await;
    }
    if q_lower.contains("admin") || q_lower.contains("админ") || q_lower.contains("препод") || q_lower.contains("учитель") {
        let key = extract_key(question);
        push_if_ok(&mut sections, "БД codeclass: администраторы/преподаватели", codeclass::query_admin(&state.pool_codeclass_ro, &key)).await;
    }
    if q_lower.contains("course") || q_lower.contains("курс") {
        let key_opt = some_if_not_empty(extract_key(question));
        push_if_ok(&mut sections, "БД codeclass: курсы", codeclass::query_courses(&state.pool_codeclass_ro, key_opt.as_deref())).await;
    }
    if q_lower.contains("price") || q_lower.contains("цен") || q_lower.contains("стоим") || q_lower.contains("pricing") {
        let key_opt = some_if_not_empty(extract_key(question));
        push_if_ok(&mut sections, "БД codeclass: цены/тарифы", codeclass::query_pricing(&state.pool_codeclass_ro, key_opt.as_deref())).await;
    }
    if q_lower.contains("schedule") || q_lower.contains("распис") {
        let key = extract_key(question);
        push_if_ok(&mut sections, "БД codeclass: расписание", codeclass::query_schedule(&state.pool_codeclass_ro, Some(key.as_str()), None)).await;
    }
    if q_lower.contains("lesson") || q_lower.contains("урок") {
        let key = extract_key(question);
        push_if_ok(&mut sections, "БД codeclass: уроки", codeclass::query_lessons(&state.pool_codeclass_ro, Some(key.as_str()), None)).await;
    }
    if q_lower.contains("enroll") || q_lower.contains("запис") {
        let key = extract_key(question);
        push_if_ok(&mut sections, "БД codeclass: записи на курс", codeclass::query_enrollments(&state.pool_codeclass_ro, &key)).await;
    }
    if q_lower.contains("order") || q_lower.contains("заказ") {
        let key = extract_key(question);
        push_if_ok(&mut sections, "БД codeclass: заказы", codeclass::query_orders(&state.pool_codeclass_ro, &key)).await;
    }
    if q_lower.contains("invoice") || q_lower.contains("счёт") || q_lower.contains("счет") {
        let key = extract_key(question);
        push_if_ok(&mut sections, "БД codeclass: счета", codeclass::query_invoices(&state.pool_codeclass_ro, &key)).await;
    }
    if q_lower.contains("loan") || q_lower.contains("кредит") || q_lower.contains("рассроч") {
        let key = extract_key(question);
        push_if_ok(&mut sections, "БД codeclass: заявки на рассрочку", codeclass::query_loan_apps(&state.pool_codeclass_ro, &key)).await;
    }
    if q_lower.contains("feedback") || q_lower.contains("отзыв") {
        // Without concrete IDs we can only show recent feedback (None, None)
        push_if_ok(&mut sections, "БД codeclass: обратная связь по урокам", codeclass::query_lesson_feedback(&state.pool_codeclass_ro, None, None)).await;
    }

    if sections.is_empty() { None } else { Some(format!("Динамический контекст из БД (codeclass):\n{}", sections.join("\n\n"))) }
}

fn extract_key(question: &str) -> String {
    // naive: take longest word-like token or quoted phrase
    if let Some(q) = extract_quoted(question) { return q; }
    let mut best = "".to_string();
    for w in question.split_whitespace() {
        let t = w.trim_matches(|c: char| !c.is_alphanumeric() && c != '@' && c != '.' && c != '_' && c != '-');
        if t.len() > best.len() { best = t.to_string(); }
    }
    best
}

fn extract_quoted(s: &str) -> Option<String> {
    let bytes = s.as_bytes();
    let mut start: Option<usize> = None;
    for (i, &b) in bytes.iter().enumerate() {
        if b == b'"' || b == b'\'' {
            if let Some(st) = start { return Some(s[st+1..i].to_string()); } else { start = Some(i); }
        }
    }
    None
}

fn some_if_not_empty(s: String) -> Option<String> { if s.trim().is_empty() { None } else { Some(s) } }


