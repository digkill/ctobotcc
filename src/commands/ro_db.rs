use crate::db::codeclass::*;
use crate::AppState;
use regex::Regex;

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

pub async fn handle_ro_db_queries(state: &AppState, text: &str) -> Option<String> {
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
