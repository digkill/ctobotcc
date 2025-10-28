use crate::AppState;
use regex::Regex;
use serde::{Deserialize, Serialize};
use sqlx::Row;
use tracing::{error, info};

/* ================== Q&A knowledge (JSON) ============================= */

#[derive(Clone, Default)]
pub struct KnowledgeBase {
    pub qa: Vec<QaPair>,
}

#[derive(Clone, Deserialize, Serialize)]
pub struct QaPair {
    pub q: String,
    pub a: String,
    #[serde(default)]
    pub tags: Option<String>,
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
    if union == 0.0 {
        0.0
    } else {
        inter / union
    }
}

fn score(q: &str, cand: &str) -> f64 {
    let n_q = normalize(q);
    let n_c = normalize(cand);
    0.65 * jw(&n_q, &n_c) + 0.35 * token_overlap(&n_q, &n_c)
}

pub fn load_kb_json(path: &str) -> KnowledgeBase {
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

pub async fn try_answer_from_codeclass_qa(state: &AppState, query: &str) -> Option<String> {
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

pub fn try_answer_from_json_qa(state: &AppState, query: &str) -> Option<String> {
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

pub async fn weak_hints_from_codeclass_and_json(state: &AppState, query: &str) -> Option<String> {
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
