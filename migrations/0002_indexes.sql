CREATE INDEX idx_chat_history_chat_created ON chat_history (chat_id, created_at);
CREATE INDEX idx_facts_chat_tag            ON facts        (chat_id, tag);
