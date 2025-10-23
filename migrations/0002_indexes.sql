ALTER TABLE chat_history ADD INDEX idx_chat_history_chat_created (chat_id, created_at);
ALTER TABLE facts        ADD INDEX idx_facts_chat_tag              (chat_id, tag);
-- FULLTEXT индексы (если включён innodb_fulltext)
-- CREATE FULLTEXT INDEX ftx_qa_question ON qa_pairs (question);
