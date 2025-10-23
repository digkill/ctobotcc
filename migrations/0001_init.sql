CREATE TABLE IF NOT EXISTS chat_history (
                                            id BIGINT UNSIGNED PRIMARY KEY AUTO_INCREMENT,
                                            chat_id BIGINT NOT NULL, -- ключ истории (чаты: chat_id; личка: -user_id)
                                            role ENUM('system','user','assistant') NOT NULL,
    content TEXT NOT NULL,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE IF NOT EXISTS facts (
                                     id BIGINT UNSIGNED PRIMARY KEY AUTO_INCREMENT,
                                     chat_id BIGINT NULL, -- NULL = глобальные факты
                                     tag VARCHAR(64) NULL,
    content TEXT NOT NULL,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE IF NOT EXISTS qa_pairs (
                                        id BIGINT UNSIGNED PRIMARY KEY AUTO_INCREMENT,
                                        question TEXT NOT NULL,
                                        answer   TEXT NOT NULL,
                                        tags     VARCHAR(255) NULL,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
