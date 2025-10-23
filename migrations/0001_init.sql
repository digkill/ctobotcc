CREATE TABLE IF NOT EXISTS chat_history (
                                            id BIGINT UNSIGNED PRIMARY KEY AUTO_INCREMENT,
                                            chat_id BIGINT NOT NULL,
                                            role ENUM('system','user','assistant') NOT NULL,
    content TEXT NOT NULL,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE IF NOT EXISTS facts (
                                     id BIGINT UNSIGNED PRIMARY KEY AUTO_INCREMENT,
                                     chat_id BIGINT NULL, -- NULL = глобальный факт для всех чатов
                                     tag VARCHAR(64) NULL,
    content TEXT NOT NULL,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
