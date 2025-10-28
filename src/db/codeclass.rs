use anyhow::Result;
use sqlx::{MySql, Pool, Row};

/* ===================== Конкретные SELECT-запросы (RO) ===================== */

pub async fn query_user(pool: &Pool<MySql>, q: &str) -> Result<String> {
    // users: name, last_name, username, email
    let rows = sqlx::query(
        r#"
        SELECT CAST(id AS SIGNED) AS id,
               CONCAT_WS(' ', name, last_name) AS full_name,
               username, email
        FROM users
        WHERE email LIKE ? OR username LIKE ?
           OR name LIKE ? OR last_name LIKE ?
        ORDER BY id DESC
        LIMIT 10
        "#,
    )
    .bind(format!("%{}%", q))
    .bind(format!("%{}%", q))
    .bind(format!("%{}%", q))
    .bind(format!("%{}%", q))
    .fetch_all(pool)
    .await?;

    let mut out = String::new();
    for r in rows {
        let id: i64 = r.try_get("id")?;
        let name: String = r.try_get("full_name").unwrap_or_default();
        let username: String = r.try_get("username").unwrap_or_default();
        let email: String = r.try_get("email").unwrap_or_default();
        out.push_str(&format!("ID:{id} • {name} • @{username} • {email}\n"));
    }
    Ok(out)
}

pub async fn query_admin(pool: &Pool<MySql>, q: &str) -> Result<String> {
    let rows = sqlx::query(
        r#"
        SELECT CAST(id AS SIGNED) AS id, name, username, email
        FROM admins
        WHERE email LIKE ? OR name LIKE ? OR username LIKE ?
        ORDER BY id DESC
        LIMIT 10
        "#,
    )
    .bind(format!("%{}%", q))
    .bind(format!("%{}%", q))
    .bind(format!("%{}%", q))
    .fetch_all(pool)
    .await?;

    let mut out = String::new();
    for r in rows {
        let id: i64 = r.try_get("id")?;
        let name: String = r.try_get("name").unwrap_or_default();
        let username: String = r.try_get("username").unwrap_or_default();
        let email: String = r.try_get("email").unwrap_or_default();
        out.push_str(&format!("ID:{id} • {name} • @{username} • {email}\n"));
    }
    Ok(out)
}

pub async fn query_courses(pool: &Pool<MySql>, q: Option<&str>) -> Result<String> {
    let rows = if let Some(k) = q {
        sqlx::query(
            r#"
            SELECT CAST(id AS SIGNED) AS id, title
            FROM courses
            WHERE title LIKE ?
            ORDER BY id DESC
            LIMIT 10
            "#,
        )
        .bind(format!("%{}%", k))
        .fetch_all(pool)
        .await?
    } else {
        sqlx::query(r#"SELECT CAST(id AS SIGNED) AS id, title FROM courses ORDER BY id DESC LIMIT 10"#)
            .fetch_all(pool)
            .await?
    };
    let mut out = String::new();
    for r in rows {
        let id: i64 = r.try_get("id")?;
        let title: String = r.try_get("title")?;
        out.push_str(&format!("ID:{id} • {title}\n"));
    }
    Ok(out)
}

pub async fn query_pricing(pool: &Pool<MySql>, course_like: Option<&str>) -> Result<String> {
    // pricing: title, count_lesson, price + join courses.title
    let rows = if let Some(k) = course_like {
        sqlx::query(
            r#"
            SELECT c.title AS course,
                   p.title AS plan_title,
                   p.count_lesson,
                   p.price
            FROM pricing p
            LEFT JOIN courses c ON c.id = p.course_id
            WHERE c.title LIKE ?
            ORDER BY p.course_id DESC, p.price ASC
            LIMIT 10
            "#,
        )
        .bind(format!("%{}%", k))
        .fetch_all(pool)
        .await?
    } else {
        sqlx::query(
            r#"
            SELECT c.title AS course,
                   p.title AS plan_title,
                   p.count_lesson,
                   p.price
            FROM pricing p
            LEFT JOIN courses c ON c.id = p.course_id
            ORDER BY p.course_id DESC, p.price ASC
            LIMIT 10
            "#,
        )
        .fetch_all(pool)
        .await?
    };

    let mut out = String::new();
    for r in rows {
        let course: String = r.try_get("course").unwrap_or_default();
        let plan: String = r.try_get("plan_title").unwrap_or_default();
        let count_lesson: i64 = r.try_get("count_lesson").unwrap_or(0);
        let price: i64 = r.try_get("price").unwrap_or(0);
        out.push_str(&format!(
            "{course}: {plan} — {count_lesson} занятий • {price} ₽\n"
        ));
    }
    Ok(out)
}

pub async fn query_schedule(
    pool: &Pool<MySql>,
    course_like: Option<&str>,
    date: Option<&str>,
) -> Result<String> {
    // schedules → groups → courses, дата: COALESCE(start_at, date_start)
    let mut q = String::from(
        r#"
        SELECT CAST(s.id AS SIGNED) AS id,
               c.title AS course,
               DATE_FORMAT(COALESCE(s.start_at, s.date_start), '%Y-%m-%d %H:%i') AS dt,
               g.title AS group_name
        FROM schedules s
        LEFT JOIN groups g ON g.id = s.group_id
        LEFT JOIN courses c ON c.id = g.course_id
        WHERE 1=1
        "#,
    );
    let mut binds: Vec<String> = vec![];
    if let Some(k) = course_like {
        q.push_str(" AND (c.title LIKE ?)");
        binds.push(format!("%{}%", k));
    }
    if let Some(d) = date {
        q.push_str(" AND DATE(COALESCE(s.start_at, s.date_start)) = ?");
        binds.push(d.to_string());
    }
    q.push_str(" ORDER BY COALESCE(s.start_at, s.date_start) ASC LIMIT 10");

    let mut query = sqlx::query(&q);
    for b in binds {
        query = query.bind(b);
    }
    let rows = query.fetch_all(pool).await?;

    let mut out = String::new();
    for r in rows {
        let id: i64 = r.try_get("id")?;
        let course: String = r.try_get("course").unwrap_or_default();
        let dt: String = r.try_get("dt").unwrap_or_default();
        let group: String = r.try_get("group_name").unwrap_or_default();
        out.push_str(&format!("#{id} • {dt} • {course} • {group}\n"));
    }
    Ok(out)
}

pub async fn query_lessons(
    pool: &Pool<MySql>,
    course_like: Option<&str>,
    date: Option<&str>,
) -> Result<String> {
    // lessons → course_lesson → courses
    let mut q = String::from(
        r#"
        SELECT CAST(l.id AS SIGNED) AS id,
               l.title AS lesson,
               c.title AS course,
               DATE_FORMAT(l.created_at, '%Y-%m-%d %H:%i') AS dt
        FROM lessons l
        LEFT JOIN course_lesson cl ON cl.lesson_id = l.id
        LEFT JOIN courses c ON c.id = cl.course_id
        WHERE 1=1
        "#,
    );
    let mut binds: Vec<String> = vec![];
    if let Some(k) = course_like {
        q.push_str(" AND (c.title LIKE ?)");
        binds.push(format!("%{}%", k));
    }
    if let Some(d) = date {
        // если в lessons есть starts_at — подменить на него здесь
        q.push_str(" AND DATE(l.created_at) = ?");
        binds.push(d.to_string());
    }
    q.push_str(" ORDER BY l.created_at ASC LIMIT 10");

    let mut query = sqlx::query(&q);
    for b in binds {
        query = query.bind(b);
    }
    let rows = query.fetch_all(pool).await?;

    let mut out = String::new();
    for r in rows {
        let id: i64 = r.try_get("id")?;
        let course: String = r.try_get("course").unwrap_or_default();
        let lesson: String = r.try_get("lesson").unwrap_or_default();
        let dt: String = r.try_get("dt").unwrap_or_default();
        out.push_str(&format!("#{id} • {dt} • {course} — {lesson}\n"));
    }
    Ok(out)
}

pub async fn find_user_id(pool: &Pool<MySql>, q: &str) -> Result<Option<i64>> {
    let row = sqlx::query(
        r#"
        SELECT CAST(id AS SIGNED) AS id
        FROM users
        WHERE email LIKE ? OR phone LIKE ? OR username LIKE ?
           OR name LIKE ? OR last_name LIKE ?
        ORDER BY id DESC LIMIT 1
        "#,
    )
    .bind(format!("%{}%", q))
    .bind(format!("%{}%", q))
    .bind(format!("%{}%", q))
    .bind(format!("%{}%", q))
    .bind(format!("%{}%", q))
    .fetch_optional(pool)
    .await?;

    Ok(row.and_then(|r| r.try_get::<i64, _>("id").ok()))
}

pub async fn query_enrollments(pool: &Pool<MySql>, user_q: &str) -> Result<String> {
    if let Some(uid) = find_user_id(pool, user_q).await? {
        let rows = sqlx::query(
            r#"
            SELECT CAST(e.id AS SIGNED) AS id,
                   c.title AS course,
                   e.status
            FROM enrollments e
            LEFT JOIN courses c ON c.id = e.course_id
            WHERE e.user_id = ?
            ORDER BY e.id DESC
            LIMIT 10
            "#,
        )
        .bind(uid)
        .fetch_all(pool)
        .await?;

        let mut out = format!("Записи для user_id={uid}:\n");
        for r in rows {
            let id: i64 = r.try_get("id")?;
            let course: String = r.try_get("course").unwrap_or_default();
            let status: String = r.try_get("status").unwrap_or_default();
            out.push_str(&format!("#{id} • {course} • {status}\n"));
        }
        return Ok(out);
    }
    Ok("Пользователь не найден.".into())
}

pub async fn query_orders(pool: &Pool<MySql>, user_q: &str) -> Result<String> {
    if let Some(uid) = find_user_id(pool, user_q).await? {
        let rows = sqlx::query(
            r#"
            SELECT CAST(o.id AS SIGNED) AS id,
                   o.total_amount,
                   o.status,
                   DATE_FORMAT(o.created_at, '%Y-%m-%d') AS dt
            FROM `order` o
            WHERE EXISTS (
                SELECT 1 FROM invoices i
                WHERE i.user_id = ? AND i.id = o.invoice_id
            )
            ORDER BY o.id DESC
            LIMIT 10
            "#,
        )
        .bind(uid)
        .fetch_all(pool)
        .await?;

        let mut out = format!("Заказы для user_id={uid} (через invoices):\n");
        for r in rows {
            let id: i64 = r.try_get("id")?;
            let total: f64 = r.try_get("total_amount").unwrap_or(0.0);
            let status: i64 = r.try_get("status").unwrap_or(0);
            let dt: String = r.try_get("dt").unwrap_or_default();
            out.push_str(&format!("#{id} • {dt} • {total} • status={status}\n"));
        }
        return Ok(out);
    }
    Ok("Пользователь не найден.".into())
}

pub async fn query_invoices(pool: &Pool<MySql>, user_q: &str) -> Result<String> {
    if let Some(uid) = find_user_id(pool, user_q).await? {
        let rows = sqlx::query(
            r#"
            SELECT CAST(id AS SIGNED) AS id,
                   pay_amount,
                   status,
                   DATE_FORMAT(created_at, '%Y-%m-%d') AS dt
            FROM invoices
            WHERE user_id = ?
            ORDER BY id DESC
            LIMIT 10
            "#,
        )
        .bind(uid)
        .fetch_all(pool)
        .await?;

        let mut out = format!("Счета для user_id={uid}:\n");
        for r in rows {
            let id: i64 = r.try_get("id")?;
            let amount: f64 = r.try_get("pay_amount").unwrap_or(0.0);
            let status: i64 = r.try_get("status").unwrap_or(0);
            let dt: String = r.try_get("dt").unwrap_or_default();
            out.push_str(&format!("#{id} • {dt} • {amount} • status:{status}\n"));
        }
        return Ok(out);
    }
    Ok("Пользователь не найден.".into())
}

pub async fn query_partner_payments(pool: &Pool<MySql>, _user_q: &str) -> Result<String> {
    // В таблице payments_partners нет user_id — отдаем последние записи
    let rows = sqlx::query(
        r#"
        SELECT CAST(id AS SIGNED) AS id, name, total_payable,
               DATE_FORMAT(period_from, '%Y-%m-%d') AS dfrom,
               DATE_FORMAT(period_to, '%Y-%m-%d')   AS dto
        FROM payments_partners
        ORDER BY period_to DESC
        LIMIT 10
        "#,
    )
    .fetch_all(pool)
    .await?;

    let mut out = String::from("Последние партнёрские выплаты:\n");
    for r in rows {
        let id: i64 = r.try_get("id")?;
        let name: String = r.try_get("name").unwrap_or_default();
        let total: f64 = r.try_get("total_payable").unwrap_or(0.0);
        let dfrom: String = r.try_get("dfrom").unwrap_or_default();
        let dto: String = r.try_get("dto").unwrap_or_default();
        out.push_str(&format!("#{id} • {dfrom}..{dto} • {name} • {total}\n"));
    }
    Ok(out)
}

pub async fn query_loan_apps(pool: &Pool<MySql>, user_q: &str) -> Result<String> {
    let q = user_q.trim();
    if q.is_empty() {
        return Ok("Укажи запрос: имя/телефон/email/номер заказа или id.".into());
    }

    // Если это чисто число — считаем, что хотят точный поиск по id.
    let numeric_id = q.parse::<i64>().ok();

    let rows = if let Some(id) = numeric_id {
        sqlx::query(
            r#"
            SELECT
              CAST(id AS SIGNED)            AS id,
              CAST(franchise_id AS SIGNED)  AS franchise_id,
              CAST(school_id AS SIGNED)     AS school_id,
              first_name, last_name, middle_name,
              client_phone, client_email,
              order_id, tinkoff_order_id, link,
              is_course, is_test,
              status, amount,
              DATE_FORMAT(created_at, '%Y-%m-%d %H:%i') AS dt,
              DATE_FORMAT(confirm_date, '%Y-%m-%d %H:%i') AS cdt
            FROM loan_application
            WHERE id = ?
            ORDER BY created_at DESC
            LIMIT 20
            "#,
        )
        .bind(id)
        .fetch_all(pool)
        .await?
    } else {
        // Текстовый поиск по основным полям
        sqlx::query(
            r#"
            SELECT
              CAST(id AS SIGNED)            AS id,
              CAST(franchise_id AS SIGNED)  AS franchise_id,
              CAST(school_id AS SIGNED)     AS school_id,
              first_name, last_name, middle_name,
              client_phone, client_email,
              order_id, tinkoff_order_id, link,
              is_course, is_test,
              status, amount,
              DATE_FORMAT(created_at, '%Y-%m-%d %H:%i') AS dt,
              DATE_FORMAT(confirm_date, '%Y-%m-%d %H:%i') AS cdt
            FROM loan_application
            WHERE
                first_name       LIKE CONCAT('%', ?, '%')
             OR last_name        LIKE CONCAT('%', ?, '%')
             OR middle_name      LIKE CONCAT('%', ?, '%')
             OR client_phone     LIKE CONCAT('%', ?, '%')
             OR client_email     LIKE CONCAT('%', ?, '%')
             OR order_id         LIKE CONCAT('%', ?, '%')
             OR tinkoff_order_id LIKE CONCAT('%', ?, '%')
            ORDER BY created_at DESC
            LIMIT 20
            "#,
        )
        .bind(q) // first_name
        .bind(q) // last_name
        .bind(q) // middle_name
        .bind(q) // client_phone
        .bind(q) // client_email
        .bind(q) // order_id
        .bind(q) // tinkoff_order_id
        .fetch_all(pool)
        .await?
    };

    if rows.is_empty() {
        return Ok("Ничего не нашёл по запросу.".into());
    }

    let mut out = String::new();
    for r in rows {
        let id: i64 = r.try_get("id")?;
        let fid: Option<i64> = r.try_get("franchise_id").ok();
        let sid: Option<i64> = r.try_get("school_id").ok();

        let first_name: String = r.try_get("first_name").unwrap_or_default();
        let last_name: String = r.try_get("last_name").unwrap_or_default();
        let middle_name: String = r.try_get("middle_name").unwrap_or_default();

        let phone: String = r.try_get("client_phone").unwrap_or_default();
        let email: String = r.try_get("client_email").unwrap_or_default();

        let order_id: String = r.try_get("order_id").unwrap_or_default();
        let tinkoff_order_id: String = r.try_get("tinkoff_order_id").unwrap_or_default();
        let link: String = r.try_get("link").unwrap_or_default();

        let is_course: i64 = r.try_get("is_course").unwrap_or(0);
        let is_test: i64 = r.try_get("is_test").unwrap_or(0);

        let status: i64 = r.try_get("status").unwrap_or(0);
        let amount: f64 = r.try_get("amount").unwrap_or(0.0);

        let dt: String = r.try_get("dt").unwrap_or_default();
        let cdt: String = r.try_get("cdt").unwrap_or_default();

        let fio = [
            last_name.as_str(),
            first_name.as_str(),
            middle_name.as_str(),
        ]
        .iter()
        .filter(|s| !s.is_empty())
        .cloned()
        .collect::<Vec<_>>()
        .join(" ");

        out.push_str(&format!(
            "#{id} • {dt} • {fio}\n\
             └ phone:{phone} • email:{email}\n\
             └ amount:{amount} • status:{status} • course:{is_course} • test:{is_test}\n\
             └ order:{order_id} • tinkoff:{tinkoff_order_id}\n"
        ));
        if let Some(fid) = fid {
            out.push_str(&format!("└ franchise:{fid}"));
            if let Some(sid) = sid {
                out.push_str(&format!(" • school:{sid}"));
            }
            out.push('\n');
        }
        if !link.is_empty() {
            out.push_str(&format!("└ link:{link}\n"));
        }
        if !cdt.is_empty() {
            out.push_str(&format!("└ confirmed:{cdt}\n"));
        }
        out.push('\n');
    }

    Ok(out)
}

pub async fn query_lesson_feedback(
    pool: &Pool<MySql>,
    user_q: Option<&str>,
    lesson_id: Option<i64>,
) -> Result<String> {
    if let Some(id) = lesson_id {
        let rows = sqlx::query(
            r#"
            SELECT CAST(id AS SIGNED) AS id,
                   CAST(lesson_id AS SIGNED) AS lesson_id,
                   CAST(user_id  AS SIGNED)  AS user_id,
                   rating,
                   LEFT(comment, 140) AS c,
                   DATE_FORMAT(created_at, '%Y-%m-%d') AS dt
            FROM lesson_feedback
            WHERE lesson_id = ?
            ORDER BY created_at DESC
            LIMIT 10
            "#,
        )
        .bind(id)
        .fetch_all(pool)
        .await?;

        let mut out = format!("Фидбек по уроку #{id}:\n");
        for r in rows {
            let fid: i64 = r.try_get("id")?;
            let uid: i64 = r.try_get("user_id")?;
            let rating: i64 = r.try_get("rating").unwrap_or(0);
            let c: String = r.try_get("c").unwrap_or_default();
            let dt: String = r.try_get("dt").unwrap_or_default();
            out.push_str(&format!("#{fid} • {dt} • user:{uid} • {rating}/5 • {c}\n"));
        }
        return Ok(out);
    } else if let Some(uq) = user_q {
        if let Some(uid) = find_user_id(pool, uq).await? {
            let rows = sqlx::query(
                r#"
                SELECT CAST(id AS SIGNED) AS id,
                       CAST(lesson_id AS SIGNED) AS lesson_id,
                       rating,
                       LEFT(comment, 140) AS c,
                       DATE_FORMAT(created_at, '%Y-%m-%d') AS dt
                FROM lesson_feedback
                WHERE user_id = ?
                ORDER BY created_at DESC
                LIMIT 10
                "#,
            )
            .bind(uid)
            .fetch_all(pool)
            .await?;

            let mut out = format!("Фидбек пользователя user_id={uid}:\n");
            for r in rows {
                let fid: i64 = r.try_get("id")?;
                let lid: i64 = r.try_get("lesson_id")?;
                let rating: i64 = r.try_get("rating").unwrap_or(0);
                let c: String = r.try_get("c").unwrap_or_default();
                let dt: String = r.try_get("dt").unwrap_or_default();
                out.push_str(&format!(
                    "#{fid} • {dt} • lesson:{lid} • {rating}/5 • {c}\n"
                ));
            }
            return Ok(out);
        }
        return Ok("Пользователь не найден.".into());
    }
    Ok("Укажи /feedback user <запрос> или /feedback lesson <id>.".into())
}
