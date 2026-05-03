from contextlib import contextmanager
from psycopg import connect as pgsql_connect, OperationalError as PgsqlOperationalError


_DEFAULT_PGSQL_SCHEMA_NAME = "trash_monitor"


@contextmanager
def connect_pgsql(
    state_dsn: str,
    schema_name: str = _DEFAULT_PGSQL_SCHEMA_NAME,
    timeout_seconds: int = 10,
):
    try:
        with pgsql_connect(
            state_dsn,
            options=f"-c search_path={schema_name}",
            connect_timeout=timeout_seconds,
        ) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    f"SET LOCAL statement_timeout = '{timeout_seconds}s'"
                )
                cur.execute(
                    f"SET LOCAL lock_timeout = '{timeout_seconds}s'"
                )
            yield conn
    except PgsqlOperationalError as exc:
        raise RuntimeError(
            f"Не удалось подключиться к БД в течение {timeout_seconds} с."
        ) from exc
    except Exception as exc:
        sqlstate = getattr(exc, "sqlstate", None)
        if sqlstate in {"57014", "55P03"}:
            raise TimeoutError(
                f"Запрос к БД выполнялся или ожидал блокировки дольше {timeout_seconds} с."
            ) from exc
        raise


def insert_relevant_messages(
    state_dsn: str,
    messages: list[dict[str, object]],
    source: str,
    schema_name: str = _DEFAULT_PGSQL_SCHEMA_NAME,
    timeout_seconds: int = 10,
) -> int:
    rows = [
        (
            msg["group_id"],
            msg["post_id"],
            msg["text"],
            msg.get("latitude"),
            msg.get("longitude"),
            source,
        )
        for msg in messages
    ]

    insert_sql = (
        "INSERT INTO relevant_messages "
        "(group_id, post_id, text, latitude, longitude, source) "
        "VALUES (%s, %s, %s, %s, %s, %s) "
        "ON CONFLICT DO NOTHING"
    )

    with connect_pgsql(
        state_dsn, schema_name=schema_name, timeout_seconds=timeout_seconds
    ) as conn:
        with conn.cursor() as cur:
            cur.execute(
                "CREATE TABLE IF NOT EXISTS relevant_messages ("
                "group_id BIGINT, "
                "post_id BIGINT,"
                "text TEXT, "
                "latitude FLOAT, "
                "longitude FLOAT, "
                "source TEXT, "
                "PRIMARY KEY (group_id, post_id)"
                ")"
            )
            cur.executemany(insert_sql, rows)
            return cur.rowcount
    return 0
