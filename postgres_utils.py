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
