from psycopg import OperationalError as PgsqlOperationalError, connect as pgsql_connect
from contextlib import contextmanager
from vk_api import VkApi
from pymorphy3 import MorphAnalyzer
from re import compile as re_compile
from sentence_transformers import SentenceTransformer
from numpy import array as np_array, float32 as np_float32
from faiss import IndexFlatIP as FaissIndexFlatIP
from openai import OpenAI

_DEFAULT_COMMUNITY_STOP_WORDS = ["бизнес", "видеоигра", "животное", "знакомство", "игра", "история", "кафе", "музыка", "передача",
                                 "путешествие", "реклама", "ресторан", "рыбалка", "техника", "технология", "товар", "туризм", "философия", "шоу", "экономика", "юмор"]
_DEFAULT_COMMUNITY_POSITIVE_KEYWORDS = ["адрес", "город", "дом", "жизнь", "квартира", "лифт", "магазин", "метро", "микрорайон",
                                        "москва", "набережная", "подъезд", "помощь", "работа", "район", "ребёнок", "ремонт", "улица", "участок", "школа"]
_DEFAULT_COMMUNITY_NEGATIVE_KEYWORDS = ["ai", "военкомат", "интеллект", "искусственный",
                                        "контракт", "модель", "нейросеть", "подписать", "ранение", "робот", "сво"]


@contextmanager
def _connect_pgsql(state_dsn: str, connect_timeout_seconds: int):
    try:
        with pgsql_connect(state_dsn, connect_timeout=connect_timeout_seconds) as conn:
            yield conn
    except PgsqlOperationalError as exc:
        raise RuntimeError(
            "Не удалось подключиться к PostgreSQL в течение настроенного времени ожидания. "
            "Проверьте POSTGRESQL_DSN и доступность базы данных."
        ) from exc
    except Exception as exc:
        sqlstate = getattr(exc, "sqlstate", None)
        if sqlstate in {"57014", "55P03"}:
            raise TimeoutError(
                "Запрос к базе данных превысил время ожидания или долго ожидал блокировку. "
                "Другая сессия может удерживать блокировку на таблицах vk_monitor_*."
            ) from exc
        raise


def get_new_communities(
    api_key: str,
    queries: list[str],
    state_dsn: str,
    count_per_query: int = 100,
    db_connect_timeout_seconds: int = 10,
    db_statement_timeout_seconds: int = 10,
    db_lock_timeout_seconds: int = 10,
) -> list[dict[str, object]]:
    new_communities = []
    api = VkApi(token=api_key, api_version="5.199").get_api()

    with _connect_pgsql(state_dsn, db_connect_timeout_seconds) as conn:
        with conn.cursor() as cur:
            cur.execute(
                f"SET LOCAL statement_timeout = '{db_statement_timeout_seconds}s'")
            cur.execute(
                f"SET LOCAL lock_timeout = '{db_lock_timeout_seconds}s'")

            cur.execute(
                "CREATE TABLE IF NOT EXISTS vk_monitor_offsets (query TEXT PRIMARY KEY, query_offset SMALLINT NOT NULL DEFAULT 0)"
            )
            cur.execute(
                "CREATE TABLE IF NOT EXISTS vk_monitor_seen (group_id BIGINT PRIMARY KEY, reason_discarded SMALLINT DEFAULT 0)"
            )
            cur.execute(
                "CREATE TABLE IF NOT EXISTS vk_monitor_seen_queries (group_id BIGINT NOT NULL REFERENCES vk_monitor_seen(group_id) ON DELETE CASCADE, query TEXT NOT NULL, PRIMARY KEY (group_id, query))"
            )

            cur.execute(
                "SELECT query, query_offset FROM vk_monitor_offsets")
            offsets = dict(tuple([row[0], row[1]])
                           for row in cur.fetchall())
            cur.execute("SELECT group_id FROM vk_monitor_seen")
            seen = set(int(row[0]) for row in cur.fetchall())

            for query in queries:
                offset = offsets.get(query, 0)
                api_response = api.groups.search(
                    q=query,
                    type="group",
                    count=count_per_query,
                    offset=offset,
                    sort=0,
                    fields="activity",
                )
                items = api_response.get("items", [])
                offsets[query] = offset + len(items)

                for group in items:
                    group_id = group.get("id")
                    if group_id is None:
                        continue

                    if group_id not in seen:
                        seen.add(group_id)
                        cur.execute(
                            "INSERT INTO vk_monitor_seen (group_id) VALUES (%s) ON CONFLICT (group_id) DO NOTHING",
                            tuple([group_id]),
                        )
                        if group["is_closed"] == 0:
                            new_communities.append(
                                dict(
                                    id=group_id,
                                    name=group.get("name"),
                                    category=group.get("activity"),
                                )
                            )

                    cur.execute(
                        "INSERT INTO vk_monitor_seen_queries (group_id, query) VALUES (%s, %s) ON CONFLICT (group_id, query) DO NOTHING",
                        tuple([group_id, query]),
                    )

            for query, offset in offsets.items():
                cur.execute(
                    "INSERT INTO vk_monitor_offsets (query, query_offset) VALUES (%s, %s) ON CONFLICT (query) DO UPDATE SET query_offset = EXCLUDED.query_offset",
                    tuple([query, offset]),
                )

    return new_communities


def filter_communities_by_stop_words(
    communities: list[dict[str, object]],
    state_dsn: str,
    stop_words: list[str] = _DEFAULT_COMMUNITY_STOP_WORDS,
    db_connect_timeout_seconds: int = 10,
    db_statement_timeout_seconds: int = 10,
    db_lock_timeout_seconds: int = 10,
) -> list[dict[str, object]]:
    with _connect_pgsql(state_dsn, db_connect_timeout_seconds) as conn:
        with conn.cursor() as cur:
            cur.execute(
                f"SET LOCAL statement_timeout = '{db_statement_timeout_seconds}s'")
            cur.execute(
                f"SET LOCAL lock_timeout = '{db_lock_timeout_seconds}s'")

            word_re = re_compile(r"[^\W\d_]+")
            morph = MorphAnalyzer()

            normalized_stop_words = {morph.parse(
                word)[0].normal_form for word in stop_words}

            filtered_communities = []
            for community in communities:
                category = community.get("category")
                if not isinstance(category, str):
                    filtered_communities.append(community)
                    continue

                normalized_category_words = {
                    morph.parse(word)[0].normal_form
                    for word in word_re.findall(category.lower())
                }
                if normalized_category_words.intersection(normalized_stop_words):
                    group_id = community.get("id")
                    cur.execute(
                        "INSERT INTO vk_monitor_seen (group_id, reason_discarded) VALUES (%s, 1) "
                        "ON CONFLICT (group_id) DO UPDATE SET reason_discarded = 1",
                        (group_id,),
                    )
                    continue

                filtered_communities.append(community)

    return filtered_communities


def filter_communities_semantically(
    api_key: str,
    communities: list[dict[str, object]],
    state_dsn: str,
    positive_keywords: list[str] = _DEFAULT_COMMUNITY_POSITIVE_KEYWORDS,
    negative_keywords: list[str] = _DEFAULT_COMMUNITY_NEGATIVE_KEYWORDS,
    db_connect_timeout_seconds: int = 10,
    db_statement_timeout_seconds: int = 10,
    db_lock_timeout_seconds: int = 10
) -> list[dict[str, object]]:
    with _connect_pgsql(state_dsn, db_connect_timeout_seconds) as conn:
        with conn.cursor() as cur:
            cur.execute(
                f"SET LOCAL statement_timeout = '{db_statement_timeout_seconds}s'")
            cur.execute(
                f"SET LOCAL lock_timeout = '{db_lock_timeout_seconds}s'")
            api = VkApi(token=api_key, api_version="5.199").get_api()
            communities_bundles: dict[str, dict[str, str | list[str]]] = {}
            for community in communities:
                group_data = api.groups.getById(group_id=community.get(
                    "id"), fields="description")["groups"][0]
                communities_bundles[community.get("id")] = dict(
                    name=group_data["name"], description=group_data.get("description"))

                owner_id = -int(community.get("id"))
                group_posts = api.wall.get(
                    owner_id=owner_id, count=100, extended=1)["items"]
                communities_bundles[community.get("id")]["pin_post"] = group_posts[0].get(
                    "text") if group_posts[0].get("is_pinned") == 1 else ""

                communities_bundles[community.get("id")]["reg_posts"] = []
                for post_id in range(group_posts[0].get("is_pinned") == 1, len(group_posts)):
                    communities_bundles[community.get("id")]["reg_posts"].append(
                        group_posts[post_id].get("text"))

            scores = {}
            for community_id, community_bundle in communities_bundles.items():
                corpus = " ".join(
                    [community_bundle.get("name", ""),
                     community_bundle.get("description", ""),
                     community_bundle.get("pin_post", ""),
                     *community_bundle.get("reg_posts", [])]
                ).lower()
                scores[community_id] = (
                    sum(1 for term in positive_keywords if term in corpus)
                    - sum(1 for term in negative_keywords if term in corpus)
                )

            filtered_communities = []
            for community in communities:
                if scores.get(community.get("id"), 0) >= 5:
                    filtered_communities.append(community)
                else:
                    group_id = community.get("id")
                    with _connect_pgsql(state_dsn, db_connect_timeout_seconds) as conn:
                        with conn.cursor() as cur:
                            cur.execute(
                                "INSERT INTO vk_monitor_seen (group_id, reason_discarded) VALUES (%s, 2) "
                                "ON CONFLICT (group_id) DO UPDATE SET reason_discarded = 2",
                                (group_id,),
                            )
    return filtered_communities


def filter_posts_with_faiss(
    posts: list[str],
    target_phrase: str = "мусор около железной дороги",
    threshold: float = 0.3
) -> list[str]:
    embedder = SentenceTransformer(
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )
    post_vecs = embedder.encode(posts, normalize_embeddings=True)
    query_vec = embedder.encode([target_phrase], normalize_embeddings=True)

    post_embeds = np_array(post_vecs, dtype=np_float32)
    query_embed = np_array(query_vec, dtype=np_float32)

    index = FaissIndexFlatIP(post_embeds.shape[1])
    index.add(post_embeds)
    similarities, indices = index.search(query_embed, len(posts))

    scores = [0.0] * len(posts)
    for rank in range(len(posts)):
        scores[int(indices[0, rank])] = float(similarities[0, rank])

    filtered_posts = []
    for post, similarity_score in zip(posts, scores):
        if similarity_score >= threshold:
            filtered_posts.append(post)
    return filtered_posts


def filter_posts_with_llm(
    posts: list[str],
    api: str,
    api_key: str,
    model: str
) -> list[str]:
    def _normalize_answer(raw_text: str) -> str:
        text = (raw_text).strip().lower()
        for char in ".,!?;:\n\r\t\"'()[]{}":
            text = text.replace(char, " ")

        tokens = [token for token in text.split(" ")]
        for token in tokens:
            if token == "да":
                return "да"
            if token == "нет":
                return "нет"

        return ""

    client = OpenAI(base_url=api, api_key=api_key)
    filtered_posts = []
    for post in posts:
        text = post[:1500]
        resp = client.chat.completions.create(
            model=model,
            temperature=0,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Определи, описан ли в тексте конкретный случай, где мусор лежит рядом с железнодорожной инфраструктурой (пути, станция, переезд). Если в тексте говорится, что мусор уже убран, отвечай 'нет'. "
                        "Отвечай только одним словом: 'да' или 'нет'."
                    ),
                },
                {"role": "user", "content": text},
            ],
        )
        resp = _normalize_answer(
            (resp.choices[0].message.content or "").strip())

        if resp == "да":
            filtered_posts.append(post)
    return filtered_posts
