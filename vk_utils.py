from vk_api import VkApi
from postgres_utils import connect_pgsql
from re import compile as re_compile
from pymorphy3 import MorphAnalyzer
from openai import OpenAI


_DEFAULT_COMMUNITY_CATEGORY_STOP_WORDS = ["бизнес", "видеоигра", "животное", "знакомство", "игра", "история", "кафе", "музыка", "передача",
                                          "путешествие", "реклама", "ресторан", "рыбалка", "техника", "технология", "товар", "туризм", "философия", "шоу", "экономика", "юмор"]
_DEFAULT_COMMUNITY_POSITIVE_WORDS = ["адрес", "город", "дом", "жизнь", "квартира", "лифт", "магазин", "метро", "микрорайон",
                                     "москва", "набережная", "подъезд", "помощь", "работа", "район", "ребёнок", "ремонт", "улица", "участок", "школа"]
_DEFAULT_COMMUNITY_NEGATIVE_WORDS = ["ai", "военкомат", "интеллект", "искусственный",
                                     "контракт", "модель", "нейросеть", "подписать", "ранение", "робот", "сво"]


def get_new_vk_communities(
    api_key: str,
    queries: list[str],
    state_dsn: str,
    count_per_query: int = 100,
    db_timeout_seconds: int = 10,
) -> list[dict[str, object]]:
    new_communities = []
    api = VkApi(token=api_key, api_version="5.199").get_api()

    with connect_pgsql(state_dsn, timeout_seconds=db_timeout_seconds) as conn:
        with conn.cursor() as cur:
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
            offsets = dict(cur.fetchall())
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
                            (group_id,),
                        )
                        if group["is_closed"] == 0:
                            new_communities.append(
                                {
                                    "id": group_id,
                                    "name": group.get("name"),
                                    "category": group.get("activity"),
                                }
                            )

                    cur.execute(
                        "INSERT INTO vk_monitor_seen_queries (group_id, query) VALUES (%s, %s) ON CONFLICT (group_id, query) DO NOTHING",
                        (group_id, query),
                    )

            for query, offset in offsets.items():
                cur.execute(
                    "INSERT INTO vk_monitor_offsets (query, query_offset) VALUES (%s, %s) ON CONFLICT (query) DO UPDATE SET query_offset = EXCLUDED.query_offset",
                    (query, offset),
                )

    return new_communities


def filter_vk_communities_by_category_stop_words(
    communities: list[dict[str, object]],
    state_dsn: str,
    stop_words: list[str] = _DEFAULT_COMMUNITY_CATEGORY_STOP_WORDS,
    db_timeout_seconds: int = 10,
) -> list[dict[str, object]]:
    with connect_pgsql(state_dsn, timeout_seconds=db_timeout_seconds) as conn:
        with conn.cursor() as cur:
            word_re = re_compile(r"[^\W\d_]+")
            morph = MorphAnalyzer()
            normalized_stop_words = set(morph.parse(
                word)[0].normal_form for word in stop_words)

            filtered_communities = []
            for community in communities:
                category = community.get("category")
                if not isinstance(category, str):
                    filtered_communities.append(community)
                    continue

                normalized_category_words = set(
                    morph.parse(word)[0].normal_form
                    for word in word_re.findall(category.lower())
                )
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


def filter_vk_communities_semantically(
    api_key: str,
    communities: list[dict[str, object]],
    state_dsn: str,
    positive_keywords: list[str] = _DEFAULT_COMMUNITY_POSITIVE_WORDS,
    negative_keywords: list[str] = _DEFAULT_COMMUNITY_NEGATIVE_WORDS,
    db_timeout_seconds: int = 10
) -> list[dict[str, object]]:
    with connect_pgsql(state_dsn, timeout_seconds=db_timeout_seconds) as conn:
        with conn.cursor() as cur:
            api = VkApi(token=api_key, api_version="5.199").get_api()
            community_bundles = {}
            for community in communities:
                group_data = api.groups.getById(group_id=community.get(
                    "id"), fields="description")["groups"][0]
                community_bundles[community.get("id")] = {
                    "name": group_data["name"],
                    "description": group_data.get("description"),
                }

                owner_id = -int(community.get("id"))
                group_posts = api.wall.get(
                    owner_id=owner_id, count=100)["items"]
                community_bundles[community.get("id")]["pin_post"] = group_posts[0].get(
                    "text") if group_posts[0].get("is_pinned") == 1 else ""

                community_bundles[community.get("id")]["reg_posts"] = []
                for post_id in range(group_posts[0].get("is_pinned") == 1, len(group_posts)):
                    community_bundles[community.get("id")]["reg_posts"].append(
                        group_posts[post_id].get("text"))

            scores = {}
            for community_id, community_bundle in community_bundles.items():
                metadata = " ".join(
                    [community_bundle.get("name", ""),
                     community_bundle.get("description", ""),
                     community_bundle.get("pin_post", "")]
                ).lower()
                reg_posts = " ".join(
                    [*community_bundle.get("reg_posts", [])]).lower()
                scores[community_id] = (
                    sum(30 for term in positive_keywords if term in metadata)
                    - sum(30 for term in negative_keywords if term in metadata)
                    + sum(1 for term in positive_keywords if term in reg_posts)
                    - sum(1 for term in negative_keywords if term in reg_posts)
                )

            filtered_communities = []
            for community in communities:
                if scores.get(community.get("id"), 0) >= 0:
                    filtered_communities.append(community)
                else:
                    group_id = community.get("id")
                    with connect_pgsql(state_dsn, timeout_seconds=db_timeout_seconds) as conn:
                        with conn.cursor() as cur:
                            cur.execute(
                                "INSERT INTO vk_monitor_seen (group_id, reason_discarded) VALUES (%s, 2) "
                                "ON CONFLICT (group_id) DO UPDATE SET reason_discarded = 2",
                                (group_id,),
                            )
    return filtered_communities


def get_new_vk_community_queries(
    queries: list[str],
    openai_api: str,
    openai_api_key: str,
    openai_model: str
) -> list[str]:
    client = OpenAI(base_url=openai_api, api_key=openai_api_key)
    text = ", ".join(queries)
    resp = client.chat.completions.create(
        model=openai_model,
        temperature=1.0,
        messages=[
            {
                "role": "system",
                "content": (
                    "Тебе дан список запросов для поиска сообществ в социальной сети, они расположены в порядке убывания количества полезных сообществ, найденных по каждому запросу. Придумай новые запросы, которые могут быть полезны для поиска сообществ."
                    "Отвечай только новыми запросами, не повторяй старые. Каждый запрос должен быть отделен запятой. Новые запросы должны быть максимально релевантными и разнообразными, не должны быть синонимами друг друга и не должны быть слишком похожими на уже имеющиеся запросы."
                ),
            },
            {"role": "user", "content": text},
        ],
    )

    return resp.choices[0].message.content.strip().split(", ")


def refresh_vk_monitor_polling_list(
    api_key: str,
    state_dsn: str,
    openai_api: str,
    openai_api_key: str,
    openai_model: str,
    count_per_query: int = 100,
    community_category_stop_words: list[str] = _DEFAULT_COMMUNITY_CATEGORY_STOP_WORDS,
    community_positive_keywords: list[str] = _DEFAULT_COMMUNITY_POSITIVE_WORDS,
    community_negative_keywords: list[str] = _DEFAULT_COMMUNITY_NEGATIVE_WORDS,
    db_timeout_seconds: int = 10,
) -> bool:
    default_search_queries = ["подслушано", "район"]
    default_community = 237677627

    with connect_pgsql(state_dsn, timeout_seconds=db_timeout_seconds) as conn:
        with conn.cursor() as cur:
            cur.execute(
                "CREATE TABLE IF NOT EXISTS vk_monitor_queries ("
                "query TEXT PRIMARY KEY, "
                "is_active BOOLEAN NOT NULL DEFAULT TRUE "
                ")"
            )
            cur.execute("SELECT 1 FROM vk_monitor_queries LIMIT 1")
            if cur.fetchone() is None:
                cur.executemany(
                    "INSERT INTO vk_monitor_queries (query) "
                    "VALUES (%s) ON CONFLICT (query) DO NOTHING",
                    [(query,) for query in default_search_queries],
                )
            cur.execute(
                "SELECT query FROM vk_monitor_queries "
                "WHERE is_active = TRUE"
            )
            search_queries = [row[0] for row in cur.fetchall()]

            if not search_queries:
                cur.execute(
                    "SELECT sq.query, COUNT(pl.group_id) AS community_count\n"
                    "FROM vk_monitor_seen_queries sq\n"
                    "JOIN vk_monitor_polling_list pl ON sq.group_id = pl.group_id\n"
                    "GROUP BY sq.query\n"
                    "ORDER BY community_count DESC",
                )
                search_queries = [row[0] for row in cur.fetchall()[:100]]
                new_search_queries = get_new_vk_community_queries(
                    queries=search_queries,
                    openai_api=openai_api,
                    openai_api_key=openai_api_key,
                    openai_model=openai_model
                )
                cur.executemany(
                    "INSERT INTO vk_monitor_queries (query) "
                    "VALUES (%s) ON CONFLICT (query) DO NOTHING",
                    [(query,) for query in new_search_queries],
                )
                search_queries = new_search_queries.copy()

    new_communities = get_new_vk_communities(
        api_key=api_key,
        queries=search_queries,
        state_dsn=state_dsn,
        count_per_query=count_per_query,
        db_timeout_seconds=db_timeout_seconds,
    )
    filtered_stage_1 = filter_vk_communities_by_category_stop_words(
        communities=new_communities,
        state_dsn=state_dsn,
        stop_words=community_category_stop_words,
        db_timeout_seconds=db_timeout_seconds,
    )
    filtered_stage_2 = filter_vk_communities_semantically(
        api_key=api_key,
        communities=filtered_stage_1,
        state_dsn=state_dsn,
        positive_keywords=community_positive_keywords,
        negative_keywords=community_negative_keywords,
        db_timeout_seconds=db_timeout_seconds,
    )

    if not filtered_stage_2:
        return False

    with connect_pgsql(state_dsn, timeout_seconds=db_timeout_seconds) as conn:
        with conn.cursor() as cur:
            cur.execute(
                "CREATE TABLE IF NOT EXISTS vk_monitor_polling_list ("
                "group_id BIGINT PRIMARY KEY"
                ")"
            )
            cur.execute("SELECT 1 FROM vk_monitor_polling_list LIMIT 1")
            if cur.fetchone() is None:
                cur.execute(
                    "INSERT INTO vk_monitor_polling_list (group_id) "
                    "VALUES (%s) ON CONFLICT (group_id) DO NOTHING",
                    (default_community,),
                )
            values = [
                (community.get("id"),)
                for community in filtered_stage_2
                if community.get("id") is not None
            ]
            if values:
                cur.executemany(
                    "INSERT INTO vk_monitor_polling_list (group_id) "
                    "VALUES (%s) "
                    "ON CONFLICT (group_id) DO NOTHING",
                    values,
                )

    return True
