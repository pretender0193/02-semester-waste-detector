from vk_api import VkApi
from postgres_utils import connect_pgsql
from re import compile as re_compile
from pymorphy3 import MorphAnalyzer


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
