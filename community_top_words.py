from dotenv import load_dotenv
from os import getenv
from vk_api import VkApi
from pymorphy3 import MorphAnalyzer
from re import findall as re_findall
from collections import Counter

load_dotenv()


def community_word_counts(community_id: str | int, api_key: str = getenv("VK_API_KEY")) -> dict[str, int]:
    api = VkApi(token=api_key, api_version="5.199").get_api()

    group_data = api.groups.getById(
        group_id=community_id,
        fields="description",
    )["groups"][0]

    owner_id = -int(group_data["id"])
    group_posts = api.wall.get(
        owner_id=owner_id, count=100, extended=1)["items"]

    pin_post = ""
    reg_posts: list[str] = []
    if group_posts:
        pin_post = group_posts[0].get(
            "text", "") if group_posts[0].get("is_pinned") == 1 else ""
        start_idx = 1 if group_posts[0].get("is_pinned") == 1 else 0
        for post in group_posts[start_idx:]:
            reg_posts.append(post.get("text", ""))

    corpus = " ".join([
        group_data.get("name", ""),
        group_data.get("description", ""),
        pin_post,
        *reg_posts,
    ]).lower()

    morph = MorphAnalyzer()
    words = re_findall(r"[a-zа-яё0-9]+", corpus)
    normalized_words = [morph.parse(word)[0].normal_form for word in words]

    return dict(Counter(normalized_words))


top_words = [
    word
    for word, _ in sorted(
        community_word_counts(community_id="academ").items(),
        key=lambda item: item[1],
        reverse=True,
    )[:100]
]
print(top_words)
