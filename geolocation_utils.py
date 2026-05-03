from urllib.parse import urlencode
from urllib.request import Request, urlopen
from json import loads as json_loads

from openai import OpenAI


def _http_get_json(url: str, headers: dict[str, str]) -> list[dict[str, object]]:
    req = Request(url, headers=headers)
    with urlopen(req, timeout=10) as resp:
        raw = resp.read().decode("utf-8", errors="ignore")
    return json_loads(raw)


def geocode_address(
        address: str,
) -> tuple[float, float] | None:
    address = (address or "").strip()
    if not address:
        return None

    query = urlencode({"q": address, "format": "json", "limit": 1})
    url = f"https://nominatim.openstreetmap.org/search?{query}"
    data = _http_get_json(url, headers={"User-Agent": "trash-monitor"})
    if isinstance(data, list) and data:
        return float(data[0]["lat"]), float(data[0]["lon"])

    return None


def extract_address_with_llm(
        text: str,
        openai_api: str,
        openai_api_key: str,
        openai_model: str,
) -> str | None:
    client = OpenAI(base_url=openai_api, api_key=openai_api_key)
    resp = client.chat.completions.create(
        model=openai_model,
        temperature=0,
        messages=[
            {
                "role": "system",
                "content": (
                        "Найти адрес в тексте (населенный пункт, улица, дом). "
                        "Если адрес найден, ответь только адресом в одной строке. "
                        "Если адрес не найден, ответь только \"нет\"."
                ),
            },
            {"role": "user", "content": text[:1500]},
        ],
    )
    raw = (resp.choices[0].message.content or "").strip()
    if not raw or raw.lower() == "нет":
        return None
    return raw


def enrich_posts_with_coords(
        posts: list[dict[str, object]],
        openai_api: str,
        openai_api_key: str,
        openai_model: str,
) -> list[dict[str, object]]:
    for post in posts:
        if post.get("latitude") is not None and post.get("longitude") is not None:
            continue

        address = extract_address_with_llm(
            text=str(post.get("text", "")),
            openai_api=openai_api,
            openai_api_key=openai_api_key,
            openai_model=openai_model,
        )
        if not address:
            continue

        coords = geocode_address(address)
        if coords is None:
            continue

        lat, lon = coords
        post["latitude"] = lat
        post["longitude"] = lon

    return posts
