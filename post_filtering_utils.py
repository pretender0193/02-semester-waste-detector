from sentence_transformers import SentenceTransformer
from numpy import array as np_array, float32 as np_float32
from faiss import IndexFlatIP as FaissIndexFlatIP
from openai import OpenAI


def filter_posts_with_faiss(
    posts: list[str],
    target_phrase: str = "мусор около железной дороги",
    threshold: float = 0.3
) -> list[str]:
    embedder = SentenceTransformer(
        "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
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
    openai_api: str,
    openai_api_key: str,
    openai_model: str
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

    client = OpenAI(base_url=openai_api, api_key=openai_api_key)
    filtered_posts = []
    for post in posts:
        text = post[:1500]
        resp = client.chat.completions.create(
            model=openai_model,
            temperature=0,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Определи, описан ли в тексте конкретный случай, где мусор в данный момент лежит рядом с железнодорожной инфраструктурой (путь, станция, переезд)."
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
