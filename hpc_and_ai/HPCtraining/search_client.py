import os
import requests


SEARXNG_URL = os.getenv("SEARXNG_URL", "http://searxng:8080")


def search_web(query: str, limit: int = 5, language: str = "et", safesearch: int = 2):
    params = {
        "q": query,
        "format": "json",
        "language": language,
        "safesearch": safesearch,
    }

    r = requests.get(f"{SEARXNG_URL}/search", params=params, timeout=15)
    r.raise_for_status()
    data = r.json()

    results = []
    for item in data.get("results", [])[:limit]:
        results.append(
            {
                "title": item.get("title", ""),
                "url": item.get("url", ""),
                "content": item.get("content", "") or item.get("snippet", "") or "",
            }
        )
    return results


def build_search_context(results: list[dict]) -> str:
    if not results:
        return "Otsing ei andnud häid tulemusi."

    blocks = []
    for i, r in enumerate(results, start=1):
        blocks.append(
            f"[{i}] {r['title']}\n{r['content']}\n{r['url']}"
        )
    return "\n\n".join(blocks)