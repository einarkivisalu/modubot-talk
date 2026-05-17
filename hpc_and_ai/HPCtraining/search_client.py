import os
import re
import html
import requests


SEARXNG_URL = os.getenv("SEARXNG_URL", "http://searxng:8080")
SEARCH_TIMEOUT = 15
SEARCH_LIMIT = 5
SEARCH_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"
)


def build_search_query(question: str) -> str:
    query = re.sub(r"^(tere|tsau|hei|palun|aitäh)[,\s]+", "", question.strip(), flags=re.IGNORECASE)
    query = re.sub(r"\s+", " ", query)
    return query.strip(" ?!.,")


def _candidate_searxng_urls() -> list[str]:
    candidates = [
        SEARXNG_URL,
        "http://localhost:8080",
        "http://127.0.0.1:8080",
        "http://host.docker.internal:8080",
        "http://searxng:8080",
    ]

    unique: list[str] = []
    seen: set[str] = set()
    for url in candidates:
        normalized = url.rstrip("/")
        if normalized and normalized not in seen:
            seen.add(normalized)
            unique.append(normalized)
    return unique


def _search_searxng(base_url: str, query: str, language: str = "et", safesearch: int = 2) -> list[dict]:
    r = requests.get(
        f"{base_url}/search",
        params={
            "q": query,
            "format": "json",
            "language": language,
            "safesearch": safesearch,
        },
        headers={"User-Agent": SEARCH_USER_AGENT},
        timeout=SEARCH_TIMEOUT,
    )
    r.raise_for_status()
    data = r.json()

    results = []
    for item in data.get("results", [])[:SEARCH_LIMIT]:
        results.append(
            {
                "title": item.get("title", "").strip(),
                "url": item.get("url", "").strip(),
                "content": (item.get("content", "") or item.get("snippet", "")).strip(),
            }
        )
    return results


def _strip_html(value: str) -> str:
    return re.sub(r"<[^>]+>", "", html.unescape(value)).strip()


def _search_duckduckgo(query: str) -> list[dict]:
    headers = {
        "User-Agent": SEARCH_USER_AGENT,
        "Accept-Language": "et-EE,et;q=0.9",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Referer": "https://duckduckgo.com/",
    }
    
    r = requests.get(
        "https://duckduckgo.com/html/",
        params={"q": query, "kl": "et-et"},
        headers=headers,
        timeout=SEARCH_TIMEOUT,
    )
    r.raise_for_status()
    page = r.text

    if not page or len(page) < 100:
        print(f"[SEARCH] DuckDuckGo returned empty/too small response ({len(page)} bytes)")
        return []

    title_matches = re.findall(
        r'class="result__a"[^>]*href="([^"]+)"[^>]*>(.*?)</a>',
        page,
        flags=re.IGNORECASE | re.DOTALL,
    )

    if not title_matches:
        print(f"[SEARCH] DuckDuckGo returned no matches. Sample HTML: {page[1000:1200]}")
        return []

    snippet_matches = re.findall(
        r'class="result__snippet"[^>]*>(.*?)</(?:a|div)>',
        page,
        flags=re.IGNORECASE | re.DOTALL,
    )

    results: list[dict] = []
    for index, (url, title) in enumerate(title_matches[:SEARCH_LIMIT]):
        snippet = snippet_matches[index] if index < len(snippet_matches) else ""
        results.append(
            {
                "title": _strip_html(title),
                "url": html.unescape(url).strip(),
                "content": _strip_html(snippet),
            }
        )
    return results


def search_web(query: str, limit: int = 5, language: str = "et", safesearch: int = 2) -> list[dict]:
    cleaned_query = build_search_query(query)
    last_searxng_error = None

    for base_url in _candidate_searxng_urls():
        try:
            results = _search_searxng(base_url, cleaned_query, language, safesearch)
            if results:
                print(f"[SEARCH] searxng -> {base_url}")
                return results[:limit]
        except Exception as exc:
            last_searxng_error = exc

    if last_searxng_error is not None:
        print(f"[SEARCH] searxng unavailable, using fallback: {last_searxng_error}")

    try:
        results = _search_duckduckgo(cleaned_query)
        if results:
            print("[SEARCH] fallback -> duckduckgo")
            return results[:limit]
    except Exception as exc:
        print(f"[SEARCH] duckduckgo unavailable: {exc}")

    print("[SEARCH] all methods failed")
    return []


def build_search_context(results: list[dict]) -> str:
    if not results:
        return "Otsing ei andnud häid tulemusi."

    blocks = []
    for i, r in enumerate(results, start=1):
        blocks.append(
            f"[{i}] {r['title']}\n{r['content']}\n{r['url']}"
        )
    return "\n\n".join(blocks)