import os
import re
import html
import time
import requests


SEARXNG_URL = os.getenv("SEARXNG_URL", "http://searxng:8080")
SEARCH_TIMEOUT = 15
SEARCH_LIMIT = 5
SEARCH_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"
)

_last_wikipedia_request_time = 0.0
_wikipedia_request_delay = 1.0
MAX_SEARCH_RPS = 20
_min_wikipedia_interval = 1.0 / MAX_SEARCH_RPS


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


def _search_wikipedia(query: str) -> list[dict]:
    """Search Wikipedia (Estonian) for topic information with rate limiting."""
    global _last_wikipedia_request_time, _wikipedia_request_delay
    
    if not query or len(query) < 2:
        return []

    max_retries = 3
    for attempt in range(max_retries):
        # enforce both adaptive wikipedia delay and a hard cap from RPS
        effective_interval = max(_wikipedia_request_delay, _min_wikipedia_interval)
        time_since_last_request = time.time() - _last_wikipedia_request_time
        if time_since_last_request < effective_interval:
            time.sleep(effective_interval - time_since_last_request)

        try:
            _last_wikipedia_request_time = time.time()
            
            r = requests.get(
                "https://et.wikipedia.org/w/api.php",
                params={
                    "action": "query",
                    "list": "search",
                    "srsearch": query,
                    "srnamespace": 0,
                    "srlimit": 3,
                    "format": "json",
                },
                headers={
                    "User-Agent": SEARCH_USER_AGENT,
                },
                timeout=SEARCH_TIMEOUT,
            )
            r.raise_for_status()
            data = r.json()

            results = []
            for item in data.get("query", {}).get("search", []):
                title = item.get("title", "").strip()
                snippet = _strip_html(item.get("snippet", "")).strip()
                if title and snippet:
                    results.append(
                        {
                            "title": title,
                            "url": f"https://et.wikipedia.org/wiki/{title.replace(' ', '_')}",
                            "content": snippet,
                        }
                    )

            return results
            
        except requests.exceptions.HTTPError as exc:
            if exc.response.status_code in (429, 403):
                # Respect server-supplied Retry-After header when present.
                retry_after = None
                try:
                    hdr = exc.response.headers.get("Retry-After")
                    if hdr:
                        if hdr.isdigit():
                            retry_after = float(int(hdr))
                        else:
                            from email.utils import parsedate_to_datetime
                            dt = parsedate_to_datetime(hdr)
                            if dt is not None:
                                retry_after = max(0.0, (dt - __import__("datetime").datetime.utcnow()).total_seconds())
                except Exception:
                    retry_after = None

                if retry_after is not None:
                    _wikipedia_request_delay = max(retry_after, _min_wikipedia_interval)
                    print(f"[SEARCH] Wikipedia 429/403 received, honoring Retry-After={retry_after}s")
                else:
                    _wikipedia_request_delay = min(max(_wikipedia_request_delay * 1.5, _min_wikipedia_interval), 5.0)
                    print(f"[SEARCH] Wikipedia blocked (rate limit/policy), retry {attempt+1}/{max_retries}, backoff={_wikipedia_request_delay}s")

                if attempt < max_retries - 1:
                    time.sleep(min(_wikipedia_request_delay, 2 ** attempt))
                    continue
            print(f"[SEARCH] Wikipedia error: {exc}")
            return []
        except Exception as exc:
            print(f"[SEARCH] Wikipedia error: {exc}")
            return []
    
    print("[SEARCH] Wikipedia max retries exceeded")
    return []


def _search_weather(query: str = "") -> list[dict]:
    """Get weather for any Estonian city."""
    cities = {
        "tallinn": (59.4370, 24.7536),
        "tartu": (58.3806, 26.7219),
        "pärnu": (58.3853, 24.5014),
        "haapsalu": (58.9455, 23.5447),
        "narva": (59.3742, 28.1948),
        "kuressaare": (58.2548, 22.4898),
        "rakvere": (59.3528, 26.3597),
        "kärdla": (59.0097, 23.1940),
        "viljandi": (58.3645, 25.5889),
        "põltsamaa": (58.7514, 25.9706),
        "estonia": (58.5953, 25.0136),
        "tallinna": (59.4370, 24.7536),
        "tallinnas": (59.4370, 24.7536),
    }

    lat, lon = 59.4370, 24.7536
    city_name = "Tallinn"
    query_lower = query.lower()

    for city_key, coords in cities.items():
        if city_key in query_lower:
            lat, lon = coords
            city_name = city_key.replace("a", "").replace("s", "").title()
            break

    try:
        r = requests.get(
            "https://api.open-meteo.com/v1/forecast",
            params={
                "latitude": lat,
                "longitude": lon,
                "current": "temperature_2m,weather_code,wind_speed_10m",
                "temperature_unit": "celsius",
                "timezone": "Europe/Tallinn",
            },
            timeout=SEARCH_TIMEOUT,
        )
        r.raise_for_status()
        data = r.json()

        current = data.get("current", {})
        temp = current.get("temperature_2m", "?")
        wind = current.get("wind_speed_10m", "?")
        weather_code = current.get("weather_code", 0)

        conditions = {
            0: "selge",
            1: "peaaegu selge",
            2: "osaliselt pilvine",
            3: "pilvine",
            45: "udu",
            48: "külmuva udu",
            51: "kerge vihmane",
            61: "vihmane",
            71: "kerge lumesadu",
            81: "intensiivne vihm",
        }

        condition = conditions.get(weather_code, "tundmatu")
        result_text = f"{city_name} ilm: {temp}°C, {condition}, tuul {wind} m/s"

        return [
            {
                "title": f"{city_name} ilm",
                "url": "https://open-meteo.com/",
                "content": result_text,
            }
        ]

    except Exception as exc:
        print(f"[SEARCH] Weather API error: {exc}")
        return []


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

    if "ilm" in cleaned_query.lower() or "weather" in cleaned_query.lower():
        weather_results = _search_weather(cleaned_query)
        if weather_results:
            print("[SEARCH] fallback -> weather API")
            return weather_results[:limit]

    wiki_results = _search_wikipedia(cleaned_query)
    if wiki_results:
        print("[SEARCH] fallback -> wikipedia API")
        return wiki_results[:limit]

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