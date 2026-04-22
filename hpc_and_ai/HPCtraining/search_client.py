import requests

class SearxngClient:
    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip("/")

    def search(self, query: str, language: str = "et", safesearch: int = 2, limit: int = 5):
        params = {
            "q": query,
            "format": "json",
            "language": language,
            "safesearch": safesearch,
        }
        r = requests.get(f"{self.base_url}/search", params=params, timeout=15)
        r.raise_for_status()
        data = r.json()
        results = data.get("results", [])[:limit]
        return [
            {
                "title": x.get("title", ""),
                "url": x.get("url", ""),
                "snippet": x.get("content", "") or x.get("snippet", ""),
            }
            for x in results
        ]
