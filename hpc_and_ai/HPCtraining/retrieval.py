def build_search_context(results: list[dict]) -> str:
    blocks = []
    for i, r in enumerate(results, start=1):
        blocks.append(
            f"[{i}] {r['title']}\n{r['snippet']}\n{r['url']}"
        )
    return "\n\n".join(blocks)
