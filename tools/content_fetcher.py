import requests
from bs4 import BeautifulSoup


MAX_CHARS_PER_ARTICLE = 2500   # critical limit


def fetch_content(search_results):
    pages = []

    for result in search_results:
        url = result.get("url")
        if not url:
            continue

        try:
            response = requests.get(
                url,
                timeout=15,
                headers={"User-Agent": "Mozilla/5.0"}
            )
            response.raise_for_status()

            soup = BeautifulSoup(response.text, "html.parser")

            # Remove noise
            for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
                tag.decompose()

            paragraphs = [
                p.get_text(strip=True)
                for p in soup.find_all("p")
                if len(p.get_text(strip=True)) > 50
            ]

            # 🔑 CRITICAL PART — limit content size
            text = ""
            for p in paragraphs:
                if len(text) + len(p) > MAX_CHARS_PER_ARTICLE:
                    break
                text += p + " "

            if len(text) < 400:
                continue  # skip thin pages

            pages.append({
                "url": url,
                "content": text.strip()
            })

        except Exception:
            continue

    return pages
