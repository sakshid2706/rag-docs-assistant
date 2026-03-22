import requests
from bs4 import BeautifulSoup
import json
from tqdm import tqdm
import os

BASE_URL = "https://pytorch.org/docs/stable/"

# Start with a few important pages (expand later)
URLS = [
    "https://pytorch.org/docs/stable/tensors.html",
    "https://pytorch.org/docs/stable/autograd.html",
    "https://pytorch.org/docs/stable/optim.html",
    "https://pytorch.org/docs/stable/nn.html",
]

def fetch_page(url):
    try:
        response = requests.get(url)
        response.raise_for_status()
        return response.text
    except Exception as e:
        print(f"Error fetching {url}: {e}")
        return None

def extract_text(html):
    soup = BeautifulSoup(html, "html.parser")

    # Remove unwanted tags
    for tag in soup(["script", "style", "nav", "footer", "header"]):
        tag.decompose()

    # 🔥 Try multiple selectors (robust scraping)
    main_content = (
        soup.find("div", {"role": "main"}) or
        soup.find("div", {"class": "document"}) or
        soup.find("main") or
        soup.body
    )

    if not main_content:
        return ""

    text = main_content.get_text(separator="\n")

    # Clean text (important)
    lines = []
    for line in text.split("\n"):
        line = line.strip()
        if len(line) > 30:   # remove noisy short lines
            lines.append(line)

    return "\n".join(lines)

def scrape():
    data = []

    os.makedirs("../data/raw", exist_ok=True)
    os.makedirs("../data/processed", exist_ok=True)

    for url in tqdm(URLS):
        html = fetch_page(url)
        if not html:
            continue

        # ✅ Save raw HTML
        # filename = url.split("/")[-1] or "index"
        # filepath = f"../data/raw/{filename}.html"

        filename = url.rstrip("/").split("/")[-1]

        if not filename.endswith(".html"):
            filename += ".html"

        filepath = f"../data/raw/{filename}"

        with open(filepath, "w", encoding="utf-8") as f:
            f.write(html)

        # Existing logic
        text = extract_text(html)

        data.append({
            "url": url,
            "content": text,
            "raw_path": filepath   # 🔥 add this (important)
        })

    with open("../data/processed/docs.json", "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print("Scraping complete!")

if __name__ == "__main__":
    scrape()