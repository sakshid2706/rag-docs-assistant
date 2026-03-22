import json
import os


INPUT_PATH = "../data/processed/docs.json"
OUTPUT_PATH = "../data/processed/cleaned_docs.json"


def clean_text_v1(text):
    """
    Basic text cleaning:
    - Remove extra spaces
    - Remove very short noisy lines
    """
    lines = text.split("\n")

    cleaned_lines = []
    for line in lines:
        line = line.strip()

        # Skip empty or very small noise lines
        if len(line) < 3:
            continue

        cleaned_lines.append(line)

    return "\n".join(cleaned_lines)

def clean_text_v2(text):
    # Improved version (noise removal)
    lines = text.split("\n")

    cleaned_lines = []
    for line in lines:
        line = line.strip()

        if len(line) < 30:
            continue

        if "Created On:" in line or "Last Updated" in line:
            continue

        if line.count(" ") < 5:
            continue

        cleaned_lines.append(line)

    return "\n".join(cleaned_lines)


def extract_title(content):
    """
    Extract title from content.
    Current strategy:
    - First meaningful line
    """
    lines = content.split("\n")

    for line in lines:
        if len(line.strip()) > 5:
            return line.strip()

    return "Untitled Document"


def clean_data():
    # Ensure output directory exists
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

    # Load raw processed data
    with open(INPUT_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    cleaned = []

    for doc in data:
        raw_content = doc.get("content", "")
        url = doc.get("url", "")
        raw_path = doc.get("raw_path", "")

        if not raw_content:
            continue

        # Clean text
        # content = clean_text(raw_content)
        content = clean_text_v2(raw_content)

        # Extract title
        title = extract_title(content)

        cleaned.append({
            "title": title,
            "url": url,
            "raw_path": raw_path,
            "content": content
        })

    # Save cleaned dataset
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(cleaned, f, indent=2, ensure_ascii=False)

    print(f"Cleaning complete! {len(cleaned)} documents saved.")


if __name__ == "__main__":
    clean_data()