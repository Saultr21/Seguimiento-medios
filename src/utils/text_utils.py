import re

def clean_text(text):
    text = re.sub(r"\[.*?\]", "", text)
    text = re.sub(r"\n+", "\n", text.strip())
    text = re.sub(r"(\w)\n(\w)", r"\1 \2", text)
    text = re.sub(r"\s+([.,;!?])", r"\1", text)
    text = re.sub(r"\.{3,}", " ", text)

    return text