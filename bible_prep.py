import re

import requests

# URL of the Douay-Rheims Bible on Project Gutenberg
url = "https://www.gutenberg.org/files/8300/8300-0.txt"

# Download the text
response = requests.get(url)
bible_text = response.text

# Display the first 500 characters
print(bible_text[9541:10500])


def find_all_indices(text: str, word: int) -> list[int]:
    """
    Find all the indices of a word in a text

    Args:
        text: str
        word: str

    Returns:
        list[int]
    """
    indices = []
    start = 0
    while start < len(text):
        index = text.find(word, start)
        if index == -1:
            break
        indices.append(index)
        start = index + len(word)
    return indices


def cut_past_first_number(text: str, pattern: str = "Chapter") -> str:
    """
    Cut the text past the first number after a pattern

    Args:
        text: str
        pattern: str

    Returns:
        str
    """
    match = re.search(f"{pattern} \d+", text)
    if match:
        return text[: match.end()]
    return text


def keep_down_to_newline(text: str, index: int) -> str:
    """
    Keep the text from the index to the previous newline

    Args:
        text: str
        index: int

    Returns:
        str
    """
    for i in range(index, -1, -1):
        if text[i] == "\n":
            return text[i + 1 :]
    return text


# Get where all chapters start
indices = find_all_indices(bible_text, "Chapter")

# Get the text of each chapter
chapters = {}
expand_txt_by = 40
for i in indices:
    chapter = cut_past_first_number(
        keep_down_to_newline(
            bible_text[i - expand_txt_by : i + expand_txt_by], expand_txt_by
        )
    )
    chapters.update({i: chapter})
