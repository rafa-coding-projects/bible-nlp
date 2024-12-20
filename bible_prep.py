import re
from typing import Dict, List

import requests

# URL of the Douay-Rheims Bible on Project Gutenberg
url = "https://www.gutenberg.org/files/8300/8300-0.txt"

# Download the text
response = requests.get(url)
bible_text = response.text

# Test the text
# print(bible_text[9541:10800])


def find_all_indices(text: str, word: int) -> List[int]:
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


class BibleExtractor:
    """
    Extract the chapters of a Bible from a URL, rule based

    Attributes:
        url: str
        bible_text: str
        chapters: dict[int, str]

    Methods:
        download_text: None
        extract_chapters: None

    Example of usage:
    ```
    url = "https://www.gutenberg.org/files/8300/8300-0.txt"
    bible_extractor = BibleExtractor(url)
    bible_text = bible_extractor.download_text()
    chapters = bible_extractor.extract_chapters(bible_text)
    """

    def __init__(self, url: str = "https://www.gutenberg.org/files/8300/8300-0.txt"):
        self.url = url

    def download_text(self) -> str:
        """
        Download the text of the Bible

        Returns:
            str: text of the Bible
        """
        response = requests.get(self.url)
        return response.text

    def extract_chapters(self, txt: str) -> Dict[int, str]:
        """
        Extract the chapters of the Bible

        Args:
            txt: Full Bible text string

        Returns:
            dict[int, str]: whole text index where each chapters starts

        """

        indices = find_all_indices(txt, "Chapter")
        chapters = {}
        expand_txt_by = 40
        for i in indices:
            chapter = cut_past_first_number(
                keep_down_to_newline(
                    txt[i - expand_txt_by : i + expand_txt_by],
                    expand_txt_by,
                )
            )

            # clean up the chapter title
            chapter = chapter.replace("\n", "")
            chapter = chapter.replace("(", "")
            chapter = chapter.replace(")", "")

            chapters.update({i: chapter})

        return chapters
