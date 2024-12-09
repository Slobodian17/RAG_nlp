import re
import pandas as pd

class Citation:
    def __init__(self, cleaned_text):
        """
        Initialize with cleaned text.
        :param cleaned_text: The cleaned text from the PDF.
        """
        self.cleaned_text = cleaned_text
        self.chapters = self.extract_chapter_names()
        self.chapter_data = self.extract_chapter_data()
        self.df = self.to_dataframe()

    def extract_chapter_names(self):
        """
        Extract chapter names using the provided `extract_chapter_names` function.
        :return: List of chapter names.
        """
        toc_match = re.search(r"Table of contents(.*?)Table of contents", self.cleaned_text, re.DOTALL)
        if not toc_match:
            raise ValueError("Table of Contents section not found.")

        toc_section = toc_match.group(1)
        chapter_names = re.findall(r"(.*?)\d+", toc_section)
        return [name.strip() for name in chapter_names if name.strip()]

    def extract_chapter_data(self):
        """
        Extract full text for each chapter from the second occurrence of the chapter name
        to the second occurrence of the next chapter name, or to the end if it's the last chapter.
        :return: A list of tuples (chapter_name, plain_text).
        Second occurence because first in text is in table of contents  :)
        """
        chapter_data = []

        for i, chapter in enumerate(self.chapters):
            current_chapter_pattern = re.escape(chapter)
            next_chapter_pattern = re.escape(self.chapters[i + 1]) if i + 1 < len(self.chapters) else None

            if next_chapter_pattern:
                matches = list(re.finditer(rf"{current_chapter_pattern}(.*?){next_chapter_pattern}", self.cleaned_text, re.DOTALL))
            else:
                matches = list(re.finditer(rf"{current_chapter_pattern}(.*)", self.cleaned_text, re.DOTALL))

            if len(matches) >= 2:
                start = matches[1].start(1)
                end = matches[1].end(1)

                if next_chapter_pattern:
                    next_match = re.search(rf"{next_chapter_pattern}", self.cleaned_text[end:])
                    if next_match:
                        end += next_match.start()

                chapter_content = self.cleaned_text[start:end].strip()
            else:
                chapter_content = ""

            if i == len(self.chapters) - 1:
                last_chapter_matches = list(re.finditer(rf"{current_chapter_pattern}", self.cleaned_text))

                if len(last_chapter_matches) >= 2:
                    start = last_chapter_matches[1].start()
                    chapter_content = self.cleaned_text[start:].strip()

            chapter_data.append((chapter, chapter_content))

        return chapter_data

    def to_dataframe(self):
        """
        Converts the chapter data into a Pandas DataFrame.
        :return: DataFrame with 'Chapter' and 'Content' columns.
        """
        df = pd.DataFrame(self.chapter_data, columns=['Chapter', 'Content'])
        return df

    def search_citate(self, retrieved_chunks):
        """
        Finds the chapter name for each chunk in the content and returns a formatted string.

        Args:
            retrieved_chunks (list): List of text chunks to search in the chapters' content.

        Returns:
            str: A formatted string of citations, each chunk starting on a new line.
        """
        citations = []

        for idx, chunk in enumerate(retrieved_chunks, start=1):
            for _, row in self.df.iterrows():
                chapter_name = row['Chapter']
                chapter_content = row['Content']

                if chunk in chapter_content:
                    citations.append(f"chunk [{idx}] from chapter: {chapter_name}")
                    break

        return "\n".join(citations)
