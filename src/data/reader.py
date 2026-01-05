from typing import List, Tuple
from src.utils.text_cleaner import clean_token

class DataReader:
    @staticmethod
    def read_conll(path: str) -> Tuple[List[List[str]], List[List[str]]]:
        sentences, tags = [], []
        sent, tag_seq = [], []

        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()

                if not line:
                    if sent:
                        sentences.append(sent)
                        tags.append(tag_seq)
                        sent, tag_seq = [], []
                else:
                    word, tag = line.split()[0], line.split()[-1]
                    word = clean_token(word)   # 👈 PEMBERSIHAN DI SINI
                    sent.append(word)
                    tag_seq.append(tag)

        return sentences, tags
