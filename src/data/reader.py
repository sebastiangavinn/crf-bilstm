from typing import List, Tuple

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
                    sent.append(word.lower())
                    tag_seq.append(tag)

        return sentences, tags
