import jieba
from ToolUtil import save

def preprocess_text(text):
    text = text.lower()
    words = jieba.lcut(text)
    processed_text = ' '.join(words)
    return processed_text


def get_stopword(p="D:\Code\Buzzword_github\data\stopwords.txt"):
    res = []
    with open(p, "r", encoding="utf-8") as f:
        for line in f:
            if len(line.strip()) == 0:
                continue
            res.append(line.strip())
    return list(set(res))


class GDEX:
    def __init__(self):
        self.length_limit = (10, 25)
        self.stopwords = get_stopword()
        self.common_words = get_stopword(p="D:\Code\Buzzword_github\data\commonwords.txt")
        self.length_valid = 1
        self.pron_valid = 1
        self.start_valid = 1

    def run(self, sentence):
        score = 0
        if self.length_limit[0] <= len(sentence.strip()) <= self.length_limit[1]:
            score += self.length_valid

        prm = ["它", "他", "她", "它们", "他们", "她们", "这个", "这", "那个", "这些", "那些"]
        if not any(substring in sentence for substring in prm):
            score += self.pron_valid

        score += self.contains_stop_chinese_word(sentence)

        first, last = sentence[0], sentence[-1]
        if not any(num in first or num in last for num in '0123456789') and not any(num in first or num in last for num in '，,；：.'):
            score += self.start_valid

        score += self.contains_common_chinese_word(sentence, self.common_words)

        return score

    def contains_common_chinese_word(self, text, common_words):
        segmented_words = [i for i in jieba.lcut(text) if i not in self.stopwords]
        total_words = len(segmented_words)
        if total_words == 0:
            return 0
        common_words = [i for i in segmented_words if i in common_words]
        return len(common_words) / total_words

    def contains_stop_chinese_word(self, text):
        segmented_words = [i for i in jieba.lcut(text)]
        total_words = len(segmented_words)
        common_words = [i for i in segmented_words if i not in self.stopwords]
        return len(common_words) / total_words


if __name__ == "__main__":
    d = GDEX()
    from data.data_cheating_filtered import dataclear
    for key in dataclear.keys():
        print(key)
        example_score = [[i, d.run(i)] for i in dataclear[key]["examples"]]
        dataclear[key]["examples"] = example_score
    save(dataclear, data_path="../data/gdex_score.pickle")