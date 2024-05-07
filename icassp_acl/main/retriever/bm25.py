import jieba.posseg as pseg
from rank_bm25 import BM25Okapi


class BM25Tokenizer:

    def __init__(self, stopword_list):
        self.stopword_list = stopword_list
        self.stop_flag = ['x', 'c', 'u', 'd', 'p', 't', 'uj', 'm', 'f', 'r']

    def tokenize(self, x):
        result = None
        if type(x) == str:
            result = list(pseg.cut(x))
            result = [word for word, flag in result if flag not in self.stop_flag and word not in self.stopword_list]
        elif type(x) == list:
            result = []
            for sentence in x:
                words = list(pseg.cut(sentence))
                result.append([
                    word
                    for word, flag in words
                    if flag not in self.stop_flag and word not in self.stopword_list
                ])
        else:
            Exception(f'Error input type: {type(x)}')
        return result


class BM25:
    def __init__(self, tokenizer: BM25Tokenizer, corpus):
        self.tokenizer = tokenizer
        self.corpus = corpus
        tokenized_corpus = tokenizer.tokenize(corpus)
        self.bm25 = BM25Okapi(tokenized_corpus)

    def get_top_n(self, query, n=10):
        query = self.tokenizer.tokenize(query.replace('用户说：', '').replace('系统说：', '').replace('最后一轮', ''))
        return self.bm25.get_top_n(query, self.corpus, n=n)
