import argparse
import collections
import os
from statistics import median, mean

import pandas as pd
from nltk import word_tokenize
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.exceptions import NotFittedError
from sklearn.feature_extraction.text import TfidfVectorizer


class Clusterizer:
    def __init__(self, verbose, maxlen=200):
        self.verbose = verbose
        self.maxlen = maxlen
        self.model = None

    def train(self, texts, descriptions, n_clusters, method):
        raw_X = [f'{descr} {text}' for text, descr in zip(texts, descriptions)]
        if method == 'tf-idf':
            X = self.transform_tfidf(raw_X)
        elif method == 'tokens':
            X = self.transform_tokens(raw_X)

        elif method == 'token frequency':
            X = self.transform_frequencies(raw_X)
        else:
            raise KeyError(
                f'method {method} for preprocessing is not found. Try one of these: tf-idf, tokens, tokens frequency')

        model = KMeans(n_clusters=n_clusters)
        model.fit(X)

    def transform_tfidf(self, texts):
        tfidf = TfidfVectorizer(tokenizer=word_tokenize)
        X = tfidf.fit_transform(texts)
        return X

    def transform_tokens(self, texts):
        tokenized = [text.split() for text in texts]
        words = sum(tokenized, [])
        vocabulary = list(set(words))
        if self.verbose:
            print(f'Corpus vocabulary contains {len(vocabulary)} words')
        tok2int = collections.defaultdict(lambda: len(tok2int))
        tok2int['<unk>'] = 0
        tok2int['<pad>'] = 1
        lens = [len(text) for text in texts]
        maximlen = max(lens)
        meanlen = mean(lens)
        modelen = stats.mode(lens)
        medianlen = median(lens)
        if self.verbose:
            print(
                f'Maximal length of a text is {maximlen},\n Average length is {meanlen},\n Mode of a corpus is {modelen},\n Median is {medianlen}')
            print(f'Sequences will be padded/truncated to the length {self.maxlen}')
        padded_texts = [self._resize_sentence(text, self.maxlen) for text in texts]
        converted_texts = [[tok2int.get(word, 0) for word in text] for text in padded_texts]
        return converted_texts

    def transform_frequencies(self, texts):
        return texts

    @staticmethod
    def _resize_sentence(text, maxlen):
        length = len(text)
        if length > maxlen:
            text = text[:maxlen]
        elif length < maxlen:
            to_pad = maxlen - length
            context = ['<pad>' for _ in range(to_pad)]
            text.extend(context)
        return text

    def evaluate(self):
        if not self.model:
            raise NotFittedError('Method train should be called first')
        pass

    def visualise(self):
        if not self.model:
            raise NotFittedError('Method train should be called first')
        pass


def main(inputpath, outputpath, verbose):
    if os.path.isfile(inputpath):
        df = pd.read_csv(inputpath)
    else:
        raise FileNotFoundError(f'File {inputpath} is not found. Retry with another name')

    clusterizer = Clusterizer(verbose)
    model = clusterizer.train(df['token_text'], df['token_description'], 2, 'tf-idf')
    results = model.evaluate()
    model.visualise()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Corpus Preprocessor")
    parser.add_argument("--input", type=str, default='data.csv',
                        help="path to the input file for preprocessing (a csv file obtained after running the corpus_extraction.py script)")
    parser.add_argument("--output", type=str, default='preprocessed_data.csv',
                        help="desired path to the output csv file")
    parser.add_argument('--verbose', help='print out the logs (default: False)', action='store_true')
    args = parser.parse_args()
    main(args.input, args.output, args.verbose)
