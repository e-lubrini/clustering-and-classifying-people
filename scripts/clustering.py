import argparse
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from collections import Counter
from itertools import product
from nltk import word_tokenize
from scipy import stats
from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.exceptions import NotFittedError
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.utils import shuffle
from statistics import median, mean


class Clusterizer:
    def __init__(self, verbose):
        self.verbose = verbose
        self.model = None
        self.method = None
        self.X = None

    def train(self, texts, descriptions, n_clusters, method):
        raw_X = [f'{descr} {text}' for text, descr in zip(texts, descriptions)]
        self.method = method
        if self.method == 'tf-idf':
            self.X = self.transform_tfidf(raw_X)
        elif self.method == 'tokens':
            self.X = self.transform_tokens(raw_X)

        elif self.method == 'token frequency':
            self.X = self.transform_frequencies(raw_X)
        else:
            raise KeyError(
                f'Method {method} for preprocessing is not found. Try one of these: tf-idf, tokens, token frequency')
        if self.verbose:
            print(f'▶ Training a model with {n_clusters} clusters using {self.method}')
        self.model = KMeans(n_clusters=n_clusters)
        self.model.fit(self.X)

    @staticmethod
    def transform_tfidf(texts):
        tfidf = TfidfVectorizer(tokenizer=word_tokenize)
        X = tfidf.fit_transform(texts)
        return X

    def transform_tokens(self, texts):
        tokenized = [text.split() for text in texts]
        words = sum(tokenized, [])
        vocabulary = list(set(words))
        if self.verbose:
            print(f'▶ Corpus vocabulary contains {len(vocabulary)} words')
        converted_texts = np.zeros((len(texts), len(vocabulary)))
        for i, text in enumerate(tokenized):
            converted_texts[i] = [1 if word in text else 0 for word in vocabulary]
        return converted_texts

    @staticmethod
    def transform_frequencies(texts):
        tokenized = [text.split() for text in texts]
        words = sum(tokenized, [])
        vocabulary = dict(Counter(words))
        converted_texts = np.zeros((len(texts), len(vocabulary)))
        for i, text in enumerate(texts):
            converted_texts[i] = [vocabulary[word] if word in text else 0 for word in vocabulary]

        return converted_texts

    def evaluate(self, true):
        if not self.model:
            raise NotFittedError('Method train should be called first')
        predicted = self.model.labels_
        homogeneity = metrics.homogeneity_score(true, predicted)
        completeness = metrics.completeness_score(true, predicted)
        v_measure = metrics.v_measure_score(true, predicted)
        randscore = metrics.adjusted_rand_score(true, predicted)
        silhouette = metrics.silhouette_score(self.X, predicted)

        return {'Homogeneity': homogeneity, 'Completeness': completeness, 'V measure': v_measure,
                'Adjusted Rand Index': randscore, 'Silhouette coefficient': silhouette}

    @staticmethod
    def visualise(results: dict):
        res_df = pd.DataFrame(results)
        res_df.to_csv('data/Clustering results.csv', index=False)
        for metric in list(results.values())[0]:
            x = []
            y = []
            for result in results:
                x.append(result)
                y.append(results[result][metric])
            plt.plot(x, y, label=metric)
        plt.gcf().set_size_inches(10, 5)
        plt.xlabel('number of clusters and method')
        plt.ylabel('quality')
        plt.title('Metrics')
        plt.legend()
        plt.savefig('data/Clustering visualization.png')


def main(inputpath, verbose):
    if os.path.isfile(inputpath):
        df = pd.read_csv(inputpath)
        df = shuffle(df).reset_index(drop=True)
    else:
        raise FileNotFoundError(f'File {inputpath} is not found. Retry with another name')

    all_results = dict()
    for option in product(['tf-idf', 'tokens', 'token frequency'], [2, 6]):
        clusterizer = Clusterizer(verbose)
        clusterizer.train(df['token_text'], df['token_description'], option[1], option[0])
        if option[1] == 2:
            results = clusterizer.evaluate(df['datatype'])
        else:
            results = clusterizer.evaluate(df['category'])
        all_results[f'{option[1]} clust., {option[0][:10]}'] = results
    Clusterizer.visualise(all_results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Corpus Clusterizer")
    parser.add_argument("--input", type=str, default='data/preprocessed_data.csv',
                        help="path to the preprocessed csv file")
    parser.add_argument('--verbose', help='print out the logs (default: False)', action='store_true')
    args = parser.parse_args()
    main(args.input, args.verbose)
