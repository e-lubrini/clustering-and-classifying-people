# import modules
import argparse
import os
from collections import Counter
from itertools import product
from statistics import median, mean

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from nltk import word_tokenize
from scipy import stats
from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.exceptions import NotFittedError
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.utils import shuffle


class Clusterizer:
    """
    Class for performing the automatic data clustering
    """

    def __init__(self, verbose: bool):
        """
        :param verbose:  if True, the main steps will be printed during the execution
        """
        self.verbose = verbose
        self.model = None
        self.method = None
        self.X = None

    def train(self, texts: list, descriptions: list, n_clusters: int, method: str):
        """
        Function for training the clustering algorithm (K-Means)
        :param texts: list of texts for clustering
        :param descriptions: list of descriptions used along with the texts for clustering
        :param n_clusters: the desired number of clusters
        :param method: the desired method of vectorization: tf-idf, token presence, token frequency
        """
        raw_X = [f'{descr} {text}' for text, descr in zip(texts, descriptions)]
        self.method = method
        # transforming the data
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
        # initializing the model
        self.model = KMeans(n_clusters=n_clusters)
        # fitting the model with the transformed data
        self.model.fit(self.X)

    @staticmethod
    def transform_tfidf(texts: list):
        """
        Supplementary function for texts vectorization using Tf-Idf
        :param texts: list of texts to transform
        :return: sparse matrix with tf-idf values for each text
        """
        tfidf = TfidfVectorizer(tokenizer=word_tokenize)
        X = tfidf.fit_transform(texts)
        return X

    def transform_tokens(self, texts: list):
        """
        Supplementary function for texts vectorization using token presence
        :param texts: list of texts to transform
        :return: list of lists of transformed tokens
        """
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
    def transform_frequencies(texts: list):
        """
        Supplementary function for texts vectorization using token frequency
        :param texts: list of texts to transform
        :return: list of lists of transformed tokens
        """
        tokenized = [text.split() for text in texts]
        words = sum(tokenized, [])
        vocabulary = dict(Counter(words))
        converted_texts = np.zeros((len(texts), len(vocabulary)))
        for i, text in enumerate(texts):
            converted_texts[i] = [vocabulary[word] if word in text else 0 for word in vocabulary]

        return converted_texts

    def evaluate(self, true: list):
        """
        Function for clustering algorithm evaluation
        :param true: list with the expected clusters for each text
        :return: dictionary with Homogeneity, Completeness, V measure, Adjusted Rand Index and Silhouette coefficient
                 scores
        """
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
        """
        Function for the results visualization. Stores the plot with the results into the folder named 'data'
        :param results: dictionary of dictionaries with 5 scores for each setting
        """
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


def main(inputpath: str, verbose: bool):
    """
    Function that starts after calling the script
    :param inputpath: path to the preprocessed csv file
    :param verbose: if True, the main steps will be printed during the execution
    """
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
