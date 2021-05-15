import argparse
import os
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from nltk import word_tokenize
from sklearn.exceptions import NotFittedError
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Perceptron
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

warnings.filterwarnings("ignore")


class Classifier:
    def __init__(self, verbose):
        self.verbose = verbose
        self.model = None

    def train(self, X, y):
        if self.verbose:
            print(f'Training the model with the following classes: {", ".join(np.unique(y))}')
        self.model = Perceptron(penalty='l1', alpha=0.001)
        self.model.fit(X, y)

    def predict(self, X):
        if not self.model:
            raise NotFittedError('Method train should be called first')
        if self.verbose:
            print(f'Predicting...')
        pred = self.model.predict(X)
        return pred

    def compute_scores(self, expected, predicted, num_classes):
        if self.verbose:
            print('Saving the scores')
        # get classes
        classes = self.model.classes_

        # get confusion matrix
        conf_matrix = confusion_matrix(expected, predicted)
        plt.figure(figsize=(10, 10))
        plot = sns.heatmap(conf_matrix, annot=True, cmap='Blues', xticklabels=classes, yticklabels=classes)
        plt.savefig(f'Confusion matrix {num_classes} classes.png')
        plt.clf()

        # get precision, recall, F1
        report = classification_report(predicted, expected)
        report_dict = classification_report(predicted, expected, output_dict=True)
        df = pd.DataFrame(report_dict).transpose().round(2)
        df.to_csv(f'Scores {num_classes} classes.csv', index=False)

        return conf_matrix, report, classes

    @staticmethod
    def accuracy_per_class(conf_matrix):
        output = []
        for idx in range(conf_matrix.shape[0]):
            true_negative = np.sum(np.delete(np.delete(conf_matrix, idx, axis=0), idx, axis=1))
            true_positive = conf_matrix[idx, idx]
            res = (true_negative + true_positive) / np.sum(conf_matrix)
            output.append(res)

        return output

    def visualise(self, conf_matrix2, conf_matrix6, cats2, cats6):
        if self.verbose:
            print('Saving the visualization')
        acc2 = self.accuracy_per_class(conf_matrix2)
        acc6 = self.accuracy_per_class(conf_matrix6)

        c = sns.set_palette(sns.color_palette(["#33FFC6"] * 10))
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        fig.suptitle('Accuracy per class')
        sns.barplot(ax=axes[0], x=cats2, y=acc2, color=c)
        axes[0].set_title('Accuracy per subcategory')
        sns.barplot(ax=axes[1], x=cats6, y=acc6, color=c)
        axes[1].set_title('Accuracy per category')
        fig.savefig('Accuracy visualization.png')
        fig.clf()

    def split_convert_data(self, texts, labels):
        X_train, X_test, y_train, y_test = train_test_split(texts, labels, test_size=0.3, random_state=42,
                                                            stratify=labels)
        tfidf = TfidfVectorizer(tokenizer=word_tokenize)
        tfidf.fit(X_train)
        if self.verbose:
            print(f'Number of Tf-Idf features: {len(tfidf.get_feature_names())}')
        X_train = tfidf.transform(X_train)
        X_test = tfidf.transform(X_test)
        return X_train, X_test, y_train, y_test


def main(inputpath, verbose):
    if os.path.isfile(inputpath):
        df = pd.read_csv(inputpath)
        df = shuffle(df).reset_index(drop=True)
    else:
        raise FileNotFoundError(f'File {inputpath} is not found. Retry with another name')
    classifier2 = Classifier(verbose)
    X_train2, X_test2, y_train2, y_test2 = classifier2.split_convert_data(df['token_text'], df['datatype'])
    classifier2.train(X_train2, y_train2)
    predicted2 = classifier2.predict(X_test2)
    conf_matrix2, class_report2, classes2 = classifier2.compute_scores(y_test2, predicted2, 2)

    classifier6 = Classifier(verbose)
    X_train6, X_test6, y_train6, y_test6 = classifier6.split_convert_data(df['token_text'], df['category'])
    classifier6.train(X_train6, y_train6)
    predicted6 = classifier6.predict(X_test6)
    conf_matrix6, class_report6, classes6 = classifier6.compute_scores(y_test6, predicted6, 6)

    classifier6.visualise(conf_matrix2, conf_matrix6, classes2, classes6)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Corpus Classifier")
    parser.add_argument("--input", type=str, default='preprocessed_data.csv',
                        help="path to the preprocessed csv file")
    parser.add_argument('--verbose', help='print out the logs (default: False)', action='store_true')
    args = parser.parse_args()
    main(args.input, args.verbose)
