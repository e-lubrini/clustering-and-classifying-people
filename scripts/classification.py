# import modules
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
    """
    Class for automatic texts classification using perceptron
    """

    def __init__(self, verbose: bool):
        """
        :param verbose: if True, the main steps will be printed during the execution
        """
        self.verbose = verbose
        self.model = None

    def train(self, X, y: list):
        """
        Function used for training the Perceptron algorithm on the data vectorized using Tf-Idf vectorization
        :param X: sparse matrix with the texts preprocessed by Tf-Idf Vectorizer
        :param y: expected values for each example
        """
        if self.verbose:
            print(f'▶ Training the model with the following classes: {", ".join(np.unique(y))}')
        self.model = Perceptron(penalty='l1', alpha=0.001, random_state=0)
        self.model.fit(X, y)

    def predict(self, X):
        """
        Function for predicting the values given the sparse matrix of texts preprocessed using Tf-Idf vectorization
        :param X: sparse matrix with the texts preprocessed by Tf-Idf Vectorizer
        :return: list of predicted values
        """
        if not self.model:
            raise NotFittedError('Method train should be called first')
        if self.verbose:
            print(f'▶ Predicting...')
        pred = self.model.predict(X)
        return pred

    def compute_scores(self, expected: list, predicted: list, num_classes: int):
        """
        Function for computing the confusion matrix, Recall, Precision and F1 scores
        (will be stored in two files in the 'data' folder)
        :param expected: list with the expected values for the examples
        :param predicted: list with the predicted values for the examples
        :param num_classes: number of classes that the classifier could predict
        :return: numpy array with the confusion matrix, string with the recall, precision and f1 scores
        """
        if self.verbose:
            print('▶ Saving the scores')
        # get classes
        classes = self.model.classes_

        # get confusion matrix
        conf_matrix = confusion_matrix(expected, predicted)
        plt.figure(figsize=(10, 10))
        plot = sns.heatmap(conf_matrix, annot=True, cmap='Blues', xticklabels=classes, yticklabels=classes)
        plt.savefig(f'data/Confusion matrix {num_classes} classes.png')
        plt.clf()

        # get precision, recall, F1
        report = classification_report(predicted, expected)
        report_dict = classification_report(predicted, expected, output_dict=True)
        df = pd.DataFrame(report_dict).transpose().round(2)
        df.to_csv(f'data/Scores {num_classes} classes.csv')

        return conf_matrix, report, classes

    @staticmethod
    def accuracy_per_class(conf_matrix):
        """
        Supplementary function for computing the per class accuracy (Defined by the formula: ((TP+TN)/TP+TN+FP+FN))
        :param conf_matrix: numpy array with the confusion matrix
        :return: list of accuracies for each class in the matrix
        """
        output = []
        for idx in range(conf_matrix.shape[0]):
            true_negative = np.sum(np.delete(np.delete(conf_matrix, idx, axis=0), idx, axis=1))
            true_positive = conf_matrix[idx, idx]
            res = (true_negative + true_positive) / np.sum(conf_matrix)
            output.append(res)

        return output

    def visualise(self, conf_matrix2, conf_matrix6, cats2: list, cats6: list):
        """
        Function for visualizing the classification results (comparison of per class accuracies for the models)
        :param conf_matrix2: numpy array with the confusion matrix for 2 class classifier
        :param conf_matrix6: numpy array with the confusion matrix for 6 class classifier
        :param cats2: list of categories used in the 2 class classifier
        :param cats6: list of categories used in the 6 class classifier
        :return:
        """
        if self.verbose:
            print('▶ Saving the visualization')
        acc2 = self.accuracy_per_class(conf_matrix2)
        acc6 = self.accuracy_per_class(conf_matrix6)

        c = sns.set_palette(sns.color_palette(["#33FFC6"] * 10))
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        fig.suptitle('Accuracy per class')
        sns.barplot(ax=axes[0], x=cats2, y=acc2, color=c)
        axes[0].set_title('Accuracy per subcategory')
        sns.barplot(ax=axes[1], x=cats6, y=acc6, color=c)
        axes[1].set_title('Accuracy per category')
        fig.savefig('data/Accuracy visualization.png')
        fig.clf()

    def split_convert_data(self, texts: list, labels: list):
        """
        Supplementary function for splitting the data and processing it using Tf-Idf Vectorizer
        :param texts: list of texts to preprocess
        :param labels: list of expected values to assign to each example
        :return: sparse matrix with the train data, sparse matrix with the test data,
                list of expected values for the training data,  list of expected values for the test data
        """
        X_train, X_test, y_train, y_test = train_test_split(texts, labels, test_size=0.3, random_state=42,
                                                            stratify=labels)
        tfidf = TfidfVectorizer(tokenizer=word_tokenize)
        tfidf.fit(X_train)
        if self.verbose:
            print(f'▶ Number of Tf-Idf features: {len(tfidf.get_feature_names())}')
        X_train = tfidf.transform(X_train)
        X_test = tfidf.transform(X_test)
        return X_train, X_test, y_train, y_test


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
    parser.add_argument("--input", type=str, default='data/preprocessed_data.csv',
                        help="path to the preprocessed csv file")
    parser.add_argument('--verbose', help='print out the logs (default: False)', action='store_true')
    args = parser.parse_args()
    main(args.input, args.verbose)
