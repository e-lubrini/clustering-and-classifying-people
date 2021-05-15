import argparse
import nltk
import os
import pandas as pd
import string
from nltk.corpus import stopwords
from nltk.corpus import wordnet as wn
from nltk.stem.wordnet import WordNetLemmatizer


class Preprocessor:
    """
    Class for preprocessing the data before using it for training the models
    """
    def __init__(self, lowercase=True, rmv_stopwords=True, rmv_punct=True, lemmatize=True, rmv_nums=False,
                 rmv_foreign=True, verbose=False):
        """
        :param lowercase: if True, all texts will be lower cased
        :param rmv_stopwords: if True, all stop words will be deleted
        :param rmv_punct: if True, all punctuation will be deleted
        :param lemmatize: if True, all words will be lemmatized (using nltk lemmatizer)
        :param rmv_nums: if True, all numbers will be deleted
        :param rmv_foreign: if True, all non-english letters will be deleted
        :param verbose: if True, the main steps will be printed during the execution
        """
        self.lowercase = lowercase
        self.rmv_stopwords = rmv_stopwords
        self.rmv_punct = rmv_punct
        self.lemmatize = lemmatize
        self.rmv_nums = rmv_nums
        self.rmv_foreign = rmv_foreign
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
        self.verbose = verbose

    def transform(self, texts:list):
        """
        Fucntion that applies all the chosen transformations to the texts
        :param texts: list of texts that should be preprocessed
        :return: the list with all the texts
        """
        if self.lowercase:
            if self.verbose:
                print('▶ Lower casing...')
            texts = list(map(self.lower_case, texts))
        if self.rmv_nums:
            if self.verbose:
                pass
            print('▶ Numbers removal...')
            texts = list(map(self.remove_nums, texts))
        if self.rmv_punct:
            if self.verbose:
                print('▶ Punctuation removal...')
            texts = list(map(self.remove_punct, texts))

        if self.rmv_foreign:
            if self.verbose:
                print('▶ Foreign characters removal...')
            texts = list(map(self.remove_foreign, texts))

        if self.verbose:
            print('▶ Tokenization...')
        texts = list(map(self.tokenize, texts))

        if self.rmv_stopwords:
            if self.verbose:
                print('▶ Stopwords removal...')
            texts = list(map(self.remove_stopwords, texts))

        if self.lemmatize:
            if self.verbose:
                print('▶ Lemmatization...')
            texts = list(map(self.lemmatize_text, texts))
        texts = [" ".join(text) for text in texts]
        return texts

    @staticmethod
    def tokenize(text):
        """
        Supplementary method for tokenization
        :param text: a string with a text
        :return: list of all tokens in the text
        """
        tokenised_s = nltk.word_tokenize(text)
        return tokenised_s

    @staticmethod
    def lower_case(text):
        """
        Supplementary method for changing the case
        :param text: a string with a text
        :return: the transformed string
        """
        return text.lower()

    @staticmethod
    def remove_nums(text):
        """
        Supplementary method for removing numbers
        :param text: a string with a text
        :return: the transformed string
        """
        nums_translator = str.maketrans('', '', '0123456789')
        return text.translate(nums_translator)

    @staticmethod
    def remove_punct(text):
        """
        Supplementary method for removing the punctuation
        :param text: a string with a text
        :return: the transformed string
        """
        punct_translator = str.maketrans('', '', string.punctuation)
        return text.translate(punct_translator)

    @staticmethod
    def segment_and_tokenize(text):
        """
        Supplementary method for tokenizing a text into lists of lists of tokens
        :param text: a string with a text
        :return: the list of lists of tokens
        """
        # Sentence splitting
        sentences = nltk.sent_tokenize(text)
        # tokenizing
        tokenised_s = list(map(nltk.word_tokenize, sentences))
        return tokenised_s

    def remove_stopwords(self, text):
        """
        Supplementary method for removing the stopwords
        :param text: list of tokens of the text
        :return: the transformed list
        """
        no_stopwords = list(filter(lambda x: x not in self.stop_words, text))
        return no_stopwords

    def lemmatize_text(self, text):
        """
        Supplementary method for lemmatizing the text
        :param text: list of tokens of the text
        :return: list of lemmatized tokens
        """
        text_pos = nltk.tag.pos_tag(text)
        lemmas = []
        for word, pos in text_pos:
            if pos.startswith('VB'):
                lemma = self.lemmatizer.lemmatize(word, wn.VERB)
                lemmas.append(lemma)
            elif pos.startswith('RB'):
                lemma = self.lemmatizer.lemmatize(word, wn.ADV)
                lemmas.append(lemma)
            elif pos.startswith('JJ'):
                lemma = self.lemmatizer.lemmatize(word, wn.ADJ)
                lemmas.append(lemma)
            elif pos.startswith('NN'):
                lemma = self.lemmatizer.lemmatize(word, wn.NOUN)
                lemmas.append(lemma)
            else:
                lemmas.append(word)
        return lemmas

    @staticmethod
    def remove_foreign(text):
        """
        Supplementary method for removing all the letters except for the English ones
        :param text: a string with a text
        :return: the transformed string
        """
        def checker(x):
            return ((x.isalpha() and x in list('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'))
                    or not x.isalpha())

        text = "".join(filter(checker, text))
        return text


def main(inputpath, outputpath, verbose):
    """
    Function that starts after calling the script
    :param inputpath: path to the csv with the data for preprocessing
    :param outputpath: desired path to the output csv file
    :param verbose:  if True, the main steps will be printed during the execution
    """
    if os.path.isfile(inputpath):
        df = pd.read_csv(inputpath)
    else:
        raise FileNotFoundError(f'File {inputpath} is not found. Retry with another name')
    preprocessor = Preprocessor(verbose=args.verbose)
    if verbose:
        print('▶ Descriptions processing...')
    df['token_description'] = preprocessor.transform(df['description'])
    if verbose:
        print('▶ Texts processing...')
    df['token_text'] = preprocessor.transform(df['text'])
    df = df[['title', 'text', 'token_text', 'description', 'token_description', 'category', 'datatype']]
    df.to_csv(outputpath, index=False)


# main entry point
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Corpus Preprocessor")
    parser.add_argument("--input", type=str, default='data/data.csv',
                        help="path to the input file for preprocessing (a csv file obtained after running the \
                        corpus_extraction.py script)")
    parser.add_argument("--output", type=str, default='data/preprocessed_data.csv',
                        help="desired path to the output csv file")
    parser.add_argument('--verbose', help='print out the logs (default: False)', action='store_true')
    args = parser.parse_args()
    main(args.input, args.output, args.verbose)
