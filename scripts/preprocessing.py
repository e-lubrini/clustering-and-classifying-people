import argparse
import os
import string

import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.corpus import wordnet as wn
from nltk.stem.wordnet import WordNetLemmatizer


class Preprocessor:
    def __init__(self, lowercase=True, rmv_stopwords=True, rmv_punct=True, lemmatize=True, rmv_nums=False,
                 verbose=False):
        self.lowercase = lowercase
        self.rmv_stopwords = rmv_stopwords
        self.rmv_punct = rmv_punct
        self.lemmatize = lemmatize
        self.rmv_nums = rmv_nums
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
        self.verbose = verbose

    def transform(self, texts):
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
        tokenised_s = nltk.word_tokenize(text)
        return tokenised_s

    @staticmethod
    def lower_case(text):
        return text.lower()

    @staticmethod
    def remove_nums(text):
        nums_translator = str.maketrans('', '', '0123456789')
        return text.translate(nums_translator)

    @staticmethod
    def remove_punct(text):
        punct_translator = str.maketrans('', '', string.punctuation)
        return text.translate(punct_translator)

    @staticmethod
    def segment_and_tokenize(text):
        # Sentence splitting
        sentences = nltk.sent_tokenize(text)
        # tokenizing
        tokenised_s = list(map(nltk.word_tokenize, sentences))
        return tokenised_s

    def remove_stopwords(self, text):
        no_stopwords = list(filter(lambda x: x not in self.stop_words, text))
        return no_stopwords

    def lemmatize_text(self, text):
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


def main(inputpath, outputpath, verbose):
    if os.path.isfile(inputpath):
        df = pd.read_csv(inputpath)
    else:
        raise FileNotFoundError(f'File {inputpath} is not found. Retry with another name')
    preprocessor = Preprocessor(verbose=args.verbose)
    if verbose:
        print('Descriptions processing...')
    df['token_description'] = preprocessor.transform(df['description'])
    if verbose:
        print('Texts processing...')
    df['token_text'] = preprocessor.transform(df['text'])
    df = df[['title', 'text', 'token_text', 'description', 'token_description', 'category', 'datatype']]
    df.to_csv(outputpath, index=False)


# main entry point
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Corpus Preprocessor")
    parser.add_argument("--input", type=str, default='data.csv',
                        help="path to the input file for preprocessing (a csv file obtained after running the corpus_extraction.py script)")
    parser.add_argument("--output", type=str, default='preprocessed_data.csv',
                        help="desired path to the output csv file")
    parser.add_argument('--verbose', help='print out the logs (default: False)', action='store_true')
    args = parser.parse_args()
    main(args.input, args.output, args.verbose)
