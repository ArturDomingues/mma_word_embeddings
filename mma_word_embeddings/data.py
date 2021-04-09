# This file contains a class for data loading and analysing
# MMA data provided as json files
import pandas as pd
import json
from nltk.corpus import stopwords
import matplotlib.pyplot as plt
import nltk
import gensim
import string
from bs4 import BeautifulSoup
import re
import sys
import os
from gensim.models.phrases import Phrases, ENGLISH_CONNECTOR_WORDS

nltk.download('stopwords')
nltk.download('wordnet')
stop = stopwords.words('english')

CUSTOM_STOPWORDS = []
STOPWORD_EXCEPTIONS = ['he', 'she', 'him', 'her', 'his', 'hers']
PUNCTUATION = string.punctuation.replace("_", "") + "“”’‘‚…–"  # add some symbols that have different ascii
GARBAGE = ['windowtextcolor', ]


class DexterData:

    def __init__(self, path_to_data):
        """Representation of the data.
        Args:
            path_to_data (str): path to .json data file
        """

        print("Loading data...")

        with open(path_to_data, 'r', encoding='utf8') as f:
            data = json.load(f)
        print("...done.")

        self.data = pd.DataFrame(data)
        self.description = "Data was loaded from file {}. \n".format(path_to_data)

    def head(self):
        """print head of data frame."""
        return self.data.head()

    def first_entry(self, column):
        """return first entry of the column."""
        return self.data[column].iloc[0]

    def column_names(self):
        """Return list of column names."""

        return list(self.data)

    def column(self, column):
        """Return 'column' as a list."""

        return self.data[column].to_list()

    def unique_values(self, column):
        """Return list of unique values represented in the column."""

        return self.data[column].value_counts()

    def filter(self, column, selectors):
        """Only keep rows with any of 'selectors' as value for 'column'."""

        mask = self.data[column].str.contains('|'.join(selectors))
        self.data = self.data[mask]
        self.description += "The data was filtered, keeping only rows where column <{}> contains (at least one of) the " \
                            "expression(s) {}. \n".format(column, selectors)

    def plot_historgram(self, column, selectors, x_axis, bins=20):
        fig, ax = plt.subplots()
        for selector in selectors:
            df = self.data[self.data[column] == selector][x_axis]
            plt.hist(df, bins=bins, alpha=0.5, label=selector)

        num_x_ticks = len(ax.xaxis.get_ticklabels())
        every_nth = num_x_ticks / 10
        for n, label in enumerate(ax.xaxis.get_ticklabels()):
            if n % every_nth != 0 and n != num_x_ticks:
                label.set_visible(False)

        plt.xticks(rotation='vertical')
        plt.legend(loc='upper right')
        plt.show()

    def extract_sentences(self, text_column, output_path, remove_stopwords=False, lemmatize=False):
        """Get a representation of the data that can be used to train a word2vec model.

        Args:
            text_column (str): name of the column that contains the text
            output_path (str): path to save training data to
            min_count_ngrams (int): Ignore all words and ngrams with total collected count lower than this value
            threshold_ngrams (int):  Represent a score threshold for forming the phrases (higher means fewer phrases).
                A phrase of words a followed by b is accepted if the score of the phrase is greater than threshold.
                Heavily depends on concrete scoring-function, see the scoring parameter.
            remove_stopwords (bool): If true, remove standard stop words from training data.
            lemmatize (bool): If true, replace words by their stems
            make_ngrams (bool): If true, make bi and trigrams
        """

        if os.path.exists(output_path + '-training-data.txt'):
            raise ValueError(f"File {output_path + '-training-data.txt'} exists already.")

        print("Start cleaning documents...")

        for i in range(self.data.shape[0]):

            # retrieve i'th document
            document = self.data.iloc[i, self.data.columns.get_loc(text_column)]
            # split into sentences
            sentences = document.split(".")

            for idx, sentence in enumerate(sentences):

                sentence = re.sub(r'\b[a-z]+(?:[A-Z][a-z]+)+\b', '', sentence)

                sentence = BeautifulSoup(sentence, "html.parser").text

                sentence = sentence.replace(r'\xad', '')

                sentence = sentence.replace('displayad', '')

                sentence = ''.join(char for word in sentence for char in word
                                   if char not in PUNCTUATION)

                # split string into list of words separated by whitespace
                sentence = sentence.split()

                sentence = [word for word in sentence if all(g not in word for g in GARBAGE)]

                if remove_stopwords:
                    sentence = [word for word in sentence if word not in stop or word in STOPWORD_EXCEPTIONS]

                sentence = [word.lower() for word in sentence]

                if lemmatize:
                    lm = nltk.WordNetLemmatizer()
                    sentence = [lm.lemmatize(word) for word in sentence]

                sentence = " ".join(sentence)
                # save cleaned sentence in a row
                if sentence:
                    with open(output_path + '-training-data.txt', 'a+') as f:
                        f.write('%s\n' % sentence)

            if i % 10000 == 0:
                print("...cleaned first ", i, " documents...")

        # document what has been done during pre-processing and save as file
        self.description += "Data preprocessing included the following steps: \n"
        self.description += "...split documents into sentences\n"
        self.description += r"...remove all words that have single upper case letters surrounded by lower case " \
                            r"letters (to get rid of javascript) " + "\n"
        self.description += r"...remove html formatting with BeautifulSoup (html.parser) " + "\n"
        self.description += r"...remove expression '\xad' " + "\n"
        self.description += r"...remove expression 'displayad'" + "\n"
        self.description += r"...remove punctuation (but keep digits)" + "\n"
        self.description += r"...remove words that contain substrings {}, ".format(GARBAGE) + "\n"
        if remove_stopwords:
            self.description += r"...remove words from nltk's list of english stopwords (making exceptions for {}),".format(
                STOPWORD_EXCEPTIONS) + "\n"
        self.description += r"...make all words lower case, " + "\n"
        if lemmatize:
            self.description += r"...lemmatize words with nltk's WordNetLemmatizer, " + "\n"

        with open(output_path + '-description.txt', 'a') as f:
            f.write('%s' % self.description)


def make_ngrams(input_path, description_path=None, min_count_ngrams=50, threshold_ngrams=10):
    """Replace frequently occuring word combinations with bigrams and trigrams in document
    of sentences."""

    temp_path = input_path[:-4] + "-temp.txt"

    def sentence_generator(path):
        """Read sentences from disk one-by-one"""
        with open(path, 'r') as f:
            for line in f:
                yield line.split()

    print("Making bigrams...")

    gram_model = Phrases(sentence_generator(input_path),
                         min_count=min_count_ngrams,
                         threshold=threshold_ngrams,
                         max_vocab_size=2000000,
                         connector_words=ENGLISH_CONNECTOR_WORDS)

    # slim down model (no new sentences can be added)
    gram_model.freeze()

    # write bigram sentences into temporary file
    # (couldn't figure out how to replace)
    with open(temp_path, 'w') as f:
        for sentence in sentence_generator(input_path):
            new_sentence = gram_model[sentence]
            new_sentence = " ".join(new_sentence) + "\n"
            f.write(new_sentence)

    print("...and trigrams...")
    # repeat procedure to get trigrams
    gram_model = Phrases(sentence_generator(temp_path),
                         min_count=min_count_ngrams,
                         threshold=threshold_ngrams,
                         max_vocab_size=2000000,
                         connector_words=ENGLISH_CONNECTOR_WORDS)

    gram_model.freeze()

    # overwrite input path
    with open(input_path, 'w') as f:
        for sentence in sentence_generator(temp_path):
            new_sentence = gram_model[sentence]
            new_sentence = " ".join(new_sentence) + "\n"
            f.write(new_sentence)

    # delete file at temp path
    os.remove(temp_path)
    print("...done.")

    description = "...turn common word sequences into bigrams or trigrams using gensim " \
                  "(min_count {} and threshold {})".format(min_count_ngrams, threshold_ngrams)

    if description_path is not None:
        with open(description_path, 'a') as f:
            f.write('%s' % description)
