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
from itertools import groupby
import zipfile


nltk.download('stopwords')
nltk.download('wordnet')
stop = stopwords.words('english')

CUSTOM_STOPWORDS = []
STOPWORD_EXCEPTIONS = ['he', 'she', 'him', 'her', 'his', 'hers']
PUNCTUATION = string.punctuation.replace("_", "") + "“”’‘‚…"  # add some symbols that have different ascii
GARBAGE = ['windowtextcolor', ]


class DexterData:

    def __init__(self, path_to_data):
        """Representation of the data.
        Args:
            path_to_data (str): path to .json data file
        """

        print("Loading data...")

        if path_to_data[-3:] == ".zip":
            with zipfile.ZipFile(path_to_data, 'r') as zip_ref:
                zip_ref.extractall(directory_to_extract_to)

        with open(path_to_data, 'r', encoding='utf8') as f:
            data = json.load(f)
        print("...done.")

        self.data = pd.DataFrame(data)
        self.data_path = path_to_data
        self.training_data = None
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
        every_nth = num_x_ticks/10
        for n, label in enumerate(ax.xaxis.get_ticklabels()):
            if n % every_nth != 0 and n != num_x_ticks:
                label.set_visible(False)

        plt.xticks(rotation='vertical')
        plt.legend(loc='upper right')
        plt.show()

    def get_training_data(self, text_column, min_count_ngrams=50, threshold_ngrams=10,
                          remove_stopwords=False, lemmatize=False, make_ngrams=True):
        """Get a representation of the data that can be used to train a word2vec model.

        Args:
            text_column (str): name of the column that contains the text
            min_count_ngrams (int): Ignore all words and ngrams with total collected count lower than this value
            threshold_ngrams (int):  Represent a score threshold for forming the phrases (higher means fewer phrases).
                A phrase of words a followed by b is accepted if the score of the phrase is greater than threshold.
                Heavily depends on concrete scoring-function, see the scoring parameter.
            remove_stopwords (bool): If true, remove standard stop words from training data.
            lemmatize (bool): If true, replace words by their stems
            make_ngrams (bool): If true, make bi and trigrams
        """

        # document pre-processing in description
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


        print("Clean documents...")

        # create list of word lists per document (i.e., newspaper article, utterance)
        cleaned_data = []
        corpus = self.data[text_column].to_list()

        corpus = [document.split(".") for document in corpus]
        sentences = [sentence for document in corpus for sentence in document]

        for idx, sentence in enumerate(sentences):

                if idx % 10000 == 0:
                    print("...cleaned first ", idx, " sentences...")

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

                cleaned_data.append(sentence)

        # SECOND STEP: make N-Grams #########################
        print("...make bigrams and trigrams...")

        if make_ngrams:
            # save description
            self.description += "...turn common word sequences into bigrams or trigrams using gensim " \
                                "(min_count {} and threshold {})".format(min_count_ngrams, threshold_ngrams)

            bigram = gensim.models.Phrases(cleaned_data, min_count=min_count_ngrams, threshold=threshold_ngrams)
            trigram = gensim.models.Phrases(bigram[cleaned_data], min_count=min_count_ngrams, threshold=threshold_ngrams)

            bigram_mod = gensim.models.phrases.Phraser(bigram)
            trigram_mod = gensim.models.phrases.Phraser(trigram)

            cleaned_data = [trigram_mod[bigram_mod[document]] for document in cleaned_data]

        print("...done.")

        ##########################

        self.training_data = [sentence for sentence in cleaned_data if sentence != []]

        return cleaned_data

    def save_training_data(self, output_path):
        """Save the training data in "one-sentence-per-line" format."""

        if self.training_data is None:
            raise ValueError("You need to run the get_training_data() method before saving the data.")

        # Save training data
        with open(output_path + '-training-data.txt', 'w') as f:
            for document in self.training_data:
                sentence = " ".join(word for word in document)
                f.write('%s\n' % sentence)

        # Save description
        with open(output_path + '-description.txt', 'w') as f:
            f.write('%s' % self.description)

    @classmethod
    def word_frequencies(self, list_of_token_lists):
        """Get a list of word frequencies in list_of_token_lists, sorted from the most frequent to the least."""

        flat_training_data = [word for document in list_of_token_lists for word in document]
        frequencies = [[value, len(list(freq))] for value, freq in groupby(sorted(flat_training_data))]
        frequencies = sorted(frequencies, key=lambda x: x[1], reverse=True)

        return pd.DataFrame({'word': [f[0] for f in frequencies], 'frequency': [f[1] for f in frequencies]})
