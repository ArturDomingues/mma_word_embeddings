# This file contains a class for data loading and analysing
# MMA data provided as json files
import pandas as pd
from nltk.corpus import stopwords
import nltk
import string
from bs4 import BeautifulSoup
import re
import os
from gensim.models.phrases import Phrases, ENGLISH_CONNECTOR_WORDS
from io import StringIO

nltk.download('stopwords')
nltk.download('wordnet')
stop = stopwords.words('english')

CUSTOM_STOPWORDS = []
STOPWORD_EXCEPTIONS = ['he', 'she', 'him', 'her', 'his', 'hers']
PUNCTUATION = string.punctuation.replace("_", "") + "“”’‘‚…–"  # add some symbols that have different ascii
GARBAGE = ['windowtextcolor', ]


def extract_clean_sentences(path_to_json,
                            text_column,
                            output_path,
                            filter=None,
                            remove_stopwords=False,
                            lemmatize=False):
    """Save a preprocessed representation of the data to a new file.

    Args:
        path_to_json (str): path to json-lines (.jl or .jsonl) file that contains the original data
        text_column (str): name of the column that contains the documents
        output_path (str): path to save training data and description to; does not contain a file ending
        filter (dict[str, list[str]]): Dictionary of column names as keys, and a list of strings
            to search for in the column as values. Only cells where at least one
            of the strings is found will be processed.
        remove_stopwords (bool): if true, remove standard stop words from training data
        lemmatize (bool): if true, replace words by their stems
    """
    # check if the output file already exists, to avoid overwriting
    if os.path.exists(output_path + '-training-data.txt'):
        raise ValueError(f"File {output_path + '-training-data.txt'} exists already.")

    # load a json reader that can read files line-by-line
    try:
        reader = pd.read_json(StringIO(path_to_json), lines=True, chunksize=1)
    except FileNotFoundError:
        raise ValueError("Cannot read the file. Is the path correct?")

    print("Start cleaning documents...")

    for idx, row in enumerate(reader):

        # TODO apply filter
        if True:
            continue

        # retrieve document in row
        document = row.iloc[0, row.columns.get_loc(text_column)]

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

        if idx % 10000 == 0:
            print("...cleaned first ", idx, " documents...")

    # document what has been done during pre-processing and save as file
    description = "Data was loaded from file {}. \n".format(path_to_json)
    description += "Data preprocessing included the following steps: \n"
    description += "The data was filtered, keeping only rows where the specified columns contain " \
                   "(at least one of) the following expression(s) {}. \n".format(filter)
    description += "Split documents into sentences\n"
    description += r"...remove all words that have single upper case letters surrounded by lower case " \
                   r"letters (to get rid of javascript) " + "\n"
    description += r"...remove html formatting with BeautifulSoup (html.parser) " + "\n"
    description += r"...remove expression '\xad' " + "\n"
    description += r"...remove expression 'displayad'" + "\n"
    description += r"...remove punctuation (but keep digits)" + "\n"
    description += r"...remove words that contain substrings {}, ".format(GARBAGE) + "\n"
    if remove_stopwords:
        description += r"...remove words from nltk's list of english stopwords (making exceptions for {}),".format(
            STOPWORD_EXCEPTIONS) + "\n"
    description += r"...make all words lower case, " + "\n"
    if lemmatize:
        description += r"...lemmatize words with nltk's WordNetLemmatizer, " + "\n"

    with open(output_path + '-description.txt', 'a') as f:
        f.write('%s' % description)


def make_ngrams(input_path, description_path=None, min_count_ngrams=50, threshold_ngrams=10):
    """Replace frequently occuring word combinations with bigrams and trigrams in document
    of sentences.

    Args:
        input_path (str): path to text (.txt) file of one sentence per line
        description_path (str): path to text file containing description
        min_count_ngrams (int): Ignore all words and ngrams with total collected count lower than this value
        threshold_ngrams (int): Represent a score threshold for forming the phrases (higher means fewer phrases).
    """

    temp_path = input_path[:-4] + "-temp.txt"

    def sentence_generator(path):
        """Read sentences from disk one-by-one"""
        with open(path, 'r') as f:
            for line in f:
                yield line.strip().split()

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
