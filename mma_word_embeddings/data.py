# This file contains a class for data loading and analysing
# MMA data provided as json files
import datetime
from nltk.corpus import stopwords
import nltk
import string
from bs4 import BeautifulSoup
import re
import os
from gensim.models.phrases import Phrases, ENGLISH_CONNECTOR_WORDS
import json

nltk.download('stopwords')
nltk.download('wordnet')
stop = stopwords.words('english')

CUSTOM_STOPWORDS = []
STOPWORD_EXCEPTIONS = ['he', 'she', 'him', 'her', 'his', 'hers']
PUNCTUATION = string.punctuation.replace("_", "") + "“”’‘‚…–"  # add some symbols that have different ascii
GARBAGE = ['windowtextcolor', ]


def head(path_to_data, n=5):
    """Return list of first n dictionaries extracted from a json-lines (.jl) file.
    Args:
        path_to_data (str): full path to the .jl or .txt file, including ending
        n (int): number of rows to read
    """
    data_loader = open(path_to_data)

    entries = []
    for i, row in enumerate(data_loader):
        entries.append(row)
        if i == n-1:
            break

    data_loader.close()
    return entries


def clean(path_to_json,
          text_column,
          output_path,
          filter={},
          extract_sentences=True,
          remove_stopwords=False,
          lemmatize=False,
          agent_column=None,
          leave_hashtag_at_symbol=True,
          ):
    """Save a preprocessed representation of the data to a new file.

    Args:
        path_to_json (str): full path to json-lines (.jl or .jsonl) file that contains the original data
        text_column (str): name of the column that contains the documents
        output_path (str): path to save training data and description to; does not contain a file ending
        filter (dict[str, list[str]]): Dictionary of column names as keys, and a list of strings
            to search for in the column as values. Only cells where at least one
            of the strings is found will be processed.
        extract_sentences (bool): if true, save one sentence per line into the new file; else save one document per line
        remove_stopwords (bool): if true, remove standard stop words from training data
        leave_hashtag_at_symbol (bool): if true, leave # and @
        lemmatize (bool): if true, replace words by their stems
        agent_column (None or str): if string, represents the name of the agent column;
            append text in text_column with agent tag before cleaning the data
    """
    # check if the output file already exists, to avoid overwriting
    output_train = output_path + "_training-data.txt"
    if os.path.exists(output_train):
        raise ValueError(f"File {output_train} already exists.")
    output_description = output_path + "_training-data" + "_description.txt"
    if os.path.exists(output_description):
        raise ValueError(f"File {output_description} already exists.")

    print("Start cleaning documents...")
    # load a json reader that can read files line-by-line
    data_loader = open(path_to_json, "r")

    if leave_hashtag_at_symbol:
        PUNCT = PUNCTUATION.replace("#", "").replace("@", "")

    for idx, row in enumerate(data_loader):

        if idx % 10000 == 0:
            print("...went through first ", idx, " documents...")

        # turn string into dictionary of the
        # form {'column_1': content1, 'column_2': content2,...}
        try:
            row_dict = json.loads(row)
        except json.decoder.JSONDecodeError:
            print(f"Decoding problem in row {idx} with content <{row}>. Breaking here.")
            break

        # apply filter
        ignore_row = False
        for column_name, string_list in filter.items():
            at_least_one_string_matches = any(s in row_dict[column_name] for s in string_list)
            if not at_least_one_string_matches:
                ignore_row = True

        if ignore_row:
            # go to next iteration
            continue

        # retrieve document in row
        document = row_dict[text_column]

        if extract_sentences:
            subdocuments = document.split(". ")
        else:
            subdocuments = [document]

        for chunk in subdocuments:

            if agent_column is not None:
                # prepend with agent tag
                if not isinstance(agent_column, list):
                    agent_column = [agent_column]

                name = "agent"
                for col in agent_column:
                    name += "_" + str(row_dict[col]).strip().replace(" ", "_")
                chunk = name + " " + chunk

            chunk = re.sub(r'http\S+', '', chunk)

            chunk = re.sub(r'\b[a-z]+(?:[A-Z][a-z]+)+\b', '', chunk)

            chunk = BeautifulSoup(chunk, "html.parser").text

            chunk = chunk.replace(r'\xad', '')

            chunk = chunk.replace('displayad', '')

            chunk = ''.join(char for word in chunk for char in word
                               if char not in PUNCT)

            # split string into list of words separated by whitespace
            chunk = chunk.split()

            chunk = [word for word in chunk if all(g not in word for g in GARBAGE)]

            if remove_stopwords:
                chunk = [word for word in chunk if word not in stop or word in STOPWORD_EXCEPTIONS]

            chunk = [word.lower() for word in chunk]

            if lemmatize:
                lm = nltk.WordNetLemmatizer()
                chunk = [lm.lemmatize(word) for word in chunk]

            chunk = " ".join(chunk)
            # save cleaned chunk in a row
            if chunk:
                with open(output_train, 'a+') as f:
                    f.write('%s\n' % chunk)

    data_loader.close()

    # document what has been done during pre-processing and save as file
    now = datetime.datetime.now()
    description = "{} - Data was loaded from file {}. \n".format(now, path_to_json)
    description += "Data preprocessing included the following steps: \n"
    if filter != {}:
        description += "...filtered data, keeping only rows where the specified columns contain " \
                       "(at least one of) the following expression(s) {}. \n".format(filter)
    if extract_sentences:
        description += "...split documents into sentences...\n"
    if agent_column is not None:
        description += "...prepend with agent tag...\n"
    description += r"...remove all words that have single upper case letters surrounded by lower case " \
                   r"letters (to get rid of javascript) " + "\n"
    description += r"...remove html formatting with BeautifulSoup (html.parser) " + "\n"
    description += r"...remove expression '\xad' " + "\n"
    description += r"...remove expression 'displayad'" + "\n"
    description += r"...remove punctuation (but keep digits "
    if leave_hashtag_at_symbol:
        description += r"and hashtag and at symbol)" + "\n"
    else:
        description += r")" + "\n"

    description += r"...remove words that contain substrings {}, ".format(GARBAGE) + "\n"
    if remove_stopwords:
        description += r"...remove words from nltk's list of english stopwords (making exceptions for {}),".format(
            STOPWORD_EXCEPTIONS) + "\n"
    description += r"...make all words lower case, " + "\n"
    if lemmatize:
        description += r"...lemmatize words with nltk's WordNetLemmatizer, " + "\n"
    description += r"... altogether, {} lines were processed".format(idx)

    with open(output_description, 'a') as f:
        f.write('%s' % description)


def clean_txt(path_to_txt,
          output_path,
          extract_sentences=False,
          remove_stopwords=False,
          lemmatize=False):
    """Save a preprocessed representation of the data to a new file.

    Args:
        path_to_json (str): full path to json-lines (.jl or .jsonl) file that contains the original data
        text_column (str): name of the column that contains the documents
        output_path (str): path to save training data and description to; does not contain a file ending
        extract_sentences (bool): if true, save one sentence per line into the new file; else save one document per line
        remove_stopwords (bool): if true, remove standard stop words from training data
        lemmatize (bool): if true, replace words by their stems
        agent_column (None or str): if string, represents the name of the agent column;
            append text in text_column with agent tag before cleaning the data
    """
    # check if the output file already exists, to avoid overwriting
    output_train = output_path + "_training-data.txt"
    if os.path.exists(output_train):
        raise ValueError(f"File {output_train} already exists.")
    output_description = output_path + "_training-data" + "_description.txt"
    if os.path.exists(output_description):
        raise ValueError(f"File {output_description} already exists.")

    print("Start cleaning documents...")
    # load a json reader that can read files line-by-line
    data_loader = open(path_to_txt)

    for idx, row in enumerate(data_loader):

        if not row:
            continue

        if idx % 10000 == 0:
            print("...went through first ", idx, " documents...")

        # retrieve document in row
        document = row

        if extract_sentences:
            subdocuments = document.split(". ")
        else:
            subdocuments = [document]

        for chunk in subdocuments:

            chunk = re.sub(r'http\S+', '', chunk)

            chunk = re.sub(r'\b[a-z]+(?:[A-Z][a-z]+)+\b', '', chunk)

            chunk = BeautifulSoup(chunk, "html.parser").text

            chunk = chunk.replace(r'\xad', '')

            chunk = chunk.replace('displayad', '')

            chunk = ''.join(char for word in chunk for char in word
                               if char not in PUNCTUATION)

            # split string into list of words separated by whitespace
            chunk = chunk.split()

            chunk = [word for word in chunk if all(g not in word for g in GARBAGE)]

            if remove_stopwords:
                chunk = [word for word in chunk if word not in stop or word in STOPWORD_EXCEPTIONS]

            chunk = [word.lower() for word in chunk]

            if lemmatize:
                lm = nltk.WordNetLemmatizer()
                chunk = [lm.lemmatize(word) for word in chunk]

            chunk = " ".join(chunk)
            # save cleaned chunk in a row
            if chunk:
                with open(output_train, 'a+') as f:
                    f.write('%s\n' % chunk)

    data_loader.close()

    # document what has been done during pre-processing and save as file
    now = datetime.datetime.now()
    description = "{} - Data was loaded from file {}. \n".format(now, path_to_txt)
    description += "Data preprocessing included the following steps: \n"
    if extract_sentences:
        description += "...split documents into sentences...\n"
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

    with open(output_description, 'a') as f:
        f.write('%s' % description)


def make_ngrams(path_to_txt, description_path=None, min_count_ngrams=50, threshold_ngrams=10):
    """Replace frequently occuring word combinations with bigrams and trigrams in document
    of sentences.

    Args:
        path_to_txt (str): path to text (.txt) file of one sentence per line
        description_path (str): path to text file containing description
        min_count_ngrams (int): Ignore all words and ngrams with total collected count lower than this value
        threshold_ngrams (int): Represent a score threshold for forming the phrases (higher means fewer phrases).
    """

    temp_path = path_to_txt[:-4] + "-temp.txt"

    def sentence_generator(path):
        """Read sentences from disk one-by-one"""
        with open(path, 'r') as f:
            for line in f:
                yield line.strip().split()

    print("Making bigrams...")

    gram_model = Phrases(sentence_generator(path_to_txt),
                         min_count=min_count_ngrams,
                         threshold=threshold_ngrams,
                         max_vocab_size=2000000,
                         connector_words=ENGLISH_CONNECTOR_WORDS)

    # slim down model (no new sentences can be added)
    gram_model.freeze()

    # write bigram sentences into temporary file
    # (couldn't figure out how to replace)
    with open(temp_path, 'w') as f:
        for sentence in sentence_generator(path_to_txt):
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
    with open(path_to_txt, 'w') as f:
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


def extract(path_to_json,
            output_path,
            filter):
    """Save a preprocessed representation of the data to a new file.

    Args:
        path_to_json (str): full path to json-lines (.jl or .jsonl) file that contains the original data
        output_path (str): full path to save training data and description to; does not contain a file ending
        filter (dict[str, list[str]]): Dictionary of column names as keys, and a list of strings
            to search for in the column as values. Only cells where at least one
            of the strings is found will be processed.
    """
    # check if the output file already exists, to avoid overwriting
    if os.path.exists(output_path):
        raise ValueError(f"File {output_path} already exists.")

    print("Start filtering documents...")
    # load a json reader that can read files line-by-line
    data_loader = open(path_to_json, 'r')

    for idx, row in enumerate(data_loader):

        if "items read" in row:
            print(row)
            break

        # turn string into dictionary of the
        # form {'column_1': content1, 'column_2': content2,...}
        row_dict = json.loads(row)

        # apply filter
        ignore_row = False
        for column_name, string_list in filter.items():
            at_least_one_string_matches = any(s in row_dict[column_name] for s in string_list)
            if not at_least_one_string_matches:
                ignore_row = True

        if idx % 100000 == 0 and idx != 0:
            print("...filtered first ", idx, " documents...")

        if ignore_row:
            # go to next iteration
            continue

        with open(output_path, 'a+') as f:
            json.dump(row_dict, f)
            f.write("\n")

    data_loader.close()

    description = f"Filtered data set {path_to_json} with filter: \n {filter}"
    with open(output_path[:-4] + '-description.txt', 'w') as f:
        f.write('%s' % description)


def count_first_word(path_to_txt, path_out):
    """
    Saves the first words in each line of path_to_txt, together with
    their frequency. Useful to extract a list of agents and the frequency
    of their utterances, when "agent_column" was used in clean().

    path_to_txt (str): path to data, one sentence/document per line
    path_out (str): full path to save output
    """

    if os.path.exists(path_out):
        raise ValueError(f"path {path_out} exists already.")
    first_words = {}
    with open(path_to_txt, 'r') as f:
        for line in f:

            word = line.split()[0]

            if word in first_words:
                first_words[word] += 1
            else:
                first_words[word] = 1

    # save as tuple list
    first_words = [(k, v) for k, v in first_words.items()]
    # sort
    first_words = sorted(first_words, key=lambda x: x[1], reverse=True)

    with open(path_out, 'w') as f:
        for pair in first_words:
            f.write(pair[0] + " " + str(pair[1]) + "\n")
