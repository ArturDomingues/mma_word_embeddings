# This file contains a wrapper class that represents word trained_embeddings
from gensim.models import KeyedVectors
from mma_word_embeddings.utils import normalize, make_pairs
import numpy as np
from itertools import combinations_with_replacement
import pandas as pd
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from sklearn.manifold import TSNE
import glob
from itertools import groupby

# Make pandas print full data frame
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', -1)
pd.options.display.float_format = '{:,.4f}'.format

COLORMAP = mcolors.LinearSegmentedColormap.from_list("MyCmapName",["r", "w", "g"])


class EmbeddingError(Exception):
    """Exception raised by a Model object when something is wrong.
    """


class WordEmbedding:
    """Representation of a word embedding, which is a map from word strings to vectors."""

    def __init__(self, path_to_embedding, path_training_data=None):

        print("Loading embedding {} ... ".format(path_to_embedding))

        try:
            # load the word vectors of an embedding
            self._word_vectors = KeyedVectors.load(path_to_embedding)
        except:
            raise EmbeddingError("Failed to load the embedding. In 99.999% of all cases this means your path is wrong. Good luck.")

        self.description = "This object represents the {} word embedding.".format(path_to_embedding)
        self.path_to_embedding = path_to_embedding.replace("/content/drive/My Drive/", "")

        training_data = []
        if path_training_data is not None:
            with open(path_training_data, "r") as f:
                for line in f:
                    stripped_line = line.strip()
                    line_list = stripped_line.split()
                    training_data.append(line_list)
            self.training_data = training_data
        else:
            self.training_data = None

        print("...finished loading.")

    def __str__(self):
        return "<Embedding {}>".format(self.path_to_embedding)

    def vocab(self, n_grams=None):
        """Return the vocabulary in the embedding."""

        if n_grams is not None:
            if n_grams not in range(1, 100):
                raise ValueError("n_grams arguemnt must be between 1 and 100. ")

            voc = []
            for word in self.vocab():
                if word.count("_") == n_grams - 1:
                    voc.append(word)
        else:
            voc = list(self._word_vectors.vocab)

        return sorted(voc)

    def vocab_size(self):
        """Return the size of the vocabulary in the embedding."""

        return len(list(self._word_vectors.vocab))

    def in_vocab(self, word):
        """Return whether word is in vocab."""
        return word in list(self._word_vectors.vocab)

    def load_training_data(self, path_training_data):
        """Load training data into embedding after embedding was created."""
        training_data = []
        if path_training_data is not None:
            with open(path_training_data, "r") as f:
                for line in f:
                    stripped_line = line.strip()
                    line_list = stripped_line.split()
                    training_data.append(line_list)
            self.training_data = training_data

    def context_in_training_data(self, word, n=3):
        """Return whether word is in vocab. Only works if training data was loaded.
        Args:
            word (str): Word to search for
            n (int): number of neighbouring words to print
        """
        if self.training_data is None:
            raise ValueError("This function needs access to the training data. "
                             "Please load the training data with the 'load_training_data()' "
                             "function and then try again. ")
        context = []
        for sentence in self.training_data:
            for idx, token in enumerate(sentence):
                if word == token:
                    start = 0 if (idx - n < 0) else idx - n
                    stop = len(sentence)-1 if (idx + n > len(sentence)-1) else idx + n
                    string = " ".join(sentence[start:stop+1])
                    context.append(string)
        return context

    def frequency_in_training_data(self, word):
        """Return how often the word appears in the training data. Only works if training data was loaded."""
        if self.training_data is None:
            raise ValueError("This function needs access to the training data. "
                             "Please load the training data with the 'load_training_data()' "
                             "function and then try again. ")

        counter = 0
        for sentence in self.training_data:
            for token in sentence:
                if word == token:
                    counter += 1
        return counter

    def sort_by_frequency_in_training_data(self, list_of_words):
        """Return a table in which the words are sorted by the frequency with which they appear in the training data."""
        if self.training_data is None:
            raise ValueError("This function needs access to the training data. "
                             "Please load the training data with the 'load_training_data()' "
                             "function and then try again. ")
        data = []
        for word in list_of_words:
            data.append([word, self.frequency_in_training_data(word)])

        res = pd.DataFrame(data, columns=['Word', 'Frequency'])
        res = res.sort_values(by='Frequency', axis=0, ascending=False)
        res = res.reset_index(drop=True)
        return res

    def vocab_sorted_by_frequency_in_training_data(self, n_grams=None, first_n=None, more_frequent_than=None):
        """Return the vocab sorted by the frequency with which they appear in the training data.

        Args:
             n_grams (int): by specifying "1, 2, 3..." you can limit the vocab to n_grams
             first_n (int): only return n most frequent words (-n for last n words). Overwrites up_to argument!
             more_frequent_than (int): only return words up to frequency "up_to"
        """

        if self.training_data is None:
            raise ValueError("This function needs access to the training data. "
                             "Please load the training data with the 'load_training_data()' "
                             "function and then try again. ")

        flat_training_data = [word for document in self.training_data for word in document]
        if n_grams is not None:
            if n_grams not in range(1, 100):
                raise ValueError("n_grams arguemnt must be between 1 and 100. ")
            flat_training_data = [word for word in flat_training_data if word.count("_") == n_grams - 1]

        frequencies = [[value, len(list(freq))] for value, freq in groupby(sorted(flat_training_data))]
        frequencies = sorted(frequencies, key=lambda x: x[1], reverse=True)

        res = pd.DataFrame({'Word': [f[0] for f in frequencies], 'Frequency': [f[1] for f in frequencies]})
        res = res.reset_index(drop=True)
        if first_n is not None:
            res = res.head(n=first_n)
        if more_frequent_than is not None:
            res = res[res['Frequency'] > more_frequent_than]
        return res

    def training_data_size(self):
        """Return how many words are in the training data. Only works if training data was loaded."""
        if self.training_data is None:
            raise ValueError("This function needs access to the training data. "
                             "Please load the training data with the 'load_training_data()' "
                             "function and then try again. ")

        counter = 0
        for sentence in self.training_data:
            counter += len(sentence)
        return counter

    def vocab_containing(self, word_part, show_frequency=False):
        """Return all words in the vocab that contain the word_part as a substring.

        Args:
            word_part (str): sequence of letters like "afro" or "whit"
            show_frequency (bool): if True, also show frequencies
        If the training data is loaded, this function will show frequencies as well.
        """
        if show_frequency:
            if self.training_data is None:
                raise ValueError("This function needs access to the training data. "
                                 "Please load the training data with the 'load_training_data()' "
                                 "function and then try again. ")

            subset = [[w, self.frequency_in_training_data(w)] for w in sorted(list(self._word_vectors.vocab)) if word_part in w]
            subset = pd.DataFrame(subset, columns=["Word", "Frequency"])
            subset = subset.sort_values(by='Frequency', axis=0, ascending=False)
        else:
            subset = [w for w in sorted(list(self._word_vectors.vocab)) if word_part in w]
        return subset

    def vector(self, word):
        """Return the vector representation of 'word' in the embedding."""

        return normalize(self._word_vectors[word])

    def vectors(self, list_of_words):
        """Return a list of the vector representations of each 'word' in 'list_of_words'."""

        list_of_vectors = []
        for word in list_of_words:
            list_of_vectors.append(self.vector(word))
        return list_of_vectors

    def centroid_of_difference_vectors(self, list_of_word_pairs):
        """Return the centroid vector of the differences of the word pairs provided."""

        difference_vecs = self.difference_vectors(list_of_word_pairs)
        difference_vecs = np.array(difference_vecs)
        centroid = np.mean(difference_vecs, axis=0)
        centroid = normalize(centroid)
        return centroid

    def centroid_of_vectors(self, list_of_words):
        """Return the centroid vector of the words provided."""

        vecs = [self.vector(word) for word in list_of_words]
        vecs = np.array(vecs)
        centroid = np.mean(vecs, axis=0)
        centroid = normalize(centroid)
        return centroid

    def difference_vector(self, word1, word2):
        """Return the normalised difference vector of 'word1' and 'word2'.
        Args:
            word1 (str): first word
            word2 (str): second word

        Returns:
            ndarray: normalised difference vector
            """

        vec1 = self.vector(word1)
        vec2 = self.vector(word2)
        if np.allclose(vec1, vec2, atol=1e-10):
            raise ValueError("The two words have the same vector representation, cannot compute their difference.")
        diff = vec1 - vec2
        diff = normalize(diff)
        return diff

    def difference_vectors(self, list_of_word_pairs):
        """Return a list of difference vectors for the word pairs provided."""

        difference_vecs = [self.difference_vector(word1, word2) for word1, word2 in list_of_word_pairs]
        return difference_vecs

    def variance_of_difference_vectors(self, list_of_word_pairs):
        """Return the vector of variances of the differences of the word pairs provided."""

        difference_vecs = self.difference_vectors(list_of_word_pairs)
        difference_vecs = np.array(difference_vecs)
        variances = np.var(difference_vecs, axis=0)
        return variances

    def similarity(self, word1, word2):
        """Return the cosine similarity between 'word1' and 'word2'."""

        return self._word_vectors.similarity(word1, word2)

    def similarities(self, list_of_word_pairs):
        """Return the cosine similarities between words in list of word pairs."""

        # collect results
        result = []
        for word1, word2 in list_of_word_pairs:
            sim = self.similarity(word1, word2)
            result.append([word1, word2, sim])

        # turn into dataframe
        result_dataframe = pd.DataFrame(result, columns=['Word1', 'Word2', 'Similarity'])
        return result_dataframe

    def most_similar(self, word, n=10):
        """Return the words most similar to 'word'."""

        return self._word_vectors.most_similar(word, topn=n)

    def most_similar_by_vector(self, vector, n=10):
        """Return the words most similar to 'vector'."""

        return self._word_vectors.similar_by_vector(vector, topn=n)

    def least_similar(self, word, n=10):
        """Return the words least similar to 'word'."""
        most_sim = self._word_vectors.most_similar(word, topn=self.vocab_size())
        last_n = most_sim[-n:]
        return last_n[::-1]

    def least_similar_by_vector(self, vector, n=10):
        """Return the words least similar to 'word'."""
        return self.most_similar_by_vector(-vector, n=n)

    def analogy(self, positive_list, negative_list, n=10):
        """Returns words close to positive words and far away from negative words, as
        proposed in https://www.aclweb.org/anthology/W14-1618.pdf"""
        return self._word_vectors.most_similar(positive=positive_list, negative=negative_list, topn=n)

    def dimension_alignments(self, list_of_word_pairs):
        """Construct the difference vectors for each word pair and return a dictionary of their similarities."""

        # make all unique combinations
        combinations_all_pairs = combinations_with_replacement(list_of_word_pairs, 2)
        # remove doubles
        # somehow the usual '==' does not work in colab
        combinations_all_pairs = [c for c in combinations_all_pairs if c[0][0] != c[1][0] or c[0][1] != c[1][1]]

        result = []
        for combination in list(combinations_all_pairs):
            pair1 = combination[0]
            pair2 = combination[1]

            # compute difference vectors
            diff_pair1 = self.vector(pair1[0]) - self.vector(pair1[1])
            diff_pair2 = self.vector(pair2[0]) - self.vector(pair2[1])

            # normalise differences
            diff_pair1 = normalize(diff_pair1)
            diff_pair2 = normalize(diff_pair2)

            sim = np.dot(diff_pair1, diff_pair2)
            sim = round(sim, 4)

            # save in nice format
            entry1 = pair1[0] + " - " + pair1[1]
            entry2 = pair2[0] + " - " + pair2[1]
            result.append([entry1, entry2, sim])

        result_dataframe = pd.DataFrame(result, columns=['Pair1', 'Pair2', 'Alignment'])
        return result_dataframe

    def dimension_quality(self, list_of_word_pairs):
        """Compute average similarity of difference vectors of all unique combinations of word pairs."""

        # make all unique combinations
        combinations_all_pairs = combinations_with_replacement(list_of_word_pairs, 2)
        # remove doubles
        # somehow the usual '==' does not work in colab
        combinations_all_pairs = [c for c in combinations_all_pairs if c[0][0] != c[1][0] or c[0][1] != c[1][1]]

        result = []
        for combination in list(combinations_all_pairs):
            pair1 = combination[0]
            pair2 = combination[1]

            # compute difference vectors
            diff_pair1 = self.vector(pair1[0]) - self.vector(pair1[1])
            diff_pair2 = self.vector(pair2[0]) - self.vector(pair2[1])
            # normalise differences
            diff_pair1 = normalize(diff_pair1)
            diff_pair2 = normalize(diff_pair2)

            sim = np.dot(diff_pair1, diff_pair2)
            sim = round(sim, 4)
            result.append(sim)

        return np.mean(result)

    def dimension_quality_baseline(self, n_pairs=5, n_trials=100):
        """Compute mean and variance of the distribution of social dimension quality values for randomly sampled
         word pairs. This can be used as a baseline for how likely it is that a random list of word pairs scores a high
         quality value.

        Args:
            n_pairs (int): number of word pairs to sample
            n_trials (int): number of times the experiment is repeated

        Returns:
            float, float: mean value and variance of the distribution of the resuls
        """

        all_words = np.array(self.vocab())

        results = []
        for repeat in range(n_trials):
            n_words = int(2*n_pairs)
            random_words = np.random.choice(all_words, size=n_words)
            random_word_pairs = random_words.reshape((n_pairs, 2))

            results.append(self.dimension_quality(random_word_pairs))

        results = np.array(results)

        return np.mean(results), np.var(results)

    def projection(self, neutral_word, word_pair):
        """Compute the projection of a word to the difference vector ("dimension") spanned by the word pair.

        Read the output as follows: if result is negative, it is closer to the SECOND word, else to the FIRST.

        Args:
            word (str): neutral word
            word_pair (List[str]): list of (two) words defining the dimension

        Returns:
            float
        """
        diff = self.difference_vector(word_pair[0], word_pair[1])
        vec = self.vector(neutral_word)
        projection = np.dot(diff, vec)
        return projection

    def projection_to_centroid_of_differences(self, neutral_word, word_pairs):
        """Compute the projection of a word to the centroid of the difference vectors spanned by the word pairs.

        Args:
            neutral_word (str or list[str]): neutral word like 'land' OR list of neutral words like ['land', 'nurse',...]
            word_pairs (List[List[str]]): list of word pairs
                like

                [['word1','word2'], ['anotherword1','anotherword2'], ...],

                OR dictionary of word pairs like

                {'gender': [['man', 'woman'], ['he', 'she'],...],
                 'race': [['black', 'white'], ,...],
                 ...
                 }

        Returns:
            DataFrame
        """
        if isinstance(neutral_word, str):
            neutral_word = [neutral_word]

        if isinstance(word_pairs, list):
            word_pairs = {'dim': word_pairs}

        data = []
        for neutral in neutral_word:

            neutral_vec = self.vector(neutral)

            for name, list_of_pairs in word_pairs.items():
                centroid = self.centroid_of_difference_vectors(list_of_pairs)
                dimension = "{}".format(name)
                example = "{}-{}".format(list_of_pairs[0][0], list_of_pairs[0][1])
                res = np.dot(neutral_vec, centroid)
                data.append([neutral, dimension, example, res])

        df = pd.DataFrame(data, columns=["neutral", "dimension", "example", "projection"])
        return df

    def projection_to_difference_of_cluster_centroids(self, neutral_word, word_pairs):
        """Compute the projection of a word to the difference between the two centroids computed from the cluster of
         words in each "pole".

         Args:
            neutral_word (str or list[str]): neutral word like 'land' OR list of neutral words like ['land', 'nurse',...]
            word_pairs (List[List[str]]): list of word pairs
                like

                [['word1','word2'], ['anotherword1','anotherword2'], ...],

                OR dictionary of word pairs like

                {'gender': [['man', 'woman'], ['he', 'she'],...],
                 'race': [['black', 'white'], ,...],
                 ...
                 }

        Returns:
            DataFrame
        """
        if isinstance(neutral_word, str):
            neutral_word = [neutral_word]

        if isinstance(word_pairs, list):
            word_pairs = {'dim': word_pairs}

        data = []
        for neutral in neutral_word:

            neutral_vec = self.vector(neutral)

            for name, list_of_pairs in word_pairs.items():
                left_cluster = [pair[0] for pair in list_of_pairs]
                right_cluster = [pair[1] for pair in list_of_pairs]
                centroid_left_cluster = self.centroid_of_vectors(left_cluster)
                centroid_right_cluster = self.centroid_of_vectors(right_cluster)
                diff = centroid_left_cluster - centroid_right_cluster
                diff = normalize(diff)
                res = np.dot(neutral_vec, diff)

                dimension = "{}".format(name)
                example = "{}-{}".format(list_of_pairs[0][0], list_of_pairs[0][1])
                data.append([neutral, dimension, example, res])

        df = pd.DataFrame(data, columns=["neutral", "dimension", "example", "projection"])
        return df

    def projection_to_differences_averaged(self, neutral_word, word_pairs):
        """Compute the average of the projection of a word to the difference vectors spanned by the word pairs.

        Args:
            neutral_word (str or list[str]): neutral word like 'land' OR list of neutral words like ['land', 'nurse',...]
            word_pairs (List[List[str]]): list of word pairs
                like

                [['word1','word2'], ['anotherword1','anotherword2'], ...],

                OR dictionary of word pairs like

                {'gender': [['man', 'woman'], ['he', 'she'],...],
                 'race': [['black', 'white'], ,...],
                 ...
                 }

        Returns:
            DataFrame
        """
        if isinstance(neutral_word, str):
            neutral_word = [neutral_word]

        if isinstance(word_pairs, list):
            word_pairs = {'dim': word_pairs}

        data = []
        for neutral in neutral_word:

            for name, list_of_pairs in word_pairs.items():
                projections = [self.projection(neutral, word_pair) for word_pair in list_of_pairs]
                dimension = "{}".format(name)
                example = "{}-{}".format(list_of_pairs[0][0], list_of_pairs[0][1])
                res = np.mean(projections)
                data.append([neutral, dimension, example, res])

        df = pd.DataFrame(data, columns=["neutral", "dimension", "example", "projection"])
        return df

    #
    # def projection_to_principal_component(self, neutral_word, list_of_word_pairs):
    #     """Compute the projection of a neutral word to the axis corresponding to the first principal
    #     component of the vectors corresponding to all words in the list.
    #
    #     Args:
    #         neutral_word (str): neutral word
    #         list_of_word_pairs (List[List[str]]): list of lists of two words, like [['word1','word2'],
    #         ['anotherword1','anotherword2'], ...]
    #
    #     Returns:
    #         float
    #     """
    #     neutral_word = self.vector(neutral_word)
    #
    #     left = []
    #     right = []
    #     for word_pair in list_of_word_pairs:
    #         left.append(self.vector(word_pair[0]))
    #         right.append(self.vector(word_pair[1]))
    #
    #     X_left = np.array(left)
    #     pca_transformer = PCA(n_components=1)
    #     pca_transformer.fit_transform(X_left)
    #     principal_axis_left = pca_transformer.components_[0]
    #
    #     X_right = np.array(right)
    #     pca_transformer = PCA(n_components=1)
    #     pca_transformer.fit_transform(X_right)
    #     principal_axis_right = pca_transformer.components_[0]
    #
    #     diff = principal_axis_left - principal_axis_right
    #     diff = normalize(diff)
    #     return np.dot(neutral_word, diff)

    def projections(self, neutral_words, word_pairs):
        """Compute projections of a word to difference vectors ("dimensions") spanned by multiple
        word pairs. Return result as a dataframe.

        Args:
            word (str): neutral word
            list_of_word_pairs (List[List[str]]): list of word pairs defining the dimension

        Returns:
            DataFrame
        """
        result = []
        for word in neutral_words:
            for word_pair in word_pairs:
                projection = self.projection(word, word_pair)
                result.append([word, word_pair[0] + " - " + word_pair[1], projection])

        result_dataframe = pd.DataFrame(result, columns=['Neutral word', 'Dimension', 'Projection'])
        if self.training_data is not None:
            result_dataframe['Neutral_freq'] = [self.frequency_in_training_data(word) for word in result_dataframe['Neutral word']]
        return result_dataframe

    def plot_dimension_quality_baseline(self, n_pairs=5, n_trials=100):
        """Plots the distribution of social dimension quality values for randomly sampled
         word pairs. This can be used as a baseline for how likely it is that a random list of word pairs scores a high
         quality value.

        Args:
            n_pairs (int): number of word pairs to sample
            n_trials (int): number of times the sampling is repeated
        """

        all_words = np.array(self.vocab())

        results = []
        for repeat in range(n_trials):
            n_words = int(2*n_pairs)
            random_words = np.random.choice(all_words, size=n_words)
            random_word_pairs = random_words.reshape((n_pairs, 2))

            results.append(self.dimension_quality(random_word_pairs))

        plt.hist(results, bins=200, range=(-1, 1))
        plt.show()

    def plot_pca(self, list_of_words, n_comp=2):
        """Plot the words in list_of_words in a PCA plot.

        Args:
            list_of_words (List[str]): list of words
            n_comp (int): number of principal components
        """
        X = np.array([self.vector(word) for word in list_of_words])
        pca_transformer = PCA(n_components=n_comp)
        pca = pca_transformer.fit_transform(X)

        plt.figure()
        plt.scatter(pca[:, 0], pca[:, 1])
        # Adding annotations
        for i, word in enumerate(list_of_words):
            plt.annotate(' ' + word, xy=(pca[i, 0], pca[i, 1]))

        plt.show()

    def plot_tsne(self, list_of_words, tsne_ncomp=2, pep=15):
        """Plot the words in list_of_words in a PCA plot.

        Args:
            list_of_words (List[str]): list of words
            n_comp (int): number of principal components
        """

        X = np.array([self.vector(word) for word in list_of_words])
        tsne = TSNE(n_components=tsne_ncomp, random_state=0, perplexity=pep).fit_transform(X)

        plt.figure()
        plt.scatter(tsne[:, 0], tsne[:, 1])
        # Adding annotations
        for i, word in enumerate(list_of_words):
            plt.annotate(' ' + word, xy=(tsne[i, 0], tsne[i, 1]))

        plt.show()

    def principal_component_vectors(self, list_of_words=None, n_components=3):
        """Get the n_component first principal component vectors of the words.

        Args:
            list_of_words (list[str]): list of words
            n_components (int): number of components
        Returns:
            DataFrame
        """

        X = [self.vector(word) for word in list_of_words]
        X = np.array(X)
        pca_transformer = PCA(n_components=n_components)
        pca_transformer.fit_transform(X)

        principal_vectors = pca_transformer.components_[:n_components]
        principal_vectors = [normalize(vec) for vec in principal_vectors]
        return principal_vectors

    def words_closest_to_principal_components(self, list_of_words=None, n_components=3, n=5):
        """Get the words closest to the principal components of the word vectors in the vocab.

        Args:
            n_components (int): number of components
            n (int): number of neighbours to print for each component

        Returns:
            DataFrame
        """

        X = []
        if list_of_words is None:
            list_of_words = self.vocab()
        for word in list_of_words:
            X.append(self.vector(word))

        X = np.array(X)
        pca_transformer = PCA(n_components=n_components)
        pca_transformer.fit_transform(X)

        data = {}
        for i in range(n_components):

            principal_axis = pca_transformer.components_[i]
            principal_axis = normalize(principal_axis)

            data['princ_comp' + str(i+1)] = self.most_similar([principal_axis], n=n)

        df = pd.DataFrame(data)
        return df

    def plot_word_as_colourarray(self, word):
        """Visualise a word's vector as an array of coloured blocks.
        """

        X = np.array([self.vector(word)])
        plt.figure(figsize=(20, 1))
        plt.imshow(X)
        plt.yticks(ticks=[0], labels=[word])
        plt.xlabel("dimension in embedding space")
        plt.show()

    def plot_vector_as_colourarray(self, vector, name='vector'):
        """Visualise a vector as an array of coloured blocks.
        """
        X = np.array(vector)
        plt.figure(figsize=(20, 1))
        plt.imshow(X)
        plt.yticks(ticks=[0], label=name)
        plt.xlabel("dimension in embedding space")
        plt.show()

    def plot_words_as_colourarray(self, list_of_words, include_centroid=False, include_princ_comp=None,
                                  include_diff_vectors=None):
        """Visualise word vectors as arrays of coloured blocks.

        Args:
            include_centroid (bool): also plot their centroid vector
            include_princ_comp (int): also plot principal comp vecs
            include_diff_vectors (bool): also plot difference vecs
        """
        list_of_words = list_of_words.copy()

        vecs = [self.vector(word) for word in list_of_words]

        extra_vecs = []
        extra_words = []
        if include_centroid:
            extra_vecs.append(self.centroid_of_vectors(list_of_words))
            extra_words.append('centroid')
        if include_princ_comp is not None:
            if not isinstance(include_princ_comp, int):
                raise ValueError("invclude_princ_comp must be an integer like 1, 2, 3...")
            extra_vecs.extend(self.principal_component_vectors(list_of_words))
            for i in range(include_princ_comp):
                extra_words.append('princ_comp' + str(i))
        if include_diff_vectors is not None:
            diff_pairs = make_pairs(list_of_words, list_of_words, exclude_doubles=True)
            extra_vecs.extend(self.difference_vectors(diff_pairs))
            extra_words.extend([word1 + "-" + word2 for word1, word2 in diff_pairs])

        vecs.extend(extra_vecs)
        list_of_words.extend(extra_words)

        X = np.array(vecs)
        plt.figure(figsize=(20, 1 + 0.2 * len(list_of_words)))
        plt.imshow(X)
        plt.yticks(ticks=range(len(list_of_words)), labels=list_of_words)
        plt.xlabel("dimension in embedding space")
        plt.show()

    def plot_vectors_as_colourarray(self, list_of_vectors, names=None):
        """Visualise vectors as arrays of coloured blocks."""
        if names is None:
            names = ["vector" + str(i) for i in range(len(list_of_vectors))]
        X = np.array(list_of_vectors)
        plt.figure(figsize=(20, 1 + 0.2 * len(list_of_vectors)))
        plt.imshow(X)
        plt.yticks(ticks=range(len(list_of_vectors)), labels=names)
        plt.xlabel("dimension in embedding space")
        plt.show()


class EmbeddingEnsemble:
    """Applies actions to an list_of_embeddings of trained_embeddings."""

    def __init__(self, path_to_embeddings, path_training_data=None):

        self.list_of_embeddings = []

        if isinstance(path_to_embeddings, list):

            paths = path_to_embeddings

        else:

            paths = glob.glob(path_to_embeddings + '*.emb')

            if len(paths) == 0:
                raise EmbeddingError("Failed to find any appropriate file. Please make sure that "
                                     "there are trained_embeddings under this path.".format(path_to_embeddings))

        # Iterate through all paths
        for path in paths:

            try:
                # load the word vectors of an embedding
                emb = WordEmbedding(path)
            except FileNotFoundError:
                raise EmbeddingError("Failed to load the trained_embeddings {}. Please make sure that "
                                     "the path to this file really exists.".format(path))

            self.list_of_embeddings.append(emb)

        self.description = "This object represents the list_of_embeddings {} of {} word trained_embeddings."\
            .format(path_to_embeddings, len(self.list_of_embeddings))

        training_data = []
        if path_training_data is not None:
            with open(path_training_data, "r") as f:
                for line in f:
                    stripped_line = line.strip()
                    line_list = stripped_line.split()
                    training_data.append(line_list)
            self.training_data = training_data
        else:
            self.training_data = None

        # standard columns of data frame
        self.cols = ['emb' + str(i+1) for i in range(len(self.list_of_embeddings))]

        print("Loaded {} trained_embeddings.".format(len(self.list_of_embeddings)))

    def load_training_data(self, path_training_data):
        """Load training data into embedding after embedding was created."""
        training_data = []
        if path_training_data is not None:
            with open(path_training_data, "r") as f:
                for line in f:
                    stripped_line = line.strip()
                    line_list = stripped_line.split()
                    training_data.append(line_list)
            self.training_data = training_data

    def context_in_training_data(self, word, n=3):
        """Return whether word is in vocab. Only works if training data was loaded.
        Args:
            word (str): Word to search for
            n (int): number of neighbouring words to print
        """
        if self.training_data is None:
            raise ValueError("This function needs access to the training data. "
                             "Please load the training data with the 'load_training_data()' "
                             "function and then try again. ")
        context = []
        for sentence in self.training_data:
            for idx, token in enumerate(sentence):
                if word == token:
                    start = 0 if (idx - n < 0) else idx - n
                    stop = len(sentence) - 1 if (idx + n > len(sentence) - 1) else idx + n
                    string = " ".join(sentence[start:stop + 1])
                    context.append(string)
        return context

    def frequency_in_training_data(self, word):
        """Return how often the word appears in the training data. Only works if training data was loaded."""
        if self.training_data is None:
            raise ValueError("This function needs access to the training data. "
                             "Please load the training data with the 'load_training_data()' "
                             "function and then try again. ")

        counter = 0
        for sentence in self.training_data:
            for token in sentence:
                if word == token:
                    counter += 1
        return counter

    def sort_by_frequency_in_training_data(self, list_of_words):
        """Return a table in which the words are sorted by the frequency with which they appear in the training data."""
        if self.training_data is None:
            raise ValueError("This function needs access to the training data. "
                             "Please load the training data with the 'load_training_data()' "
                             "function and then try again. ")
        data = []
        for word in list_of_words:
            data.append([word, self.frequency_in_training_data(word)])

        res = pd.DataFrame(data, columns=['Word', 'Frequency'])
        res = res.sort_values(by='Frequency', axis=0, ascending=False)
        res = res.reset_index(drop=True)
        return res

    def vocab_sorted_by_frequency_in_training_data(self, n_grams=None, first_n=None, more_frequent_than=None):
        """Return the vocab sorted by the frequency with which they appear in the training data.

        Args:
             n_grams (int): by specifying "1, 2, 3..." you can limit the vocab to n_grams
             first_n (int): only return n most frequent words (-n for last n words). Overwrites up_to argument!
             more_frequent_than (int): only return words up to frequency "up_to"
        """

        if self.training_data is None:
            raise ValueError("This function needs access to the training data. "
                             "Please load the training data with the 'load_training_data()' "
                             "function and then try again. ")

        if n_grams is not None:
            if n_grams not in range(1, 100):
                raise ValueError("n_grams arguemnt must be between 1 and 100. ")
        if n_grams is None or n_grams is 1:
            print("This method takes a while to execute...get a cup of tea if you like.")

        data = []
        for word in self.vocab():
            if n_grams is not None:
                if word.count("_") == n_grams - 1:
                    data.append([word, self.frequency_in_training_data(word)])
            else:
                data.append([word, self.frequency_in_training_data(word)])

        res = pd.DataFrame(data, columns=['Word', 'Frequency'])
        res = res.sort_values(by='Frequency', axis=0, ascending=False)
        res = res.reset_index(drop=True)
        if first_n is not None:
            res = res.head(n=first_n)
        if more_frequent_than is not None:
            res = res[res['Frequency'] > more_frequent_than]
        return res

    def training_data_size(self):
        """Return how many words are in the training data. Only works if training data was loaded."""
        if self.training_data is None:
            raise ValueError("This function needs access to the training data. "
                             "Please load the training data with the 'load_training_data()' "
                             "function and then try again. ")

        counter = 0
        for sentence in self.training_data:
            counter += len(sentence)
        return counter

    def n_embeddings(self):
        """Return the number of trained_embeddings in the list_of_embeddings."""
        return len(self.list_of_embeddings)

    def vocab_size(self):
        """Return the size of the vocabulary in the trained_embeddings."""
        individual = [emb.vocab_size() for emb in self.list_of_embeddings]

        data = [['size'] + individual]
        df = pd.DataFrame(data, columns=[''] + self.cols)
        df['MEAN'] = df.mean(numeric_only=True, axis=1)
        df['STD'] = df.std(numeric_only=True, axis=1)
        return df

    def shared_vocab(self):
        """Return the subset of the vocab that is shared by all trained_embeddings in the list_of_embeddings
        (i.e. the intersection of their vocab)."""
        vocabs = [emb.vocab() for emb in self.list_of_embeddings]
        shared_vocab = set(vocabs[0]).intersection(*vocabs)
        return list(shared_vocab)

    def in_vocab(self, word):
        """Return whether word is in vocab."""
        individual = [emb.in_vocab(word) for emb in self.list_of_embeddings]
        data = [['in_vocab'] + individual]
        df = pd.DataFrame(data, columns=[''] + self.cols)
        df['MEAN'] = df.mean(numeric_only=True, axis=1)
        return df

    def similarity(self, word1, word2):
        """Return the cosine similarities between 'word1' and 'word2'."""
        individual = [emb.similarity(word1, word2) for emb in self.list_of_embeddings]
        data = [['similarity'] + individual]
        df = pd.DataFrame(data, columns=[''] + self.cols)
        df['MEAN'] = df.mean(numeric_only=True, axis=1)
        df['STD'] = df.std(numeric_only=True, axis=1)
        return df

    def similarities(self, list_of_word_pairs):
        """Return the cosine similarities between the pairs of words."""
        base_df = self.list_of_embeddings[0].similarities(list_of_word_pairs)
        base_df = base_df.rename({"Similarity": "Sim_emb1"}, axis=1)
        for idx, emb in enumerate(self.list_of_embeddings[1:]):
            df = emb.similarities(list_of_word_pairs)
            df = df.rename({"Similarity": "Sim_emb" + str(idx+2)}, axis=1)
            base_df = pd.merge(base_df, df, on=['Word1', 'Word2'])

        base_df['MEAN'] = base_df.mean(numeric_only=True, axis=1)
        base_df['STD'] = base_df.std(numeric_only=True, axis=1)
        if self.training_data is not None:
            base_df['Word1_freq'] = [self.frequency_in_training_data(word) for word in base_df['Word1']]
            base_df['Word2_freq'] = [self.frequency_in_training_data(word) for word in base_df['Word2']]
        return base_df

    def similarity_test(self, n_pairs=1000, min_frequency=20):
        """Return the average standard deviation of the similarity between n_pairs randomly sampled pairs of words.
        All words are guaranteed to appear at least min_frequency times in the shared training data."""

        if self.training_data is None:
            raise ValueError("This function needs access to the training data. "
                             "Please load the training data with the 'load_training_data()' "
                             "function and then try again. ")

        all_words = np.array(self.shared_vocab())
        n_words = int(2 * n_pairs)

        # create pairs of random words with frequency larger than min_frequency
        random_words = []
        counter = 0
        while counter < n_words:
            random_word = np.random.choice(all_words)
            if self.frequency_in_training_data(random_word) >= min_frequency:
                random_words.append(random_word)
                counter += 1
        random_word_pairs = np.array(random_words).reshape((n_pairs, 2))

        df = self.similarities(random_word_pairs)
        average_std = df['STD'].mean()
        return average_std

    def most_similar(self, word, n=10):
        """Return the words most similar to 'word' in each embedding."""
        data_raw = [emb.most_similar(word, n=n) for emb in self.list_of_embeddings]
        data_raw = [[str(i+1) for i in range(n)]] + data_raw
        data = np.array(data_raw).T
        df = pd.DataFrame(data, columns=['rank'] + self.cols)
        return df

    def least_similar(self, word, n=10):
        """Return the words least similar to 'word' in each embedding."""
        data_raw = [emb.least_similar(word, n=n) for emb in self.list_of_embeddings]
        data_raw = [[str(i+1) for i in range(n)]] + data_raw
        data = np.array(data_raw).T
        df = pd.DataFrame(data, columns=['rank'] + self.cols)
        return df

    def analogy(self, positive_list, negative_list, n=10):
        """Returns words close to positive words and far away from negative words, as
        proposed in https://www.aclweb.org/anthology/W14-1618.pdf"""
        data_raw = [emb.analogy(positive_list, negative_list, n=n) for emb in self.list_of_embeddings]
        data_raw = [[str(i+1) for i in range(n)]] + data_raw
        data = np.array(data_raw).T
        df = pd.DataFrame(data, columns=['rank'] + self.cols)
        return df

    def dimension_alignments(self, list_of_word_pairs):
        """Construct the difference vectors for each word pair and return their similarities."""

        base_df = self.list_of_embeddings[0].dimension_alignments(list_of_word_pairs)
        base_df = base_df.rename({"Alignment": "alignment_emb1"}, axis=1)
        for idx, emb in enumerate(self.list_of_embeddings[1:]):
            df = emb.dimension_alignments(list_of_word_pairs)
            df = df.rename({"Alignment": "alignment_emb" + str(idx+2)}, axis=1)
            base_df = pd.merge(base_df, df, on=['Pair1', 'Pair2'])

        base_df['MEAN'] = base_df.mean(numeric_only=True, axis=1)
        base_df['STD'] = base_df.std(numeric_only=True, axis=1)
        return base_df

    def dimension_quality(self, list_of_word_pairs):
        """Compute average similarity of difference vectors of all unique combinations of word pairs."""
        individual = [emb.dimension_quality(list_of_word_pairs) for emb in self.list_of_embeddings]
        data = [['mean_alignment'] + individual]
        df = pd.DataFrame(data, columns=[''] + self.cols)
        df['MEAN'] = df.mean(numeric_only=True, axis=1)
        df['STD'] = df.std(numeric_only=True, axis=1)
        return df

    def projection(self, neutral_word, word_pair):
        """Compute the projection of a word to the difference vector ("dimension") spanned by the word pair.

        Args:
            word (str): neutral word
            word_pair (List[str]): list of (two) words defining the dimension

        Returns:
            float
        """
        data = [['projection'] + [emb.projection(neutral_word, word_pair) for emb in self.list_of_embeddings]]
        df = pd.DataFrame(data, columns=[''] + self.cols)
        df['MEAN'] = df.mean(numeric_only=True, axis=1)
        df['STD'] = df.std(numeric_only=True, axis=1)
        return df

    def projections(self, neutral_words, word_pairs):
        """Compute projections of a word to difference vectors ("dimensions") spanned by multiple
        word pairs. Return result as a dataframe.

        Args:
            word (str): neutral word
            list_of_word_pairs (List[List[str]]): list of word pairs defining the dimension

        Returns:
            DataFrame
        """

        base_df = self.list_of_embeddings[0].projections(neutral_words, word_pairs)
        base_df = base_df.rename({"Projection": "projection_emb1"}, axis=1)
        for idx, emb in enumerate(self.list_of_embeddings[1:]):
            df = emb.projections(neutral_words, word_pairs)
            df = df.rename({"Projection": "projection_emb" + str(idx+2)}, axis=1)
            base_df = pd.merge(base_df, df, on=['Neutral word', 'Dimension'])

        base_df['MEAN'] = base_df.mean(numeric_only=True, axis=1)
        base_df['STD'] = base_df.std(numeric_only=True, axis=1)
        return base_df

    def projection_to_centroid(self, neutral_word, list_of_word_pairs):

        individual = [emb.projection_to_centroid(neutral_word, list_of_word_pairs) for emb in self.list_of_embeddings]
        data = [['projection'] + individual]
        df = pd.DataFrame(data, columns=[''] + self.cols)
        df['MEAN'] = df.mean(numeric_only=True, axis=1)
        df['STD'] = df.std(numeric_only=True, axis=1)
        return df

    def projection_average(self, neutral_word, list_of_word_pairs):

        individual = [emb.projection_average(neutral_word, list_of_word_pairs) for emb in self.list_of_embeddings]
        data = [['projection'] + individual]
        df = pd.DataFrame(data, columns=[''] + self.cols)
        df['MEAN'] = df.mean(numeric_only=True, axis=1)
        df['STD'] = df.std(numeric_only=True, axis=1)
        return df
