# This file contains a wrapper class that represents word trained embeddings
from gensim.models import KeyedVectors
from mma_word_embeddings.utils import normalize_vector, make_pairs
import numpy as np
from itertools import combinations_with_replacement, combinations, product
import pandas as pd
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from sklearn.manifold import TSNE
from collections import Counter
from random import sample
import glob
import seaborn as sns
import networkx as nx
import scipy.cluster.hierarchy as sch
from scipy.stats import pearsonr
import os

# Make pandas print full data frame
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)
pd.options.display.float_format = '{:,.4f}'.format

COLORMAP = mcolors.LinearSegmentedColormap.from_list("MyCmapName", ["r", "w", "g"])


class EmbeddingError(Exception):
    """Exception raised by a Model object when something is wrong.
    """


class WordEmbedding:
    """Representation of a word embedding, which is a map from word strings to vectors."""

    def __init__(self, path_to_embedding):

        print("Loading embedding {} ... ".format(path_to_embedding))

        try:
            # load the word vectors of an embedding
            self._word_vectors = KeyedVectors.load(path_to_embedding)

        except:
            raise EmbeddingError("Failed to load the embedding. In 99.999% of all cases this means your "
                                 "path is wrong. Good luck.")

        self.description = "This object represents the {} word embedding.".format(path_to_embedding)
        self.path_to_embedding = path_to_embedding.replace("/content/drive/My Drive/", "")

        print("...finished loading.")

    def __str__(self):
        return "<Embedding {}>".format(self.path_to_embedding)

    def vocab(self, n_grams=None):
        """Return the vocabulary in the embedding."""

        if n_grams is not None:
            if n_grams not in range(1, 100):
                raise ValueError("n_grams arguemnt must be between 1 and 100. ")

            voc = []
            for word in self._word_vectors.key_to_index:
                if word.count("_") == n_grams - 1:
                    voc.append(word)
        else:
            voc = list(self._word_vectors.index_to_key)

        return sorted(voc)

    def vocab_size(self):
        """Return the size of the vocabulary in the embedding."""

        return len(self._word_vectors.key_to_index)

    def in_vocab(self, word):
        """Return whether word is in vocab."""
        return word in list(self._word_vectors.index_to_key)

    def random_words(self, n_words=100, min_frequency=None):
        """Return a list of random words from the vocab of this embedding.

        If the training data is loaded, the minimum frequency can be specified.

        Args:
            n_words (int): number of random words to select
            min_frequency (int): minimum frequency of the word in the training data

        Returns:
            list(str): random words
        """

        if min_frequency is not None and self.training_data is None:
            raise ValueError("This function needs access to the training data. "
                             "Please load the training data with the 'load_training_data()' "
                             "function and then try again. ")

        if min_frequency is not None:
            vocab = self.vocab_sorted_by_frequency_in_training_data(more_frequent_than=min_frequency)
            vocab = vocab['Word'].tolist()
        else:
            vocab = list(self._word_vectors.key_to_index)

        return sample(vocab, n_words)

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

            subset = [[w, self.frequency_in_training_data(w)] for w in sorted(list(self._word_vectors.key_to_index)) if
                      word_part in w]
            subset = pd.DataFrame(subset, columns=["Word", "Frequency"])
            subset = subset.sort_values(by='Frequency', axis=0, ascending=False)
        else:
            subset = [w for w in sorted(list(self._word_vectors.key_to_index)) if word_part in w]
        return subset

    def vector(self, word):
        """Return the normalized vector representation of 'word' in the embedding."""

        return normalize_vector(self._word_vectors[word])

    def vectors(self, list_of_words):
        """Return a list of the normalized vector representations of each 'word' in 'list_of_words'."""
        return [self.vector(word) for word in list_of_words]

    def difference_vector(self, word1, word2, normalize=False):
        """Return the difference vector vec(word1) - vec(word2).
        Args:
            word1 (str): first word
            word2 (str): second word
            normalize: whether the difference should be normalised

        Returns:
            ndarray: normalised difference vector
        """

        vec1 = self.vector(word1)
        vec2 = self.vector(word2)
        if np.allclose(vec1, vec2, atol=1e-10):
            raise ValueError("The two words have the same vector representation, cannot compute their difference.")
        diff = vec1 - vec2

        if normalize:
            diff = normalize_vector(diff)
        return diff

    def difference_vectors(self, list_of_word_pairs, normalize=False):
        """Return a list of difference vectors for the word pairs provided."""
        return [self.difference_vector(word1, word2, normalize=normalize) for word1, word2 in list_of_word_pairs]

    def centroid_of_difference_vectors(self, list_of_word_pairs, normalize_diffs=False, normalize_centroid=False):
        """Return the centroid vector of the differences of the word pairs provided."""

        difference_vecs = self.difference_vectors(list_of_word_pairs, normalize=normalize_diffs)
        centroid = np.mean(np.array(difference_vecs), axis=0)
        if normalize_centroid:
            return normalize_vector(centroid)
        return centroid

    def centroid_of_vectors(self, list_of_words, normalize=False):
        """Return the centroid vector of the words provided."""

        vecs = [self.vector(word) for word in list_of_words]
        vecs = np.array(vecs)
        centroid = np.mean(vecs, axis=0)
        if normalize:
            return normalize_vector(centroid)
        return centroid

    def principal_components(self, list_of_words=None, n_components=3, normalize=False):
        """Get the n_component first principal component vectors of the words.

        Args:
            list_of_words (list[str]): list of words
            n_components (int): number of components
        Returns:
            list of arrays
        """

        X = [self.vector(word) for word in list_of_words]
        X = np.array(X)
        pca_transformer = PCA(n_components=n_components)
        pca_transformer.fit_transform(X)

        principal_vectors = pca_transformer.components_[:n_components]
        if normalize:
            principal_vectors = [normalize_vector(vec) for vec in principal_vectors]
        return principal_vectors

    def principal_components_variance(self, list_of_words=None, n_components=3):
        """The amount of variance explained by each of the principal components of the words in the list.

        Args:
            list_of_words (list[str]): list of words
            n_components (int): number of components
        Returns:
            list
        """

        X = [self.vector(word) for word in list_of_words]
        X = np.array(X)
        pca_transformer = PCA(n_components=n_components)
        pca_transformer.fit_transform(X)

        explained_variance = pca_transformer.explained_variance_[:n_components]
        return explained_variance

    def similarity(self, word1, word2):
        """Return the similarity between 'word1' and 'word2'. The result is a value between -1 and 1."""
        return np.dot(self.vector(word1), self.vector(word2))

    def similarities(self, list_of_word_pairs):
        """Return the cosine similarities between words in list of word pairs."""
        result = [[word1, word2, round(self.similarity(word1, word2), 3)] for word1, word2 in list_of_word_pairs]
        result_dataframe = pd.DataFrame(result, columns=['Word1', 'Word2', 'Similarity'])
        result_dataframe = result_dataframe.sort_values(["Similarity"], axis=0)
        return result_dataframe

    def most_similar(self, word, n=10):
        """Return the words most similar to 'word'."""

        ms = self._word_vectors.most_similar(word, topn=n)
        ms = [(word, round(s, 3)) for word, s in ms]
        return ms

    def most_similar_by_vector(self, vector, n=10):
        """Return the words most similar to 'vector'."""

        return self._word_vectors.similar_by_vector(vector, topn=n)

    def similarities_of_differences(self, list_of_word_pairs):
        """Construct the difference vectors for each word pair and return a dictionary of their similarities."""

        # make all unique combinations of pairs
        combinations_all_pairs = combinations_with_replacement(list_of_word_pairs, 2)
        # remove doubles - somehow the usual '==' does not work in colab
        combinations_all_pairs = [c for c in combinations_all_pairs if c[0][0] != c[1][0] or c[0][1] != c[1][1]]

        result = []
        for combination in list(combinations_all_pairs):
            pair1 = combination[0]
            pair2 = combination[1]

            # compute difference vectors
            diff_pair1 = self.vector(pair1[0]) - self.vector(pair1[1])
            diff_pair2 = self.vector(pair2[0]) - self.vector(pair2[1])

            # normalise differences
            diff_pair1 = normalize_vector(diff_pair1)
            diff_pair2 = normalize_vector(diff_pair2)

            sim = np.dot(diff_pair1, diff_pair2)
            sim = round(sim, 4)

            # save in nice format
            entry1 = pair1[0] + " - " + pair1[1]
            entry2 = pair2[0] + " - " + pair2[1]
            result.append([entry1, entry2, sim])

        result_dataframe = pd.DataFrame(result, columns=['Pair1', 'Pair2', 'Alignment'])
        return result_dataframe

    def words_closest_to_principal_components(self, list_of_words=None, n_components=3, n=5):
        """Get the words closest to the principal components of the word vectors in the vocab.

        Args:
            list_of_words (list[str]): list of words
            n_components (int): number of components
            n (int): number of neighbours to print for each component

        Returns:
            DataFrame
        """

        if list_of_words is None:
            list_of_words = self.key_to_index

        principal_vecs = self.principal_components(list_of_words, n_components=n_components, normalize=True)

        data = {}

        for idx, vec in enumerate(principal_vecs):
            data['princ_comp' + str(idx + 1)] = self.most_similar([vec], n=n)

        df = pd.DataFrame(data)
        return df

    def analogy(self, negative_list, positive_list, n=10):
        """Returns words close to positive words and far away from negative words, as
        proposed in https://www.aclweb.org/anthology/W14-1618.pdf"""
        return self._word_vectors.most_similar(negative=negative_list, positive=positive_list, topn=n)

    def analogy_test(self, negative_list, positive_list, test_word, n=10):
        """Check whether the test word appears in the n closest words to the vector resulting from
           an analogy computation as proposed in https://www.aclweb.org/anthology/W14-1618.pdf .

        Args:
            negative_list (list[str]): list of negative words [a is to]
            positive_list (list[str]): list of positive words [b like c]
            n (int): how many neighbours to compute

        Returns:
            bool
        """

        for word in positive_list + negative_list:

            if not self.in_vocab(word):
                raise ValueError(f"{word} not found in vocab.")

        analogy_tuples = self._word_vectors.most_similar(positive=positive_list, negative=negative_list, topn=n)
        analogy_words = [tpl[0] for tpl in analogy_tuples]
        if test_word in analogy_words:
            return True
        else:
            return False

    def projection(self, test_word, word_pair):
        """Compute the projection of a word to the normalized difference vector of the word pair.

        Read the output as follows: if result is negative, it is closer to the SECOND word, else to the FIRST.

        Args:
            test_word (str): test word
            word_pair (List[str]): list of (two) words defining the dimension

        Returns:
            float in [-1, 1]
        """
        diff = self.difference_vector(word_pair[0], word_pair[1], normalize=True)
        vec = self.vector(test_word)
        projection = np.dot(diff, vec)
        return projection

    def projections(self, test_words, word_pairs):
        """Compute projections of a word to difference vectors ("dimensions") spanned by multiple
        word pairs. Return result as a dataframe.

        Args:
            word (str): test word
            list_of_word_pairs (List[List[str]]): list of word pairs defining the dimension

        Returns:
            DataFrame
        """
        result = []
        for word in test_words:
            for word_pair in word_pairs:
                projection = self.projection(word, word_pair)
                result.append([word, word_pair[0] + " - " + word_pair[1], projection])

        result_dataframe = pd.DataFrame(result, columns=['test', 'dimension', 'projection'])
        if self.training_data is not None:
            result_dataframe['test_freq'] = [self.frequency_in_training_data(word) for word in result_dataframe['test']]
        return result_dataframe

    def projections_to_bipolar_dimensions(self, test, dimensions, normalize_before=False, normalize_centroids=True):
        """ Compute the projections of test words onto bipolar dimensions. Each bipolar dimension is constructed from
        two clusters of words.

        The dimension is constructed as difference of the two (unnormalised) centroids of the words in the clusters.

        Note that up to a constant which is independent of the test word, this is the same as

        * Computing the difference between the averages of the projections of a test word to words in
          the two clusters.

        * Computing the average of projections onto the difference between word pairs formed from the cluster.

        * Computing the projection onto the centroid of differences between word pairs formed from the cluster.

         Args:
            test (str or list[str]): test word like 'land' OR list of test
              words like ['land', 'nurse',...]
            dimensions (dict): dictionary of lists of two clusters like

                    {'gender (male-female)': [['man', 'he',...], ['girl', 'her',...]],
                     'race (black-white)': [['black', ...], ['white', ...]],
                     ...
                     }
        Returns:
            DataFrame
        """
        if isinstance(test, str):
            test_words = [test]
        else:
            test_words = test

        data = []
        for test_word in test_words:

            test_vec = self.vector(test_word)
            row = [test_word]

            for dim_clusters in dimensions.values():

                if len(dim_clusters) != 2:
                    raise ValueError("Generating words must be a list of exactly two lists that contain words.")

                centroid_left_cluster = self.centroid_of_vectors(dim_clusters[0])
                centroid_right_cluster = self.centroid_of_vectors(dim_clusters[1])

                if normalize_centroids:
                    centroid_left_cluster = centroid_left_cluster / np.linalg.norm(centroid_left_cluster)
                    centroid_right_cluster = centroid_right_cluster / np.linalg.norm(centroid_right_cluster)

                diff = centroid_left_cluster - centroid_right_cluster
                if normalize_before:
                    diff = normalize_vector(diff)
                res = np.dot(test_vec, diff)
                row.append(res)

            data.append(row)

        cols = ["test_word"] + list(dimensions)

        df = pd.DataFrame(data, columns=cols)
        df = df.sort_values(cols[1:], axis=0, ascending=False)
        return df

    def projections_to_unipolar_dimensions(self, test, dimensions, normalize_before=True):
        """Compute the projection of a test word onto unipolar dimensions.

           The unipolar dimension is the centroid of a cluster of words.

        Args:
            test (str or list[str]): test word like 'land' OR list of test
                                        words like ['land', 'nurse',...]
            dimensions (dict): dictionary of clusters like

                    {'male': ['man', 'he',...]
                     'female': ['him', 'her' ...],
                     ...
                     }
        Returns:
            DataFrame
        """
        if isinstance(test, str):
            test_words = [test]
        else:
            test_words = test

        data = []
        for test_word in test_words:

            test_vec = self.vector(test_word)
            row = [test_word]

            for dim_cluster in dimensions.values():

                if len(np.array(dim_cluster).shape) != 1:
                    raise ValueError("Generating words must be a list of words.")

                centroid = self.centroid_of_vectors(dim_cluster)
                if normalize_before:
                    centroid = normalize_vector(centroid)
                res = np.dot(test_vec, centroid)
                row.append(res)

            data.append(row)

        cols = ["test_word"] + list(dimensions)

        df = pd.DataFrame(data, columns=cols)
        df = df.sort_values(cols[1:], axis=0, ascending=False)
        return df

    def projections_to_principal_components(self, test, dimensions, n_components=3, n=5):
        """Compute the projection of a test word onto the first n_components principal vectors.

        The n words closest to those principal vectors are printed to get a feeling for what these components mean.

        Args:
            test (str or list[str]): test word like 'land' OR list of test
                                        words like ['land', 'nurse',...]
            dimensions (dict): dictionary of clusters like

                    {'male': ['man', 'he',...]
                     'female': ['him', 'her' ...],
                     ...
                     }
            n_components (int): number of components to consider
            n (int): number of similar words to print

        Returns:
            DataFrame
        """
        if isinstance(test, str):
            test_words = [test]
        else:
            test_words = test

        # collect principal vectors
        principal_vecs = {}
        cols = ["test_word"]
        for dim_name, dim_cluster in dimensions.items():

            if len(np.array(dim_cluster).shape) != 1:
                raise ValueError("Generating words must be a list of words.")

            p_vecs = self.principal_components(dim_cluster, n_components=n_components, normalize=True)
            principal_vecs[dim_name] = p_vecs

            for idx, vec in enumerate(p_vecs):
                cols += ["{}-P{}".format(dim_name, idx + 1)]
                print("{}-P{} is similar to: ".format(dim_name, idx), self.most_similar([vec], n=n))

        data = []
        for test_word in test_words:

            test_vec = self.vector(test_word)
            row = [test_word]
            for dim_name, p_vecs in principal_vecs.items():

                for vec in p_vecs:
                    res = np.dot(test_vec, vec)
                    row.append(res)

            data.append(row)

        df = pd.DataFrame(data, columns=cols)
        df = df.sort_values(cols[1:], axis=0, ascending=False)
        return df

    def cluster_diversity(self, list_of_words, method="centroid_length", **kwargs):
        """Compute a measure of the diversity of a list of words.

        This is done by computing the distribution of similarities between all possible pairs of words in the cluster,
        and comparing it with a uniform distribution.
        """
        if method == "centroid_length":
            centroid = self.centroid_of_vectors(list_of_words, normalize=False)
            return np.dot(centroid, centroid)

        elif method == "mmd":
            kernel = kwargs.get("kernel", None)

        else:
            raise ValueError("Method {} not recognised.".format(method))

    def plot_diversity(self, list_of_words, bandwidth=0.1):
        """Plot density of the mutual similarities of all words. """
        similarities = []
        for word1, word2 in combinations(list_of_words, 2):
            similarities.append(self.similarity(word1, word2))

        sns.kdeplot(np.array(similarities), bw_method=bandwidth)
        plt.xlim(-1, 1)

    def plot_distance_graph(self, list_of_words, nonlinear=False, scaling=2, padding=1.2):
        """Plot a network where edge length shows the similarity between words"""
        if nonlinear:
            covariance_list = [np.tanh(scaling * self.similarity(word1, word2)) for word1, word2 in
                               product(list_of_words, repeat=2)]
        else:
            covariance_list = [self.similarity(word1, word2) for word1, word2 in product(list_of_words, repeat=2)]
        covariance = np.array(covariance_list).reshape(len(list_of_words), len(list_of_words))
        graph = nx.from_numpy_array(covariance)
        mapping = {i: word for i, word in enumerate(list_of_words)}
        graph = nx.relabel_nodes(graph, mapping)
        pos = nx.spring_layout(graph, scale=0.2)
        nx.draw_networkx_nodes(graph, pos, node_size=15, node_color='lightgray')
        nx.draw_networkx_edges(graph, pos, edge_color='lightgray')
        y_off = 0.01
        nx.draw_networkx_labels(graph, pos={k: ([v[0], v[1] + y_off]) for k, v in pos.items()})
        xmax = padding * max(xx for xx, yy in pos.values())
        ymax = padding * max(yy for xx, yy in pos.values())
        xmin = padding * min(xx for xx, yy in pos.values())
        ymin = padding * min(yy for xx, yy in pos.values())
        plt.xlim(xmin, xmax)
        plt.ylim(ymin, ymax)
        plt.box(False)
        plt.tight_layout()

    def plot_distance_matrix(self, list_of_words, cluster=True, figsize=5, nonlinear=False, nl_scaling=2,
                             normalize=False, min=-1):
        """Plot a matrix where each value shows the similarity between words"""
        if nonlinear:
            covariance_list = [np.tanh(nl_scaling * self.similarity(word1, word2))
                               for word1, word2 in product(list_of_words, repeat=2)]
        else:
            covariance_list = [self.similarity(word1, word2) for word1, word2 in product(list_of_words, repeat=2)]

        covariance = np.array(covariance_list).reshape(len(list_of_words), len(list_of_words))

        if cluster:
            # hierarchical clustering
            d = sch.distance.pdist(covariance)
            L = sch.linkage(d, method='complete')
            ind = sch.fcluster(L, 0.5 * d.max(), 'distance')
            new_indices = [i for i in list((np.argsort(ind)))]

            # reorder columns
            temp = covariance[:, new_indices]
            # reorder rows
            covariance = temp[new_indices]

            list_of_words = [list_of_words[i] for i in new_indices]

        plt.figure(figsize=(figsize, figsize))
        if normalize:
            plt.imshow(covariance, aspect='equal', cmap='BrBG', vmin=min(covariance_list), vmax=max(covariance_list))
        else:
            plt.imshow(covariance, aspect='equal', cmap='BrBG', vmin=min, vmax=1)

        plt.yticks(ticks=range(len(list_of_words)), labels=list_of_words)
        plt.xticks(ticks=range(len(list_of_words)), labels=list_of_words, rotation=90)
        plt.colorbar()
        plt.tight_layout()

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
            extra_vecs.extend(self.principal_components(list_of_words))
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

    def score_analogy_test(self, n=10, full_output=False):
        """Compute the scores of different analogy tests from Google Analogy dataset
        http://download.tensorflow.org/data/questions-words.txt.

        Args:
            n (int): number of closest neighbours to search word in
            full_output (bool): whether to print the full output

        """

        file_path = os.path.dirname(__file__)

        tests = [
            'capital-common-countries',
            'capital-world',
            'city-in-state',
            'currency',
            'family',
            'gram1-adjective-to-adverb',
            'gram2-opposite',
            'gram3-comparative',
            'gram4-superlative',
            'gram5-present-participle',
            'gram6-nationality-adjective',
            'gram7-past-tense',
            'gram8-plural',
            'gram9-plural-verbs'
        ]

        tasks = []
        ratio_in_vocab = []
        precisions = []
        for test in tests:

            with open(file_path + '/datasets/' + test + '.txt', 'r') as f:
                analogies = f.readlines()
                analogies = [i.lower().strip('\n').split(' ') for i in analogies]

            results = []
            for analogy in analogies:
                if not all([self.in_vocab(a) for a in analogy]):
                    results.append(np.nan)
                else:
                    res = self.analogy_test(negative_list=[analogy[0]],
                                            positive_list=analogy[1:3],
                                            test_word=analogy[3],
                                            n=n)
                    results.append(res)

            tasks.append(test)
            ratio = 1-results.count(np.nan)/len(results)
            ratio_in_vocab.append(ratio)
            if ratio == 0.0:
                precisions.append(np.nan)
            else:
                results_no_nan = [r for r in results if not np.isnan(r)]
                precisions.append(results_no_nan.count(True)/len(results))

        if full_output:
            df = pd.DataFrame({'task': tasks,
                               'ratio_in_vocab': ratio_in_vocab,
                               'precisions': precisions})
            return df
        else:
            return np.mean([p for p in precisions if not np.isnan(p)])

    def score_similarity_test(self, full_output=False, rescale=True):
        """Compute the scores of the wordsim similarity tests (relatedness and similarity goldstandard).



        Args:
            full_output (bool): If true, return a dataframe with the detailed results;
                else return the least-squares difference between predicted and
                target similarities.
            rescale (bool): Since the embedding's similarities between words are almost all
            in [0, 1] instead of the theoretical [-1, 1] interval, we rescale the target and
            prediction similarities so that the smallest similarity in the dataset is rescaled to -1,
            and the largest to 1. The formula takes each value vi of a list of similarities v to

            .. math::

                v \to \frac{v - min(v)}{max(v) - min(v)} * (1- (-1)) + (-1)

        Return:
            Dataframe or float
        """
        file_path = os.path.dirname(__file__)
        with open(file_path + '/datasets/wordsim_relatedness_goldstandard.txt', 'r') as f:
            relatedness = f.readlines()
            relatedness = [i.lower().strip('\n').split('\t') for i in relatedness]

        with open(file_path + '/datasets/wordsim_similarity_goldstandard.txt', 'r') as f:
            similarity = f.readlines()
            similarity = [i.lower().strip('\n').split('\t') for i in similarity]

        words1 = []
        words2 = []
        targets = []
        predictions = []
        for smpl in relatedness + similarity:
            word1 = smpl[0]
            word2 = smpl[1]

            if self.in_vocab(word1) and self.in_vocab(word2):
                prediction = self.similarity(word1, word2)
            else:
                prediction = np.nan

            words1.append(word1)
            words2.append(word2)
            targets.append((float(smpl[2])-5)/5)
            predictions.append(prediction)

        if rescale:
            min_p = min(predictions)
            max_p = max(predictions)
            min_t = min(targets)
            max_t = max(targets)

            predictions = [(p - min_p)/(max_p - min_p) * (1 - (-1)) + (-1)
                           for p in predictions]

            targets = [(t - min_t)/(max_t - min_t) * (1 - (-1)) + (-1)
                       for t in targets]


        if full_output:
            df = pd.DataFrame({'word1': words1,
                               'word2': words2,
                               'target': targets,
                               'prediction': predictions})
            return df
        else:

            least_sq = 0
            n_not_nan = 0
            for t, p in zip(targets, predictions):
                if not np.isnan(p):
                    least_sq += (p-t)**2
                    n_not_nan += 1

            return least_sq/n_not_nan


class EmbeddingEnsemble:
    """Applies actions to an list_of_embeddings of trained embeddings.
    Args:
        path_to_embeddings (list[str] or str): either list of paths to the embedding files,
            or string that constitutes the same beginning of the path to all embeddings
        load_all_embeddings_to_ram (bool): if False, load the embeddings one-by-one in each method

    """

    def __init__(self, path_to_embeddings, load_all_embeddings_to_ram=True):

        self.list_of_embeddings = []

        # convert input to list of paths to the embeddings
        # in the ensemble
        if isinstance(path_to_embeddings, list):
            # paths are already defined
            self.paths = path_to_embeddings
        else:
            # identify all paths that contain string
            self.paths = glob.glob(path_to_embeddings + '*.emb')
            if len(self.paths) == 0:
                raise EmbeddingError("Failed to find any appropriate file. Please make sure that "
                                     "there are trained embeddings under this path.".format(path_to_embeddings))

        self.size = len(self.paths)

        if load_all_embeddings_to_ram:
            # load all embeddings into a list
            for path in self.paths:
                try:
                    # load the word vectors of an embedding
                    emb = WordEmbedding(path)
                except FileNotFoundError:
                    raise EmbeddingError("Failed to load the trained embeddings {}. Please make sure that "
                                         "the path to this file really exists.".format(path))
                self.list_of_embeddings.append(emb)

    def member(self, idx):
        """Load or retrieve an embedding from the ensemble.
           If embeddings were not loaded to list_of_embeddings at initialization, this function will
           load the embedding from disk.

           idx (int): index of embedding in the list of paths
        """
        if not self.list_of_embeddings:
            try:
                # load the word vectors of an embedding
                return WordEmbedding(self.paths[idx])
            except FileNotFoundError:
                raise EmbeddingError("Failed to load the trained embeddings {}. Please make sure that "
                                     "the path to this file really exists.".format(self.paths[idx]))

        else:
            return self.list_of_embeddings[idx]

    def vocab(self, aggregate=True):
        """Retrieve the vocabulary of this ensemble.

        Args:
            aggregate (bool): whether to aggregate the results to a single answer

        Returns:
            list: if aggregate is true, return a list containing the shared words, else return a list of the
                member's individual vocabulary.
        """
        individual = [self.member(i).vocab() for i in range(self.size)]

        if aggregate:
            # get unique words across all member's vocabs
            shared_vocab = set(individual[0]).intersection(*individual)
            return list(shared_vocab)
        else:
            return individual

    def in_vocab(self, word, aggregate=True):
        """Check whether word is in vocabulary of the ensemble.

         Args:
             word (str): word to check
            aggregate (bool): whether to aggregate the results to a single answer

         Returns:
             bool or list[bool]: if aggregate is true, return whether the word is in all member's vocabularies,
                else return answer for each member

        """
        individual = [self.member(i).in_vocab(word) for i in range(self.size)]
        if aggregate:
            return all(individual)
        else:
            return individual

    def similarity(self, word1, word2, aggregate=True):
        """Return the cosine similarity between 'word1' and 'word2'.

         Args:
             word1 (str): first word
             word2 (str): second word
                aggregate (bool): whether to aggregate the results to a single answer

         Returns:
            float or list[float]: if aggregate is true, return average similarity, else return list of member's
                similarities
        """
        individual = []
        for i in range(self.size):
            emb = self.member(i)
            if not emb.in_vocab(word1):
                print(word1, "not in vocab of embedding ", i, ", will skip this member")
                individual.append(np.nan)
                continue
            if not emb.in_vocab(word2):
                print(word2, "not in vocab of embedding ", i, ", will skip this member")
                individual.append(np.nan)
                continue
            individual.append(emb.similarity(word1, word2))

        if aggregate:
            return np.mean(individual)
        else:
            return individual

    def analogy(self, negative_list, positive_list, n=10, aggregate=True):
        """Compute words closest to the vector resulting from an analogy computation as
           proposed in https://www.aclweb.org/anthology/W14-1618.pdf .

         Args:
            negative_list (list[str]): list of negative words [a is to]
            positive_list (list[str]): list of positive words [b like c to]
            n (int): how many neighbours to compute
            aggregate (bool): whether to aggregate the results to a single answer

         Returns:
             list[list] or list[tuple]: if aggregate is true, return the set of all words returned by the members,
                indicating in how many member's lists they appeared, else return list of member's word lists
        """
        individual = []
        for i in range(self.size):
            emb = self.member(i)
            if not all(emb.in_vocab(word) for word in positive_list + negative_list):
                print("some word(s) not found in vocab of embedding ", i, ", will skip this member")
                individual.append([])
                continue

            analogy_list = emb._word_vectors.most_similar(negative=negative_list, positive=positive_list, topn=n)
            individual.append(analogy_list)

        if aggregate:
            flat_analogy_list = [tpl[0] for member in individual for tpl in member]
            c = Counter(flat_analogy_list)
            return sorted([(k, v) for k, v in c.items()], key=lambda tpl: tpl[1], reverse=True)
        else:
            return individual

    def analogy_test(self, negative_list, positive_list, test_word, n=10, m=10, aggregate=True):
        """Check whether the test word appears in the n closest words to the vector resulting from
           an analogy computation as proposed in https://www.aclweb.org/anthology/W14-1618.pdf .

        Args:
            negative_list (list[str]): list of negative words ["a is to"]
            positive_list (list[str]): list of positive words ["b like c"]
            test_word (str): target ["is to d"]
            n (int): how many neighbours to compute
            m (int): how many neighbours to extract from the cumulative neighbour list if aggregate=True
            aggregate (bool): whether to aggregate the results to a single answer

        Returns:
            bool or list[bool]: If aggregate is false return a list of individual test results. If it is true,
                we check whether the test word is in the first m words of the list produced by
                analogy_test(positive_list, negative_list, n=n).
        """

        if aggregate:
            cumulative_list = self.analogy(negative_list=negative_list, positive_list=positive_list, n=n, aggregate=True)
            # extract words
            cumulative_list = [tpl[0] for tpl in cumulative_list]
            return test_word in cumulative_list[:m]

        else:

            individual = []
            for i in range(self.size):
                emb = self.member(i)
                if not all(emb.in_vocab(word) for word in positive_list + negative_list):
                    print("some word(s) not found in vocab of embedding ", i, ", will skip this member")
                    individual.append(np.nan)
                    continue

                analogy_tuples = emb._word_vectors.most_similar(positive=positive_list, negative=negative_list, topn=n)
                analogy_words = [tpl[0] for tpl in analogy_tuples]
                if test_word in analogy_words:
                    individual.append(True)
                else:
                    individual.append(False)

            return individual

    def most_similar(self, word, n=10, aggregate=True):

        """Return the words most similar to 'word'.

         Args:
             word (str): word to check
             n (int): number of neighbours
             aggregate (bool): whether to aggregate the results to a single answer

         Returns:
             list[str] or list[list[str]]: if aggregate is true, return the set of all words returned by the members,
                indicating in how many member's lists they appeared, else return list of member's word lists
        """
        individual = []
        for i in range(self.size):
            emb = self.member(i)
            if not emb.in_vocab(word):
                print(word, " not found in vocab of embedding ", i, ", will skip this member")
                individual.append([])
                continue

            ms = emb._word_vectors.most_similar(word, topn=n)
            ms = [(word, round(s, 3)) for word, s in ms]
            individual.append(ms)

        if aggregate:
            flat_list = [tpl[0] for member in individual for tpl in member]
            c = Counter(flat_list)
            return sorted([(k, v) for k, v in c.items()], key=lambda tpl: tpl[1], reverse=True)
        else:
            return individual

    def most_similar_by_vectors(self, vector_list, n=10, aggregate=True):
        """Return the words most similar to a vector.

        Args:
            vector (list[array]): vectors to check, one for each member of the ensemble
            n (int): number of neighbours
            aggregate (bool): whether to aggregate the results to a single answer

        Returns:
            list[str] or list[list[str]]: if aggregate is true, return the set of all words returned by the members,
                indicating in how many member's lists they appeared, else return list of member's word lists
        """

        individual = []
        for i in range(self.size):
            emb = self.member(i)
            ms = emb._word_vectors.similar_by_vector(vector_list[i], topn=n)
            ms = [(word, round(s, 3)) for word, s in ms]
            individual.append(ms)

        if aggregate:
            flat_list = [tpl[0] for member in individual for tpl in member]
            c = Counter(flat_list)
            return sorted([(k, v) for k, v in c.items()], key=lambda tpl: tpl[1], reverse=True)
        else:
            return individual

    def score_analogy_test(self, n=10, m=10, full_output=False):
        """Compute the scores of different analogy tests from Google Analogy dataset
        http://download.tensorflow.org/data/questions-words.txt.

        Args:
            n (int): number of closest neighbours to search word in
            full_output (bool): whether to print the full output

        """

        file_path = os.path.dirname(__file__)

        tests = [
            'capital-common-countries',
            'capital-world',
            'city-in-state',
            'currency',
            'family',
            'gram1-adjective-to-adverb',
            'gram2-opposite',
            'gram3-comparative',
            'gram4-superlative',
            'gram5-present-participle',
            'gram6-nationality-adjective',
            'gram7-past-tense',
            'gram8-plural',
            'gram9-plural-verbs'
        ]

        tasks = []
        ratio_in_vocab = []
        precisions = []
        for test in tests:

            with open(file_path + '/datasets/' + test + '.txt', 'r') as f:
                analogies = f.readlines()
                analogies = [i.lower().strip('\n').split(' ') for i in analogies]

            results = []
            for analogy in analogies:
                if not all([self.in_vocab(a) for a in analogy]):
                    results.append(np.nan)
                else:
                    res = self.analogy_test(negative_list=analogy[0:1],
                                            positive_list=analogy[1:3],
                                            test_word=analogy[3],
                                            n=n,
                                            m=m,
                                            aggregate=True)
                    results.append(res)

            tasks.append(test)
            ratio = 1-results.count(np.nan)/len(results)
            ratio_in_vocab.append(ratio)
            if ratio == 0.0:
                precisions.append(np.nan)
            else:
                results_no_nan = [r for r in results if not np.isnan(r)]
                precisions.append(results_no_nan.count(True) / len(results))

        if full_output:
            df = pd.DataFrame({'task': tasks,
                               'ratio_in_vocab': ratio_in_vocab,
                               'precisions': precisions})
            return df
        else:
            return np.mean([p for p in precisions if not np.isnan(p)])

    def score_similarity_test(self, full_output=False):
        """Compute the scores of the wordsim similarity tests (relatedness and similarity goldstandard).
        Args:
            full_output (bool): if true, return a dataframe with the detailed results;
                else return the pearson-r coefficient that measures the correlation between predicted and
                target similarities

        Return:
            Dataframe or float
        """
        file_path = os.path.dirname(__file__)
        with open(file_path + '/datasets/wordsim_relatedness_goldstandard.txt', 'r') as f:
            relatedness = f.readlines()
            relatedness = [i.lower().strip('\n').split('\t') for i in relatedness]

        with open(file_path + '/datasets/wordsim_similarity_goldstandard.txt', 'r') as f:
            similarity = f.readlines()
            similarity = [i.lower().strip('\n').split('\t') for i in similarity]

        words1 = []
        words2 = []
        targets = []
        predictions = []
        for smpl in relatedness + similarity:
            word1 = smpl[0]
            word2 = smpl[1]

            if self.in_vocab(word1) and self.in_vocab(word2):
                prediction = self.similarity(word1, word2, aggregate=True)
            else:
                prediction = np.nan

            words1.append(word1)
            words2.append(word2)
            targets.append((float(smpl[2])-5)/10)
            predictions.append(prediction)

        if full_output:
            df = pd.DataFrame({'word1': words1,
                               'word2': words2,
                               'target': targets,
                               'prediction': predictions})
            return df
        else:
            targets_no_nan = [t for t, p in zip(targets, predictions) if not np.isnan(p)]
            predictions_no_nan = [p for p in predictions if not np.isnan(p)]

            return pearsonr(targets_no_nan, predictions_no_nan)[0]