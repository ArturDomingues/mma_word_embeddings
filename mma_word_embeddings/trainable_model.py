# This file contains a wrapper class for word2vec models training word trained_embeddings
import os
import numpy as np
from gensim.models import Word2Vec


def data_generator(path):
    """generator over data"""
    with open(path, "r") as f:
        for line in f:
            yield line.strip().split()


class Word2VecModel:
    """Train a word embedding using a Word2Vec model."""

    def __init__(self, path_training_data, path_description, path_pretraining_data=None):

        self.path_training_data = path_training_data
        self.path_pretraining_data = path_pretraining_data
        self.path_description = path_description

    def train(self,
              output_path,
              hyperparameters,
              n_models=None,
              share_of_original_data=1.,
              sample_with_replacement=False,
              seed=None,
              ):
        """Trains a single embedding or an ensemble of embeddings.

        Args:
            output_path (str): where to save the model and description file; does not include an ending (.emb will
                be automatically added)
            hyperparameters (dict): dictionary of hyperparameters that are directly fed into Word2Vec model
            n_models (int or None): number of models to train; if None only one model is trained, else we train an
                ensemble
            share_of_original_data (float or None): if float, this is interpreted as the ratio of sentences sampled
                for training versus the number of sentences in the training data set; if None then the training
                data is not sampled at all
            sample_with_replacement (bool): if True (and if share_of_original_data is not None),
                sample sentences with replacement
            seed (int): random seed set for sampling
        """

        if seed is not None:
            np.random.seed(seed)

        # Train embedding(s)
        if n_models is None:
            # train embedding on full data
            emb = self.make_embedding(hyperparameters, bootstrap=False)

            # save embedding
            output_path = output_path + ".emb"
            if os.path.isfile(output_path):
                raise ValueError(
                    "Embedding {} already exists. Choose a different name or delete existing model.".format(
                        output_path))
            emb.save(output_path)

        else:
            # save multiple models trained on bootstrapped/subsampled data
            for m in range(n_models):

                print("Training model ", m+1)

                # train the embedding
                emb = self.make_embedding(hyperparameters, bootstrap=True)

                # save the embedding
                path = output_path + "-" + str(m) + ".emb"
                if os.path.isfile(path):
                    raise ValueError(
                        "Embedding {} already exists. Choose a different name or delete existing model.".format(
                            path))
                emb.save(path)

        # update description
        log = ""
        with open(self.path_description, "r") as f:
            description = f.read()
        log += "The following training data was used:\n{}\n".format(description)
        if n_models is not None:
            log += "Bootstrapped ensemble used {}% of the original documents (subsampled with replacement)" \
                   "to train each embedding.\n".format(100 * share_of_original_data, len(self.training_data))
        log += "The model generating the embedding was trained with the following " \
               "hyperparameters: \n {}\n".format(hyperparameters)
        with open(output_path + "-description.txt", "w") as f:
            f.write(log)

    def make_embedding(self, hyperparameters, bootstrap):

        if bootstrap:
            training_generator = bootstrap_data_generator(self.path_training_data)
        else:
            training_generator = data_generator(self.path_training_data)

        if self.pretraining_data_generator is None:
            model = Word2Vec(training_generator, **hyperparameters)

        else:
            pretraining_generator = data_generator(self.path_pretraining_data)
            model = Word2Vec(pretraining_generator, **hyperparameters)
            model.train(training_generator, total_examples=len(training_generator), epochs=model.epochs)

        # normalise the word vectors
        model.wv.init_sims(replace=True)

        # extract a keyed_vectors object
        return model.wv
