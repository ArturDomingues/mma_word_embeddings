# This file contains a wrapper class for word2vec models training word trained_embeddings
import os
import numpy as np
from gensim.models import Word2Vec


def data_generator(path):
    """generator over data"""
    with open(path, "r") as f:
        for line in f:
            yield line.strip().split(" ")


def train_word2vec_model(
        path_training_data,
        output_path,
        hyperparameters={},
        path_pretraining_data=None,
        path_description=None,
        n_models=1,
        share_of_original_data=None,
        sample_with_replacement=False,
        seed=None,
):
    """Trains a single embedding or an ensemble of embeddings.

    Args:
        output_path (str): where to save the model and description file; does not include an ending (.emb will
            be automatically added)
        hyperparameters (dict): dictionary of hyperparameters that are directly fed into Word2Vec model
        n_models (int): number of models to train
        share_of_original_data (float or None): if float, this is interpreted as the ratio of sentences sampled
            for training versus the number of sentences in the training data set; if None then the training
            data is not subsampled at all
        sample_with_replacement (bool): if True (and if share_of_original_data is not None),
            sample sentences with replacement
        seed (int): random seed set for sampling
    """

    if seed is not None:
        np.random.seed(seed)
    if path_description is None:
        path_description = path_training_data[:-18] + "-description.txt"

    for m in range(n_models):

        print("Training model ", m + 1)
        training_generator = data_generator(path_training_data, share_of_original_data, sample_with_replacement)

        if path_pretraining_data is None:
            # do not pretrain
            model = Word2Vec(training_generator, **hyperparameters)

        else:
            pretraining_generator = data_generator(path_pretraining_data,
                                                   share_of_original_data=None,
                                                   sample_with_replacement=None)
            model = Word2Vec(pretraining_generator, **hyperparameters)

            #TODO: how will the len work with big files?
            model.train(training_generator, total_examples=len(training_generator), epochs=model.epochs)

        # normalise the word vectors
        model.wv.init_sims(replace=True)

        emb = model.wv

        # save the embedding
        path = output_path + "-" + str(m) + ".emb"
        if os.path.isfile(path):
            raise ValueError(
                "Embedding {} already exists. Choose a different name or delete existing model.".format(
                    path))
        emb.save(path)

    # update description
    log = ""
    with open(path_description, "r") as f:
        description = f.read()
    log += "The following training data was used:\n{}\n".format(description)
    if n_models is not None:
        log += "Bootstrapped ensemble used {}% of the original documents (subsampled with replacement)" \
               "to train each embedding.\n".format(100 * share_of_original_data)
    log += "The model generating the embedding was trained with the following " \
           "hyperparameters: \n {}\n".format(hyperparameters)

    with open(path_description, "w") as f:
        f.write(log)
