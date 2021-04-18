# This file contains a wrapper class for word2vec models training word trained_embeddings
import os
import numpy as np
from gensim.models import Word2Vec
from random import shuffle


class DataGenerator(object):
    def __init__(self, path_to_data,
                 share_of_original_data=1.,
                 random_buffer_size=1000):
        """Iterator that loads a lines from a file.
        Args:
            path_to_data (str): Full path to a data file with one preprocessed sentence/document per line.
            share_of_original_data (float):  and picks each line with probability share_of_original_data, which
                effectively results in a dataset with approx n_data*share_of_original_data samples
            random_buffer_size (int): keeps multiple files in a list from which the
                next element is randomly chosen, and replaced by the next line from the
                data file
        """
        self.path_to_data = path_to_data
        self.share_of_original_data = share_of_original_data
        self.random_buffer_size = random_buffer_size

    def __iter__(self):

        # load initial buffer
        buffer = []
        with open(self.path_to_data, "r") as f:
            for i in range(self.random_buffer_size):
                line = f.readline().strip().split(" ")
                buffer.append(line)

            # continue with line random_buffer_size+1
            for line in f:

                shuffle(buffer)
                # remove first element from shuffled list
                pick = buffer.pop(0)

                # fill buffer with new element
                buffer.append(line.strip().split(" "))

                # randomly drop the element and move to the next
                if np.random.rand() > self.share_of_original_data:
                    continue

                # else return the picked one
                yield pick

            # if end of file has been reached
            # yield all elements left in the buffer
            for el in buffer:
                yield el


def data_generator(path, share_of_original_data=None):
    """iterator over the lines of the data

    Args:
        path (str): path to data file, one sentence/document per line

    """

    with open(path, "r") as f:
        for line in f:
            if share_of_original_data is None:
                yield line.strip().split(" ")

            else:
                # only return line if in some cases
                if np.random.rand() > share_of_original_data:
                    continue
                else:
                    yield line.strip().split(" ")


def train_word2vec_model(
        path_training_data,
        output_path,
        hyperparameters={},
        n_models=1,
        share_of_original_data=None,
        sample_with_replacement=False,
        path_pretraining_data=None,
        len_training_data=None,
        path_description=None,
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
        path_pretraining_data (str): if model should get pre-trained, specify this path to the pretraining data set
        len_training_data (int): pretraining requires this estimate of the length of the training data
        seed (int): random seed set for sampling
    """

    if seed is not None:
        np.random.seed(seed)
    if path_description is None:
        path_description = path_training_data[:-18] + "-description.txt"

    if not os.path.exists(output_path):
        raise ValueError(f"Description file {output_path} not found.")

    for m in range(n_models):

        print("Training model ", m + 1)
        training_generator = data_generator(path_training_data, share_of_original_data)

        if path_pretraining_data is None:
            # do not pretrain
            model = Word2Vec(corpus_file=training_generator, **hyperparameters)

        else:
            pretraining_generator = data_generator(path_pretraining_data)
            model = Word2Vec(corpus_file=pretraining_generator, **hyperparameters)
            model.train(corpus_file=training_generator, total_examples=len_training_data, epochs=model.epochs)

        # normalise the word vectors
        model.wv.init_sims(replace=True)

        emb = model.wv

        # save the current embedding
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
