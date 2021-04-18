# This file contains a wrapper class for word2vec models training word trained_embeddings
import os
import numpy as np
from gensim.models import Word2Vec
from random import seed, shuffle


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


def train_word2vec_model(
        path_training_data,
        output_path,
        hyperparameters={},
        n_models=1,
        share_of_original_data=1.,
        random_buffer_size=1000,
        path_pretraining_data=None,
        len_training_data=None,
        path_description=None,
        data_seed=None,
):
    """Trains a single embedding or an ensemble of embeddings.

    Args:
        output_path (str): where to save the model and description file; does not include an ending (.emb will
            be automatically added)
        hyperparameters (dict): dictionary of hyperparameters that are directly fed into Word2Vec model
        n_models (int): number of models to train
        share_of_original_data (float): each line loaded from the data file is discarded
            with this ratio; use 1. to use all data
        random_buffer_size (int): Keep so many lines from the data file in a buffer from which
            the samples are returned at random. Higher values take more memory but lead to more randomness
            when sampling the data. A value equal to the number of all samples would lead to perfectly
            random samples.
        path_pretraining_data (str): if model should get pre-trained, specify this path to the pretraining data set
        len_training_data (int): pretraining requires this estimate of the length of the training data
        data_seed (int): random seed set for sampling
    """
    # fix the seed of data sampling
    if data_seed is not None:
        seed(data_seed)

    # try to infer path for description file
    if path_description is None:
        path_description = path_training_data[:-18] + "-description.txt"
    if not os.path.exists(path_description):
        raise ValueError(f"Description file {path_description} not found.")
    if not os.path.exists(output_path):
        raise ValueError(f"Output path {output_path} does not exist.")

    for m in range(n_models):

        print("Training model ", m + 1)
        training_generator = DataGenerator(path_training_data,
                                           share_of_original_data,
                                           random_buffer_size)

        if path_pretraining_data is None:
            # do not pretrain
            model = Word2Vec(sentences=training_generator, **hyperparameters)

        else:
            pretraining_generator = DataGenerator(path_pretraining_data, 1.0, random_buffer_size)
            model = Word2Vec(sentences=pretraining_generator, **hyperparameters)
            model.train(sentences=training_generator, total_examples=len_training_data, epochs=model.epochs)

        # normalise the word vectors
        model.wv.init_sims(replace=True)

        emb = model.wv

        # save the current embedding
        path = output_path + "-" + str(m) + ".emb"
        if os.path.isfile(path):
            path = output_path + "-" + str(m) + "-alt.emb"
            raise ValueError(
                "Path for embedding {} already exists. Renamed path to {}.".format(
                    m, path))
        emb.save(path)

    # update description
    log = ""
    with open(path_description, "r") as f:
        description = f.read()
    log += "The following training data was used:\n{}\n".format(description)
    log += "Used {}% of original data.\n".format(100 * share_of_original_data)
    log += "Used a random buffer size of {} lines.\n".format(random_buffer_size)
    log += "The model generating the embedding was trained with the following " \
           "hyperparameters: \n {}\n".format(hyperparameters)

    with open(path_description, "w") as f:
        f.write(log)
