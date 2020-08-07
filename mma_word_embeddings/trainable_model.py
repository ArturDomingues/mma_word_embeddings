# This file contains a wrapper class for word2vec models training word trained_embeddings
import os
import numpy as np


class TrainableModel:
    """Representation of a word embedding, which is a map from word strings to vectors."""

    def __init__(self, model):
        self.model = model
        self.ensemble = []
        self.training_data = []
        self.log = ""

    def train(self, path_training_data, path_description, hyperparameters, n_models=None, share_of_original_data=1.,
              seed=None):
        """Trains a single word2vec embedding."""

        training_data = []

        with open(path_training_data, "r") as f:
            for line in f:
                stripped_line = line.strip()
                line_list = stripped_line.split()
                training_data.append(line_list)
        self.training_data = training_data

        # Save the description in the log,
        # so we remember how the training data was produced
        with open(path_description, "r") as f:
            description = f.read()
        self.log += "The following training data was used:\n{}\n".format(description)

        if seed is not None:
            np.random.seed(seed)

        if n_models is None:
            # only save a single model on the full data
            member = self.model(self.training_data, **hyperparameters)
            self.ensemble.append(member)
        else:
            # make a bootstrapped list_of_embeddings
            for m in range(n_models):
                print("Training model ", m+1)
                n_samples = int(share_of_original_data * len(training_data))
                # make bootstrapped sample of training data
                train_data = list(np.random.choice(training_data, size=n_samples, replace=True))
                # train the model
                member = self.model(train_data, **hyperparameters)
                self.ensemble.append(member)
            self.log += "Bootstrapped members of the list_of_embeddings used {} subsamples (with replacement) of {} " \
                        "documents in the training data\n.".format(n_samples, len(training_data))

        self.log += "The model generating the embedding was trained with the following hyperparameters: \n {}\n".format(hyperparameters)

    def save_embedding(self, output_path):
        """Saves the embedding of this model as well as a metadata file in output_path"""

        if len(self.ensemble) == 1:
            save_as = output_path + '.emb'
            if os.path.isfile(save_as):
                raise ValueError(
                    "Embedding {} already exists. Choose a different name or delete existing model."
                        .format(save_as))

            self.ensemble[0].wv.init_sims(replace=True)  # Normalise the vectors
            self.ensemble[0].wv.save(save_as)
        else:
            for id, member in enumerate(self.ensemble):
                save_as = output_path + "-" + str(id) + ".emb"
                if os.path.isfile(save_as):
                    raise ValueError(
                        "Embedding {} already exists. Choose a different name or delete existing model."
                            .format(save_as))
                member.wv.init_sims(replace=True)  # Normalise the vectors
                member.wv.save(save_as)

        with open(output_path + "-training-data.txt", "w") as f:
            for document in self.training_data:
                sentence = " ".join(word for word in document)
                f.write('%s\n' % sentence)
        with open(output_path + "-description.txt", "w") as f:
            f.write(self.log)
