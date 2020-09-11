# This file contains a wrapper class for word2vec models training word trained_embeddings
import os
import numpy as np
from gensim.models import Word2Vec
import torch
torch.manual_seed(0)
from transformers import BertTokenizer, BertModel, BertConfig


class TrainableModel:
    """Train a word embedding using a Word2Vec model."""

    def __init__(self):
        self.ensemble = []
        self.training_data = []
        self.log = ""

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

    def train(self, path_training_data, path_description, hyperparameters, n_models=None, share_of_original_data=1.,
              seed=None):
        """Trains a single embedding or an ensemble of embeddings."""

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
            member = self._make_embedding(self.training_data, hyperparameters)
            self.ensemble.append(member)
        else:
            # make a bootstrapped list_of_embeddings
            for m in range(n_models):
                print("Training model ", m+1)
                n_samples = int(share_of_original_data * len(training_data))
                # make bootstrapped sample of training data
                bootstrapped_train_data = list(np.random.choice(training_data, size=n_samples, replace=True))
                # train the model
                member = self._make_embedding(bootstrapped_train_data, hyperparameters)
                self.ensemble.append(member)
            self.log += "Bootstrapped members of the list_of_embeddings used {} subsamples (with replacement) of {} " \
                        "documents in the training data\n.".format(n_samples, len(training_data))

        self.log += "The model generating the embedding was trained with the following hyperparameters: \n {}\n".format(hyperparameters)

    def _make_embedding(self, train_data, hyperparameters):
        return NotImplemented


class Word2VecModel(TrainableModel):
    """Train a word embedding using a Word2Vec model."""

    def _make_embedding(self, train_data, hyperparameters):
        """Train a Word2Vec embedding."""
        return Word2Vec(self.training_data, **hyperparameters)


class BertModel(TrainableModel):
    """Train a word embedding using a BERT model."""

    def _make_embedding(self, train_data, hyperparameters):
        """Train a BERT embedding."""

        config = BertConfig()
        model = BertModel(config)
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

        # Tokenize training data
        for sent in sentences:
            encoded_dict = tokenizer.encode_plus(
                sent,
                add_special_tokens=True,
                truncation=True,
                max_length=48,
                pad_to_max_length=True,
                return_tensors='pt',
            )
            # Save tokens from sentence as a separate array.
            marked_text = "[CLS] " + sent + " [SEP]"
            tokenized_texts.append(tokenizer.tokenize(marked_text))

            # Add the encoded sentence to the list.
            input_ids.append(encoded_dict['input_ids'])

            
        return NotImplemented