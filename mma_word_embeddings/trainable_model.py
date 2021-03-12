# This file contains a wrapper class for word2vec models training word trained_embeddings
import os
import numpy as np
from gensim.models import Word2Vec


class TrainableModel:
    """Train a word embedding using a Word2Vec model."""

    def __init__(self, path_training_data, path_description, path_pretraining_data=None):

        # load training data
        self.training_data = self._load_data(path_training_data)

        if path_pretraining_data is not None:
            self.pre_training_data = self._load_data(path_pretraining_data)
        else:
            self.pre_training_data = None

        self.log = ""
        # Save the description in the log,
        # so we remember how the training data was produced
        with open(path_description, "r") as f:
            description = f.read()
        self.log += "The following training data was used:\n{}\n".format(description)

    def train(self,
              output_path,
              hyperparameters,
              n_models=None,
              share_of_original_data=1.,
              seed=None,
              ):
        """Trains a single embedding or an ensemble of embeddings."""

        if seed is not None:
            np.random.seed(seed)

        # update description already here, in case training crashes
        if n_models is not None:
            self.log += "Bootstrapped ensemble used {}% of the original documents (subsampled with replacement)" \
                        "to train each embedding.\n".format(100*share_of_original_data, len(self.training_data))
        self.log += "The model generating the embedding was trained with the following " \
                    "hyperparameters: \n {}\n".format(hyperparameters)
        with open(output_path + "-description.txt", "w") as f:
            f.write(self.log)

        # Train embeddings

        if n_models is None:
            # train embedding on full data
            emb = self.make_embedding(self.training_data, hyperparameters, self.pre_training_data)

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

                # make bootstrapped training data
                n_documents = int(share_of_original_data * len(self.training_data))
                bootstrapped_train_data = list(np.random.choice(self.training_data, size=n_documents, replace=True))

                # train the embedding
                emb = self.make_embedding(bootstrapped_train_data, hyperparameters)

                # save the embedding
                path = output_path + "-" + str(m) + ".emb"
                if os.path.isfile(path):
                    raise ValueError(
                        "Embedding {} already exists. Choose a different name or delete existing model.".format(
                            path))
                emb.save(path)

    def make_embedding(self, train_data, pre_train_data, hyperparameters):
        return NotImplemented

    def _load_data(self, path):
        return NotImplemented


class Word2VecModel(TrainableModel):
    """Train a word embedding using a Word2Vec model."""

    def __init__(self, path_training_data, path_description, path_pretraining_data=None):
        super().__init__(path_training_data, path_description, path_pretraining_data=path_pretraining_data)

    def _load_data(self, path):
        data = []
        with open(path, "r") as f:
            for line in f:
                stripped_line = line.strip()
                data.append(stripped_line.split())
        return data

    def make_embedding(self, train_data, pre_train_data, hyperparameters):
        """Train a Word2Vec model and extract the embedding."""

        if pre_train_data is None:
            model = Word2Vec(train_data, **hyperparameters)

        else:
            model = Word2Vec(pre_train_data, **hyperparameters)
            model.train(train_data, total_examples=len(train_data), epochs=model.epochs)

        # normalise the word vectors
        model.wv.init_sims(replace=True)
        # extract a keyed_vectors object
        emb = model.wv

        return emb

#
# class BertModel(TrainableModel):
#     """Train a word embedding using a BERT model."""
#
#     def __init__(self, path_training_data, path_description):
#         super().__init__(path_training_data, path_description)
#         self.MAX_LEN = 50
#
#     def _prepare_data(self):
#         tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
#         #max_document_length = np.max([len(document.split(' ')) for document in self.training_data])
#
#         input_ids = []
#         tokenized_texts = []
#
#         for doc in self.training_data:
#             encoded_dict = tokenizer.encode_plus(
#                 doc,  # Sentence to encode.
#                 add_special_tokens=True,  # Add '[CLS]' and '[SEP]'
#                 truncation=True,
#                 max_length=self.MAX_LEN,  # Pad & truncate all sentences.
#                 pad_to_max_length=True,
#                 return_tensors='pt',  # Return pytorch tensors.
#             )
#
#             # Save tokens from sentence as a separate array. We will use it later to explore and compare embeddings.
#             marked_text = "[CLS] " + doc + " [SEP]"
#             tokenized_texts.append(marked_text.split(' '))
#
#             # Add the encoded sentence to the list.
#             input_ids.append(encoded_dict['input_ids'])
#
#         # Convert the list into tensor.
#         input_ids = torch.cat(input_ids, dim=0)
#
#         return input_ids, tokenized_texts
#
#     def make_embedding(self, hyperparameters):
#         """Train a BERT embedding."""
#
#         input_ids, tokenized_texts = self._prepare_data()
#         segments_ids = torch.ones_like(input_ids)
#         NUM_DOCS = len(self.training_data)
#
#         model = BertModel.from_pretrained('bert-base-uncased',
#                                           output_hidden_states=True,  # Whether the model returns all hidden-states.
#                                           )
#         model.eval()
#
#         with torch.no_grad():
#             outputs = model(input_ids, segments_ids)
#
#             # Evaluating the model will return a different number of objects based on
#             # how it's  configured in the `from_pretrained` call earlier. In this case,
#             # becase we set `output_hidden_states = True`, the third item will be the
#             # hidden states from all layers. See the documentation for more details:
#             # https://huggingface.co/transformers/model_doc/bert.html#bertmodel
#             hidden_states = outputs[2]
#
#         token_embeddings = torch.stack(hidden_states, dim=0)
#         token_embeddings = token_embeddings.permute(1, 2, 0, 3)
#         processed_embeddings = token_embeddings[:, :, 9:, :]
#         embeddings = torch.reshape(processed_embeddings, (NUM_DOCS, self.MAX_LEN, -1))
#
#         # extract dictionary of words and vecs
#         bert_dict = {}
#         # for i in tokenized_texts:
#         for i in range(len(tokenized_texts)):
#             for j in range(len(embeddings)):
#                 if i == j:
#                     for k in range(len(tokenized_texts[i])):
#                         for l in range(len(embeddings[j])):
#                             if k == l:
#                                 bert_dict['{}'.format(tokenized_texts[i][k])] = embeddings[j][l].numpy()
#
#         # turn dict to gensim embedding
#         emb = WordEmbeddingsKeyedVectors(vector_size=len(list(bert_dict.values())[0]))
#         entities = list(bert_dict.keys())
#         weights = list(bert_dict.values())
#         emb.add(entities, weights)
#
#         return emb
