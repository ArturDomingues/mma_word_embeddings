# Helper functions for working with word trained_embeddings
import numpy as np
import matplotlib.colors as mcolors
COLORMAP = mcolors.LinearSegmentedColormap.from_list("MyCmapName",["r", "w", "g"])


def make_pairs(left_list, right_list, exclude_doubles=False):
    """Takes two lists of words and returns all pairs that can be formed between them."""

    pairs = [[left_word, right_word] for left_word in left_list for right_word in right_list]

    if exclude_doubles:
        pairs = [[word1, word2] for word1, word2 in pairs if word1 != word2]

    return pairs


def normalize(vector):
    """Normalize a vector and convert to numpy array."""

    vector = np.array(vector)
    norm = np.linalg.norm(vector)
    if norm == 0:
        raise ValueError("vector is zero, cannot normalize!")
    else:
        return vector / norm


def cell_colour(s, columns=None):
    """Can be used to colour cells in dataframe: df.style.apply(cell_colour)"""
    if columns is not None:
        if s.name in columns:
          cmap = COLORMAP
          norm = mcolors.DivergingNorm(vmin=-1, vcenter=0, vmax=1)
          return ['background-color: {:s}'.format(mcolors.to_hex(c.flatten())) for c in cmap(norm(s.values))]
        else:
            if all(isinstance(v, float) for v in s.values):
              cmap = COLORMAP
              norm = mcolors.DivergingNorm(vmin=-1, vcenter=0, vmax=1)
              return ['background-color: {:s}'.format(mcolors.to_hex(c.flatten())) for c in cmap(norm(s.values))]
    else:
        return [''] * len(s)


def remove_if_not_in_vocab(vocab, list_of_words):
    """Return new list that only contains words found in the vocab"""
    not_in_vocab = [word for word in list_of_words if word not in vocab]
    cleaned_list = [word for word in list_of_words if word not in not_in_vocab]
    print("Removed the following words:", not_in_vocab)
    return cleaned_list