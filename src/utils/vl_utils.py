import string
import numpy as np
import torch
import torch.nn.functional as F

from tqdm import tqdm
from collections import OrderedDict


# NLP utils
def sent_tokenize(text):
    """ Tokenize text on sentence level.
    Note we consider comma as separator, too.

    Args:
        text: str
    Returns:
        [str_sentence]
    """
    def contains_at_least_one_alpha(text):
        for c in text:
            if str.isalpha(c):
                return True
        return False
    # change punctuations except comma and period to space
    puncts = string.punctuation.replace(",", "").replace(".", "")
    translator = str.maketrans(puncts, " " * len(puncts))
    text = str(text).lower().translate(translator)

    res = []
    for sent in text.replace(",", ".").strip(" .").split("."):
        if contains_at_least_one_alpha(sent):
            res.append(sent)
    return res


def word_tokenize(text):
    """ Tokenize text on word level: converting to lower case, eliminating punctuations.
    Args:
        text: str
    Returns:
        [str_word]
    """
    translator = str.maketrans(string.punctuation, " " * len(string.punctuation))
    tokens = str(text).lower().translate(translator).strip().split()
    return tokens


class GloVe(object):
    """
    Attributes:
        self.glove: {str: tensor}
    """
    def __init__(self, glove_path):
        self.glove_path = glove_path
        self.dim = 300
        self.glove = self._load()
        self.glove["<PAD>"] = torch.zeros(self.dim)
        self.glove["<UNK>"] = torch.randn(self.dim)

    def get(self, word):
        if self.contains(word):
            return self.glove[word]
        else:
            return self.glove["<UNK>"]

    def contains(self, word):
        return word in self.glove.keys()

    def _load(self):
        """ Load GloVe embeddings of this vocabulary.
        """
        glove = dict()
        with open(self.glove_path, 'r') as f:
            for line in tqdm(f.readlines(), desc="Reading GloVe from {}".format(self.glove_path)):
                split_line = line.split()
                word = " ".join(split_line[0: len(split_line) - self.dim])  # some words include space
                embedding = torch.from_numpy(np.array(split_line[-self.dim:], dtype=np.float32))
                glove[word] = embedding

        return glove


class Vocabulary(object):
    """ Natural language vocabulary.
    """
    def __init__(self, *word_set):
        """
        Args:
            *word_set: any number of {str}
        """
        self.special_words = ["<PAD>", "<UNK>"]
        self.wtoi, self.itow = OrderedDict(), OrderedDict()
        self._build(word_set)

    def _build(self, word_set_tuple):
        # 0: <PAD>, 1: <UNK>
        for i, word in enumerate(self.special_words):
            self.wtoi[word] = i
            self.itow[i] = word

        words = set()
        for x in word_set_tuple:
            words.update(x)

        for i, word in enumerate(sorted(words)):
            j = i + len(self.special_words)
            self.wtoi[word] = j
            self.itow[j] = word

    def __len__(self):
        return len(self.wtoi)


def resample(video, target_length):
    """ Make loaded video to target_length by resampling or padding.
    Args:
        video: (T, dim)
        target_length: int
    Returns:
        res: (target_length, dim)
        ori_nframes: int
    """
    ori_nframes = video.shape[0]
    if ori_nframes > target_length:
        idxs = torch.arange(0, target_length + 1, 1.0) / target_length * ori_nframes
        idxs = torch.min(torch.round(idxs).to(torch.long), torch.tensor(ori_nframes - 1))
        starts = idxs[:-1]
        ends = idxs[1:]
        idxs = torch.round((starts + ends) / 2).to(torch.long)
        res = video[idxs]
    elif ori_nframes < target_length:
        res = F.pad(
            video, [0, 0, 0, target_length - ori_nframes]
        )
    else:
        res = video

    return res, ori_nframes
