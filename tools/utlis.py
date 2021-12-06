import torch
import torch.nn as nn
from torch.autograd import Variable
import collections
import numpy as np


class strLabelConverter(object):
    """Convert between str and label.

    NOTE:
        Insert `blank` to the alphabet for CTC.

    Args:
        alphabet (str): set of the possible characters.
        ignore_case (bool, default=True): whether or not to ignore all of the case.
    """
    def __init__(self, alphabet, ignore_case=True):
        self._ignore_case = ignore_case
        if self._ignore_case:
            alphabet = alphabet.lower()
        self.alphabet = alphabet + '-'  # for `-1` index

        self.dict = {}
        for i, char in enumerate(alphabet):
            # NOTE: 0 is reserved for 'blank' required by wrap_ctc
            self.dict[char] = i + 1

    def encode(self, text):
        """Support batch or single str.

        Args:
            text (str or list of str): texts to convert.

        """
        if isinstance(text, str):
            text = [
                self.dict[char.lower() if self._ignore_case else char]
                for char in text
            ]
            length = [len(text)]

            return torch.Tensor(text), torch.Tensor(length)
        elif isinstance(text, collections.Iterable):
            temp = []
            length = []
            for s in text:
                st, le = self.encode(s)
                temp.append(st)
                length.append(float(le.item()))

            return nn.utils.rnn.pad_sequence([
                torch.Tensor(iter) for iter in temp
            ]).transpose(0, 1), torch.tensor(length)

    def decode(self, t, length, raw=False):
        """Decode encoded texts back into strs.

        Args:
            torch.IntTensor [length_0 , length_1 , ... length_{n - 1}]: encoded texts.
            torch.IntTensor [n]: length of each text.

        Raises:
            AssertionError: when the texts and its length does not match.

        Returns:
            text (str or list of str): texts to convert.
        """
        t.cpu()
        length.cpu()
        if length.numel() == 1:
            length = length.item()

            if raw:
                return ''.join([self.alphabet[i - 1] for i in t])
            else:
                char_list = []
                for i in range(length):
                    if t[i] != 0 and (not (i > 0 and t[i - 1] == t[i])):
                        char_list.append(self.alphabet[t[i] - 1])
                return ''.join(char_list)
        else:
            # batch mode
            texts = []
            for i in range(length.numel()):
                texts.append(
                    self.decode(t[i, :], length[i].clone().detach(), raw=raw))
            return texts