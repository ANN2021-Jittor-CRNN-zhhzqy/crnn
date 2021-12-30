import jittor as jt
import Levenshtein


class strLabelConverter(object):
    """
    Convert between str and label.

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
        self.alphabet = alphabet  # for `-1` index

        self.dict = {}
        for i, char in enumerate(alphabet):
            # NOTE: 0 is reserved for 'blank' required by wrap_ctc
            self.dict[char] = i + 1  # !!

    def encode(self, texts):
        """
        Only Support Batch of str!

        Args:
            text (list of str): texts to convert.

        """
        sts = []
        length = [len(s) for s in texts]
        max_len = max(length)
        for s in texts:
            st = [self.dict[char.lower() if self._ignore_case else char] for char in s]
            st = jt.nn.pad(jt.array(st), (0, max_len - len(st)), "constant", 0)
            sts.append(st)
        return jt.stack(sts).clone(), jt.array(length).clone()
        """
        length = [len(s) for s in texts]
        s = ''.join(texts)
        sts = [self.dict[char.lower() if self._ignore_case else char] for char in s]
        print("sts ", sts)
        print("length ", length)
        return jt.array(sts), jt.array(length)
        """


    def decode(self, encoded_texts):
        """
        Decode encoded_texts back into strs.

        Args:
            jt.array, shape(batch_size, seq, num_class): probs.

        Returns:
            list of str, shape(batch_size): texts to convert.
        """
        texts = []
        for st in encoded_texts:
            last_char_i = 0  # !!
            char_list = []
            for char_i in st.tolist():
                if char_i != 0 and char_i != last_char_i:  # !!
                    char_list.append(self.alphabet[char_i - 1])  # !!
                last_char_i = char_i
            texts.append(''.join(char_list))
        return texts


class BKNode(object):
    def __init__(self, w):
        self.w = w
        self.children = {}

    def insert(self, w):
        dis = Levenshtein.distance(self.w, w)
        if dis in self.children:
            self.children[dis].insert(w)
        else:
            self.children[dis] = BKNode(w)
    
    def find(self, w, threshold, ret_list):
        dis = Levenshtein.distance(self.w, w)
        if dis <= threshold:
            ret_list.append(self.w)
        
        for dis_c, c in self.children.items():
            if dis - threshold <= dis_c <= dis + threshold:
                c.find(w, threshold, ret_list)


class BKTree(object):
    def __init__(self, lex_dict):
        root_w = lex_dict.pop(0)
        self.root = BKNode(root_w)
        for w in lex_dict:
            self.root.insert(w)

    def find(self, s, threshold):
        ret_list = []
        self.root.find(s, threshold, ret_list)
        return ret_list

