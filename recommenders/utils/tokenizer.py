import json


class Tokenizer:
    def __init__(self, vocabulary, unknown=False, padding=True):
        """
        Init Tokenizer. Vocabulary contains all unique elements
        as a list.
        """
        # Stoi - String to index
        self.stoi_ = {string: index for index, string in enumerate(vocabulary)}

        # Add <pad> and <unk> to stoi
        # Add <unk> first so <pad> is last one
        self.unknown = False
        if unknown:
            self.unknown = True
            self.unk_token = "<unk>"
            self.unk_idx = len(self.stoi_)
            self.stoi_[self.unk_token] = len(self.stoi_)

        if padding:
            self.pad_token = "<pad>"
            self.pad_idx = len(self.stoi_)
            self.stoi_[self.pad_token] = len(self.stoi_)

        # Itos - index to string
        self.itos_ = [
            string for string, index in sorted(self.stoi_.items(), key=lambda x: x[1])
        ]

    def stoi(self, string):
        """
        Returns index of passed string.
        """
        if not self.unknown:
            return self.stoi_[string]
        else:
            idx = (
                self.stoi_[string]
                if string in self.stoi_.keys()
                else self.stoi_[self.unk_token]
            )
            return idx

    def itos(self, index):
        """
        Returns string of passed index.
        """
        return self.itos_[index]

    def extend(self, new_vocabulary):
        """
        Extend current tokenizer with new vocabulary.
        """
        curr_len = len(self.stoi_)
        new_stoi = {
            string: (index + curr_len - 1)
            for index, string in enumerate(new_vocabulary)
        }
        new_itos = [
            string for string, index in sorted(new_stoi.items(), key=lambda x: x[1])
        ]

        self.stoi_ = {**self.stoi_, **new_stoi}
        self.itos_.extend(new_itos)

    def __len__(self):
        return len(self.itos_)

    def save_to_file(self, file_path):
        """
        Save tokenizer to a JSON file.
        """
        tokenizer_dict = {
            "stoi": self.stoi_,
            "itos": self.itos_,
        }
        with open(file_path, "w") as f:
            json.dump(tokenizer_dict, f)

    @classmethod
    def from_file(cls, file_path):
        """
        Load tokenizer from a JSON file.
        """
        with open(file_path, "r") as f:
            tokenizer_dict = json.load(f)

        tokenizer = cls([])
        tokenizer.stoi_ = tokenizer_dict["stoi"]
        tokenizer.itos_ = tokenizer_dict["itos"]

        if "<pad>" in tokenizer.stoi_.keys():
            tokenizer.padding = True
            tokenizer.pad_token = "<pad>"
            tokenizer.pad_idx = tokenizer.stoi_[tokenizer.pad_token]

        if "<unk>" in tokenizer.stoi_.keys():
            tokenizer.unknown = True
            tokenizer.unk_token = "<unk>"
            tokenizer.unk_idx = tokenizer.stoi_[tokenizer.unk_token]
        else:
            tokenizer.unknown = False

        return tokenizer

    @classmethod
    def from_dict(cls, tokenizer_dict):
        """
        Load tokenizer from a JSON file.
        """

        tokenizer = cls([])
        tokenizer.stoi_ = tokenizer_dict["stoi"]
        tokenizer.itos_ = tokenizer_dict["itos"]

        if "<pad>" in tokenizer.stoi_.keys():
            tokenizer.padding = True
            tokenizer.pad_token = "<pad>"
            tokenizer.pad_idx = tokenizer.stoi_[tokenizer.pad_token]

        if "<unk>" in tokenizer.stoi_.keys():
            tokenizer.unknown = True
            tokenizer.unk_token = "<unk>"
            tokenizer.unk_idx = tokenizer.stoi_[tokenizer.unk_token]
        else:
            tokenizer.unknown = False

        return tokenizer
