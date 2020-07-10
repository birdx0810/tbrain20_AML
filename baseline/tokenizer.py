# -*- coding: UTF-8 -*-
import re
import unicodedata

from tqdm import tqdm

class Tokenizer(object):
    def __init__(self, embeddings=None):
        super(Tokenizer).__init__()

        self.pad_token = "[PAD]"
        self.unk_token = "[UNK]"
        self.bos_token = "[BOS]"
        self.eos_token = "[EOS]"

        self.pad_token_idx = 0
        self.unk_token_idx = 1
        self.bos_token_idx = 2
        self.eos_token_idx = 3

        self.vocab = []
        self.vectors = []

        self.vocab.append(self.pad_token)
        self.vocab.append(self.unk_token)
        self.vocab.append(self.bos_token)
        self.vocab.append(self.eos_token)

        if embeddings is not None:
            embedding_dim = len(embeddings[0])
            self.build_vocabulary(embeddings=embeddings)

            self.vectors.append([0.0 for _ in range(len(embedding_dim))])
            self.vectors.append([0.0 for _ in range(len(embedding_dim))])
            self.vectors.append([0.0 for _ in range(len(embedding_dim))])
            self.vectors.append([0.0 for _ in range(len(embedding_dim))])

    def vocabulary_size(self):
        """Gets the vocabulary size
        Args:
            None

        Returns:
            len(vocab) (int): The vocabulary size
        """
        return len(vocab)

    def build_vocabulary(self, tokens=None, embeddings=None):
        """Builds vocabulary from sentences or pretrained embeddings
        Args:
            sentences (list): Construct vocabulary from tokenized sentences

        Returns:
            None
        """
        if tokens is not None and embeddings is not None:
            raise ValueError("Only accepts either `tokens` or `embeddings`.")

        if tokens is not None:
            # Build from tokenized tokens
            # for sentence in tqdm(tokens):
            #     for word in tokens:
            #         print(type(word))
            #         exit()
            self.vocab.extend(
                list(set([
                    word
                    for sentence in tqdm(tokens)
                    for word in sentence
                ]))
            )
        elif embeddings is not None:
            # Build from pretrained embeddings
            for word in tqdm(embeddings):
                word = word.strip("\n")
                word = word.split(" ")

                self.vocab.append(word[0])
                vector = word[1:]
                self.vectors.append(vector)

    def tokenize(self, sentences):
        """A tokenizer for tokenizing Chinese characters and English words
        Args:
            sentences (list): A list of documents(news).

        Returns:
            tokens (list): A list of documents(list of tokens).
        """
        tokenized = []
        en = []
        prog = re.compile(r'\s+')
        for sentence in tqdm(sentences):
            sentence = unicodedata.normalize("NFKD", sentence.strip())
            sentence = prog.sub(' ', sentence)

            sentence = "".join([" "+character+" " if re.match(r"[^a-zA-Z]", character) else character for character in sentence])
            tokenized.append(sentence.split())

        self.build_vocabulary(tokenized)

        return tokenized

    def detokenize(self, tokens):
        """A detokenizer for detokenizing Chinese and English tokens into sentences
        Args:
            tokens (list): A list of documents(list of tokens).

        Returns:
            sentences (list): A list of documents(news).
        """
        joined = []
        for sentence in tqdm(tokens):
            detokened = " ".join(sentence)
            joined.append(detokened)

        return joined


    def encode(self, tokens):
        """Encoder for encoding tokens to their respective indexes of the vocabulary
        Args:
            tokens (list): A list of documents(list of tokens).

        Returns:
            encoded (list): A list of documents(list of idx).
        """
        encoded = []
        for sentence in tqdm(tokens):
            S = [self.bos_token]
            S.extend(sentence)
            S.append(self.eos_token)
            tmp = []
            for token in S:
                try:
                    index = self.vocab.index(token)
                    tmp.append(index)
                except:
                    tmp.append(self.unk_token_idx)
            encoded.append(tmp)
        return encoded

    def decode(self, encoded):
        """Encoder for encoding tokens to their respective indexes of the vocabulary
        Args:
            encoded (list): A list of documents(list of idx).

        Returns:
            tokens (list): A list of documents(list of tokens).
        """
        decoded = []
        for codes in encoded:
            tmp = []
            for code in codes:
                try:
                    word = self.vocab[code]
                    tmp.append(word)
                except:
                    tmp.append(self.unk_token)
            decoded.append(tmp)
        return decoded

    def labeler(self, labels, tokens):
        """Maps labels to their indexes within the document
        Args:
            labels (list): A list of key names of the news
            tokens (list): A list of news documents that are tokenized
        Returns:
            labels (list): Encoded list of documents with the location of label == 1
        """
        encoded = []
        for idx, document in tqdm(enumerate(tokens)):
            if labels[idx] is not []:
                tmp = []
                for token in document:
                    for label in labels[idx]:
                        is_kw = 0
                        if token in label:
                            tmp.append(1)
                            is_kw = 1
                            break
                        if is_kw == 0:
                            tmp.append(0)
                encoded.append(tmp)
            else:
                encoded.append([0 for token in document])

        return encoded

