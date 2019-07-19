from nltk import word_tokenize
import collections


class Parser:
    """
    Class for parsing text data to feed into transformer. Expects two text
    files in two different languages with one sentence per line in each
    file.
    """


    def __init__(self,lang1_file,lang2_file,vocab_dict):
        self.lang1 = open(lang1_file)
        self.lang2 = open(lang2_file)
        self.encoding_map = byte_pair_encoding()


    def generate_sentence_pair(self):
        """
        Creates a tokenized pair of sentences in the
        two given languages
        ::Params::
            None
        ::Outputs::
            [["lang1_word","lang1_word",...],["lang2_word","lang2_word",...]]
        """
        lang1_token_sentence = word_tokenize(self.lang1.readline())
        lang2_token_sentence = word_tokenize(self.lang2.readline())
        return [lang1_token_sentence,lang2_token_sentence]

    def byte_pair_encoding(self):
        pass

    def generate_batch(self,batch_size):
        """
        Creates a batch of tokenized sentence pairs of size batch size
        ::Params::
            batch_size: Size of the batch to generate
        ::Outputs::
            [[sentence_pair1],[sentence_pair2],...]
        """
        pass

    def close_parser(self):
        """
        Closes both text files when parser is no longer needed
        ::Params::
            None
        ::Outputs::
            None
        """
        self.lang1.close()
        self.lang2.close()

class Vocabulary_Encoder:
    """
    This is a class to perform data preprocessing on the text data, such as
    learning the joint byte-pair encoding between the target and source
    languages
    """

    def __init__(self,lang1_file,lang2_file,total_vocab_size):
        self.lang1_file = open(lang1_file)
        self.lang2_file = open(lang2_file)
        self.total_vocab_size = total_vocab_size

    def get_digram_freqs(self):
        """
        Generates a dictionary mapping from a symbol pair
        to its frequency in the training data
        """
        pairs = collections.defaultdict()


    def get_vocab_freqs(self):
        vocab = collections.defaultdict(int)
        for sentence in self.lang1_file.readlines:
            for word in word_tokenize(sentence):
                vocab[word] += 1

        for sentence in self.lang2_file.readlines():
            for word in word_tokenize(sentence):
                vocab[word] += 1
        return vocab
