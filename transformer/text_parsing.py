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





    def generate_batch(self, batch_size):
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

    def __get_vocabulary(self):
        """
        Creates a dictionary mapping words in training text to their frequencies
        ::Params::
            None
        ::Outputs::
            dict(word,frequency)
        """
        vocab = collections.defaultdict(int)
        for sentence in self.lang1.readlines():
            tokens = word_tokenize(sentence)
            for token in tokens:
                vocab[token] += 1
        for sentence in self.lang2.readlines():
            tokens = word_tokenize(sentence):
            for token in tokens:
                vocab[token] += 1
        return vocab

    def byte_pair_encoding(self):
        """
        Generates joint byte pair encoding of source and target language
        """
        vocab = self._get_vocabulary()
        vocab = dict([(tuple(x[:-1])+(x[-1]+'</w>',) ,y) for (x,y) in vocab.items()])
        sorted_vocab = sorted(vocab.items(),key=lambda x: x[1], reverse=True)

    def _get_pair_statistics(self,sorted_vocab):

        digram_freq = defaultdict(int)
        indices = defaultdict(lambda: defaultdict(int))
        for i, (word, freq) in enumerate(sorted_vocab):
            curr_char = word[0]
            for next_char in word[1:]:
                digram_freq[curr_char, next_char] += 1
                indices[curr_char, next_char][i] += 1
                curr_char = next_char
        return digram_freq, indices
