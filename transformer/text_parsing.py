from nltk import word_tokenize



class Parser:
    """
    Class for parsing text data to feed into transformer. Expects two text
    files in two different languages with one sentence per line in each
    file.
    """


    def __init__(self,lang1_file,lang2_file):
        self.lang1 = open(lang1_file)
        self.lang2 = open(lang2_file)


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

    def byte_pair_encode(self):
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
