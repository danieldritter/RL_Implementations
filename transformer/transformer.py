import torch.nn as nn


class Transformer(nn.Module):

    def __init__(self,vocab_size,embedding_size):
        super(Transformer,self).__init__()
        self.embeddings = nn.Embedding(vocab_size,embedding_size)

    def encoding_pass(self,input):
        pass

    def decoding_pass(self,input):
        pass

    def forward(self,input):
        pass

    def layer_normalization(self,layer):
        pass

    class Encoder(nn.Module):

        def __init__(self):
            super(Encoder,self).__init__()

        def forward(self):
            pass

        def self_attention_layer(self,Q,K,V):
            """
            Calculates Multi-Head Self-Attention
            """
            pass

        def feed_forward_layer(self,input):
            pass

    class Decoder(nn.Module):

        def __init__(self):
            super(Decoder,self).__init__()

        def forward(self):
            pass

        def calc_masked_self_attention(self,Q,K,V):
            pass

def __main__():
    pass

if __name__ == "__main__":
    __main__()
