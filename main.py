import torch
import torch.nn as nn


class Attention(nn.Module):
    def __init__(self):
        super(Attention, self).__init__()


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()


class Transformer(nn.Module):
    def __init__(self, number_of_encoders=3, number_of_decoders=3, embedding_size=10, number_of_attention_heads=8):
        super(Transformer, self).__init__()
        self.number_of_encoders = number_of_encoders
        self.number_of_decoders = number_of_decoders
        self.embedding_size = embedding_size
        self.number_of_attention_heads = number_of_attention_heads

        self.stack_of_encoders = nn.ModuleList()
        # self.stack_of_decoders = nn.ModuleList()


if __name__ == "__main__":
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print(DEVICE)

    model = Transformer().to(DEVICE)