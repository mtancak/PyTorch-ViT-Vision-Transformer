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
    def __init__(self):
        super(Transformer, self).__init__()


if __name__ == "__main__":
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print(DEVICE)

    model = Transformer().to(DEVICE)