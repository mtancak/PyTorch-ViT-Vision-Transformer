import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision


BATCH_SIZE = 1


class SelfAttention(nn.Module):
    def __init__(self, num_embeddings=50, len_embedding=256, num_heads=8):
        super(SelfAttention, self).__init__()

        self.len_embedding = len_embedding
        self.num_embeddings = num_embeddings
        self.num_heads = num_heads
        assert (len_embedding % num_heads == 0)
        self.len_head = len_embedding // num_heads
        data_dim = tuple((num_heads, len_embedding, self.len_head))

        print("num_embeddings = " + str(num_embeddings))
        print("num_heads = " + str(num_heads))
        print("len_embedding = " + str(len_embedding))
        print("len_head = " + str(self.len_head))

        self.WK = nn.Conv1d(
            in_channels=self.len_embedding,
            out_channels=num_heads * self.len_head,
            kernel_size=1
        )
        self.WQ = nn.Conv1d(
            in_channels=self.len_embedding,
            out_channels=num_heads * self.len_head,
            kernel_size=1
        )
        self.WV = nn.Conv1d(
            in_channels=self.len_embedding,
            out_channels=num_heads * self.len_head,
            kernel_size=1
        )

        self.WZ = nn.Conv1d(
            in_channels=num_heads * self.len_head,
            out_channels=len_embedding,
            kernel_size=1
        )

    def forward(self, input):
        print("input shape = " + str(input.shape))
        K = self.WK(input)
        print("K1 shape = " + str(K.shape))
        K = K.T.reshape(self.num_heads, self.num_embeddings, self.len_head).squeeze()
        print("K2 shape = " + str(K.shape))
        Q = self.WQ(input).reshape(self.num_heads, self.num_embeddings, self.len_head).squeeze()
        V = self.WV(input).reshape(self.num_heads, self.num_embeddings, self.len_head).squeeze()

        print("k shape = " + str(K.shape))
        print("q shape = " + str(Q.shape))
        print("v shape = " + str(V.shape))

        score = Q.bmm(K.transpose(dim0=1, dim1=2))
        print("score shape = " + str(score.shape))
        print("len_head = " + str(self.len_head))

        indexes = torch.softmax(score / self.len_head, dim=2)
        print("indexes shape = " + str(indexes.shape))

        Z = indexes.bmm(V)
        print("Z1 shape = " + str(Z.shape))
        Z = Z.moveaxis(1, 2)
        print("Z2 shape = " + str(Z.shape))
        Z = Z.flatten(start_dim=0, end_dim=1)
        print("Z3 shape = " + str(Z.shape))
        Z = Z.unsqueeze(dim=0)
        print("Z4 shape = " + str(Z.shape))

        output = self.WZ(Z)
        return output


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.selfAttention = SelfAttention()
        self.ff = nn.Linear(1, 1)

    def forward(self, input):
        output = self.selfAttention(input)
        output = self.ff(output)

        return output


class ViT(nn.Module):
    def __init__(self, num_encoders=3, len_embedding=10, num_heads=8, patch_size=20, input_length=5, num_classes=10):
        super(ViT, self).__init__()
        self.num_encoders = num_encoders
        self.positional_embedding = nn.Embedding(input_length + 1, patch_size * patch_size)
        self.cls_token = nn.Parameter(torch.rand(patch_size * patch_size))
        self.convolution_embedding = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=patch_size, stride=patch_size)
        self.classification_head = nn.Linear(patch_size * patch_size, num_classes, bias=False)

        self.stack_of_encoders = nn.ModuleList()
        for i in range(num_encoders):
            self.stack_of_encoders.append(Encoder())

    def forward(self, x):
        y_ = self.convolution_embedding(x)
        for encoder in self.stack_of_encoders:
            x_ = encoder(y_)
        y_ = self.classification_head(y_)

        return y_


if __name__ == "__main__":
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print(DEVICE)

    x = torch.rand(1, 256, 50).to(DEVICE)
    print(x.shape)

    sa = SelfAttention().to(DEVICE)
    pred = sa(x)
    print("output shape = " + str(pred.shape))

    train_dataset = torchvision.datasets.MNIST(
        'C:/Users/Milan/Documents/Fast_Datasets/MNIST/',
        train=True,
        download=True)

    validation_dataset = torchvision.datasets.MNIST(
        'C:/Users/Milan/Documents/Fast_Datasets/MNIST/',
        train=False,
        download=True)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        num_workers=0,
        shuffle=True)

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=1,
        num_workers=0,
        shuffle=False)

    model = ViT().to(DEVICE)