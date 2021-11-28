import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision

class MHA(nn.Module):
    def __init__(self, num_embeddings=50, len_embedding=256, num_heads=8):
        super(MHA, self).__init__()

        self.len_embedding = len_embedding
        self.num_embeddings = num_embeddings
        self.num_heads = num_heads

        assert (len_embedding % num_heads == 0)
        self.len_head = len_embedding // num_heads

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
        
    def forward(self, inp):
        inp = inp.swapaxes(1, 2)
        K = self.WK(inp).T.reshape(self.num_heads, self.num_embeddings, self.len_head).squeeze()
        Q = self.WQ(inp).reshape(self.num_heads, self.num_embeddings, self.len_head).squeeze()
        V = self.WV(inp).reshape(self.num_heads, self.num_embeddings, self.len_head).squeeze()

        score = Q.bmm(K.transpose(dim0=1, dim1=2))

        indexes = torch.softmax(score / self.len_head, dim=2)
        
        Z = indexes.bmm(V)
        Z = Z.moveaxis(1, 2)
        Z = Z.flatten(start_dim=0, end_dim=1)
        Z = Z.unsqueeze(dim=0)

        output = self.WZ(Z)
        output = output.swapaxes(1, 2)
        return output


class Encoder(nn.Module):
    def __init__(self, num_embeddings=50, len_embedding=256, num_heads=8, device="cpu"):
        super(Encoder, self).__init__()
        self.num_embeddings = num_embeddings
        self.len_embedding = len_embedding
        self.MHA = MHA(num_embeddings, len_embedding, num_heads)
        self.ff = nn.Linear(len_embedding, len_embedding)
        
        self.device = device

    def forward(self, inp):
        output = nn.BatchNorm1d(self.num_embeddings, device=self.device)(inp)
        skip = self.MHA(inp) + output
        output = nn.BatchNorm1d(self.num_embeddings, device=self.device)(skip)
        output = self.ff(output) + skip

        return output


class ViT(nn.Module):
    def __init__(self, num_encoders=5, len_embedding=128, num_heads=8, patch_size=4, input_res=28, num_classes=10, device="cpu"):
        super(ViT, self).__init__()

        patches_per_dim = (input_res // patch_size) * (input_res // patch_size)

        self.num_encoders = num_encoders
        self.positional_embedding = nn.Embedding(patches_per_dim + 1, len_embedding)
        self.cls_token = nn.Parameter(torch.rand(1, len_embedding))
        self.convolution_embedding = nn.Conv2d(
            in_channels=1,
            out_channels=len_embedding,
            kernel_size=patch_size,
            stride=patch_size)
        self.classification_head = nn.Linear(
            in_features=len_embedding,
            out_features=num_classes,
            bias=False)

        self.stack_of_encoders = nn.ModuleList()
        for i in range(num_encoders):
            self.stack_of_encoders.append(Encoder(patches_per_dim + 1, len_embedding, num_heads, device))

    def forward(self, x):
        z = self.convolution_embedding(x)
        z = z.flatten(start_dim=2, end_dim=3).swapaxes(1, 2)
        z = torch.cat((self.cls_token, z.squeeze()))
        for e in range(len(z)):
            z[e] = z[e] + self.positional_embedding.weight[e]

        z = z.T.unsqueeze(dim=0)
        z = z.swapaxes(1, 2)

        for encoder in self.stack_of_encoders:
            z = encoder(z)

        z = z[:, 0, :]

        y_ = self.classification_head(z)
        return y_


if __name__ == "__main__":
    
    model = ViT()
    model(torch.rand((1, 1, 28, 28)))
