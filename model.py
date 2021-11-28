import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 1
NUMBER_OF_EPOCHS = 20
BATCHES_PER_EPOCH = 100
SAVE_MODEL_LOC = "./model_"


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

    def forward(self, input):
        input = input.swapaxes(1, 2)
        K = self.WK(input)
        K = K.T.reshape(self.num_heads, self.num_embeddings, self.len_head).squeeze()
        Q = self.WQ(input).reshape(self.num_heads, self.num_embeddings, self.len_head).squeeze()
        V = self.WV(input).reshape(self.num_heads, self.num_embeddings, self.len_head).squeeze()

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
    def __init__(self, num_embeddings=50, len_embedding=256, num_heads=8):
        super(Encoder, self).__init__()
        self.num_embeddings = num_embeddings
        self.len_embedding = len_embedding
        self.MHA = MHA(num_embeddings, len_embedding, num_heads)
        self.ff = nn.Linear(len_embedding, len_embedding)

    def forward(self, input):
        output = nn.BatchNorm1d(self.num_embeddings, device=DEVICE)(input)
        skip = self.MHA(input) + output
        output = nn.BatchNorm1d(self.num_embeddings, device=DEVICE)(skip)
        output = self.ff(output) + skip

        return output


class ViT(nn.Module):
    def __init__(self, num_encoders=5, len_embedding=128, num_heads=8, patch_size=4, input_res=28, num_classes=10):
        super(ViT, self).__init__()

        patches_per_dim = (input_res // patch_size) * (input_res // patch_size)

        self.num_encoders = num_encoders
        self.positional_embedding = nn.Embedding(patches_per_dim + 1, len_embedding)
        self.cls_token = nn.Parameter(torch.rand(1, len_embedding))
        self.convolution_embedding = nn.Conv2d(in_channels=1, out_channels=len_embedding, kernel_size=patch_size, stride=patch_size)
        self.classification_head = nn.Linear(len_embedding, num_classes, bias=False)

        self.stack_of_encoders = nn.ModuleList()
        for i in range(num_encoders):
            self.stack_of_encoders.append(Encoder(patches_per_dim + 1, len_embedding, num_heads))

    def forward(self, x):
        y_ = self.convolution_embedding(x)
        y_ = y_.flatten(start_dim=2, end_dim=3).swapaxes(1, 2)
        y_ = torch.cat((self.cls_token, y_.squeeze()))
        for e in range(len(y_)):
            y_[e] = y_[e] + self.positional_embedding.weight[e]
        
        y_ = y_.T.unsqueeze(dim=0)
        y_ = y_.swapaxes(1, 2)
        
        for encoder in self.stack_of_encoders:
            y_ = encoder(y_)

        y_ = self.classification_head(y_[:, 0, :])
        return y_


# measures accuracy of predictions at the end of an epoch (bad for semantic segmentation)
def accuracy(model, loader):
    correct = 0

    with torch.no_grad():
        for i, (x, y) in enumerate(loader):
            if i > BATCHES_PER_EPOCH:
                break
            y_ = model(x.to(DEVICE))
            y_ = torch.argmax(y_, dim=1)

            correct += (y_ == y.to(DEVICE))
            i += 1

    return correct / BATCHES_PER_EPOCH)


# a training loop that runs a number of training epochs on a model
def train(model, loss_function, optimizer, train_loader, validation_loader):
    for epoch in range(NUMBER_OF_EPOCHS):
        model.train()
        progress = tqdm(train_loader)

        for i, (x, y) in enumerate(progress):
            if i > BATCHES_PER_EPOCH:
                break

            y_ = model(x.to(DEVICE))
            loss = loss_function(y_, y.to(DEVICE))

            # make the progress bar display loss
            progress.set_postfix(loss=loss.item())

            # back propagation
            optimizer.zero_grad()  # zeros out the gradients from previous batch
            loss.backward()
            optimizer.step()

        model.eval()

        print("Test Accuracy for epoch (" + str(epoch) + ") is: " + str(accuracy(model, validation_loader)))
        print("Train Accuracy for epoch (" + str(epoch) + ") is: " + str(accuracy(model, train_loader)))

        torch.save(model.state_dict(), SAVE_MODEL_LOC + str(epoch))


if __name__ == "__main__":
    print(DEVICE)

    train_dataset = torchvision.datasets.MNIST(
        'C:/Users/Milan/Documents/Fast_Datasets/MNIST/',
        train=True,
        download=True,
        transform=torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(
                (0.1307,), (0.3081,))
        ]))

    validation_dataset = torchvision.datasets.MNIST(
        'C:/Users/Milan/Documents/Fast_Datasets/MNIST/',
        train=False,
        download=True,
        transform=torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(
                (0.1307,), (0.3081,))
        ]))

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
    
    # x, y = train_dataset[0]
    # # print(x.size)
    # plt.imshow(x)
    # plt.show()
    # print(y)

    model = ViT().to(DEVICE)
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    train(model, loss_function, optimizer, train_loader, validation_loader)