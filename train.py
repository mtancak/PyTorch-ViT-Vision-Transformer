import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision

from model import ViT

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 1
NUMBER_OF_EPOCHS = 50
BATCHES_PER_EPOCH = 100
LOAD_MODEL_LOC = None
SAVE_MODEL_LOC = "./model_"
PRINT = True
PRINT_GRAPH = True
PRINT_CM = True


# measures accuracy of predictions at the end of an epoch (bad for semantic segmentation)
def accuracy(model, loader, num_classes=10):
    correct = 0
    cm = np.zeros((num_classes, num_classes))

    with torch.no_grad():
        for i, (x, y) in enumerate(loader):
            if i > BATCHES_PER_EPOCH:
                break
            y_ = model(x.to(DEVICE))
            y_ = torch.argmax(y_, dim=1)

            correct += (y_ == y.to(DEVICE))
            cm[y][y_] += 1

    if PRINT_CM:
        class_labels = list(range(num_classes))
        ax = sn.heatmap(
            cm,
            annot=True,
            cbar=False,
            xticklabels=class_labels,
            yticklabels=class_labels, 
            fmt='g')
        ax.set(
            xlabel="prediction",
            ylabel="truth",
            title="Confusion Matrix for " + ("Training set" if loader.dataset.train else "Validation dataset"))
        plt.show()

    return (correct / BATCHES_PER_EPOCH).item() * 100


# a training loop that runs a number of training epochs on a model
def train(model, loss_function, optimizer, train_loader, validation_loader):
    accuracy_per_epoch_train = []
    accuracy_per_epoch_val = []

    for epoch in range(NUMBER_OF_EPOCHS):
        model.train()
        progress = tqdm(train_loader)
        
        for i, (x, y) in enumerate(progress):
            x, y = x.to(DEVICE), y.to(DEVICE)
            
            if i > BATCHES_PER_EPOCH:
                break

            y_ = model(x)
            loss = loss_function(y_, y)

            # make the progress bar display loss
            progress.set_postfix(loss=loss.item())

            # back propagation
            optimizer.zero_grad()  # zeros out the gradients from previous batch
            loss.backward()
            optimizer.step()

        model.eval()

        accuracy_per_epoch_train.append(accuracy(model, train_loader))
        accuracy_per_epoch_val.append(accuracy(model, validation_loader))

        if SAVE_MODEL_LOC:
            torch.save(model.state_dict(), SAVE_MODEL_LOC + str(epoch))

        if PRINT:
            print("Test Accuracy for epoch (" + str(epoch) + ") is: " + str(accuracy_per_epoch_val[-1]))
            print("Train Accuracy for epoch (" + str(epoch) + ") is: " + str(accuracy_per_epoch_train[-1]))

        if PRINT_GRAPH:
            plt.figure(figsize=(10, 10), dpi=100)
            plt.plot(range(0, epoch + 1), accuracy_per_epoch_train,
                     color='b', marker='o', linestyle='dashed', label='Training')
            plt.plot(range(0, epoch + 1), accuracy_per_epoch_val,
                     color='r', marker='o', linestyle='dashed', label='Validation')
            plt.legend()
            plt.title("Graph of accuracy over time")
            plt.xlabel("epoch #")
            plt.ylabel("accuracy %")
            if epoch < 20:
                plt.xticks(range(0, epoch + 1))
            plt.ylim(0, 100)
            plt.show()
        

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

    model = ViT(device=DEVICE).to(DEVICE)
    if LOAD_MODEL_LOC:
        model.load_state_dict(torch.load(LOAD_MODEL_LOC))

    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    train(model, loss_function, optimizer, train_loader, validation_loader)
