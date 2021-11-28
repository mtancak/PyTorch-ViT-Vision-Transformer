# PyTorch-ViT-Visual-Transformer
## Introduction
PyTorch implementation of the Visual Transformer architecture using this paper: https://arxiv.org/pdf/2010.11929.pdf. This architecture is trained with a basic MNIST placeholder to test that it works. Ideally you would do a grid search for finding hyperparameters, especially transformations, and you would use such architectures for more complicated datasets, however, this is out of scope for this repo. You can also add whatever classification head you want, including random forests, with good results. 

*To-do:* display attention as images

## Results on MNIST with a shallow network and a very small number of batches per epoch due to lack of resources.
![](Images/accuracy.png)
![](Images/cm_train.png)
![](Images/cm_val.png)
