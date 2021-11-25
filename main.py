import torch
import torch.nn as nn


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

        self.WK = nn.Conv2d(
            in_channels=num_embeddings,
            out_channels=num_heads * num_embeddings,
            kernel_size=(1, 1)
        )
        self.WQ = nn.Conv2d(
            in_channels=num_embeddings,
            out_channels=num_heads * num_embeddings,
            kernel_size=(1, 1)
        )
        self.WV = nn.Conv2d(
            in_channels=num_embeddings,
            out_channels=num_heads * num_embeddings,
            kernel_size=(1, 1)
        )

        self.WZ = nn.Conv2d(
            in_channels=num_heads * num_embeddings,
            out_channels=num_embeddings,
            kernel_size=(1, 1)
        )

    def forward(self, embeddings):
        K = self.WK(embeddings).reshape(self.num_heads, self.num_embeddings, 1, self.len_embedding).squeeze()
        Q = self.WQ(embeddings).reshape(self.num_heads, self.num_embeddings, 1, self.len_embedding).squeeze()
        V = self.WV(embeddings).reshape(self.num_heads, self.num_embeddings, 1, self.len_embedding).squeeze()

        print("k shape = " + str(K.shape))
        print("q shape = " + str(Q.shape))
        print("v shape = " + str(V.shape))

        score = Q.bmm(K.transpose(dim0=1, dim1=2))
        print("score shape = " + str(score.shape))
        print("len_head = " + str(self.len_head))

        indexes = torch.softmax(score / self.len_head, dim=2)
        print("indexes shape = " + str(indexes.shape))

        Z = indexes.bmm(V).reshape(1, 400, 1, 256)
        print("Z shape = " + str(Z.shape))

        output = self.WZ(Z)
        return output


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()


class ViT(nn.Module):
    def __init__(self, num_encoders=3, num_decoders=3, len_embedding=10, num_heads=8):
        super(ViT, self).__init__()
        self.num_encoders = num_encoders
        self.num_decoders = num_decoders
        self.len_embedding = len_embedding
        self.num_heads = num_heads

        self.stack_of_encoders = nn.ModuleList()
        # self.stack_of_decoders = nn.ModuleList()


if __name__ == "__main__":
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print(DEVICE)

    x = torch.rand(1, 50, 1, 256).to(DEVICE)
    print(x.shape)

    sa = SelfAttention().to(DEVICE)
    pred = sa(x)
    print(pred.shape)
