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
        Z = Z.moveaxis(1, 0)
        print("Z2 shape = " + str(Z.shape))
        Z = Z.flatten(start_dim=1, end_dim=2).unsqueeze(dim=2)
        print("Z3 shape = " + str(Z.shape))

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

    x = torch.rand(50, 256, 1).to(DEVICE)
    print(x.shape)

    sa = SelfAttention().to(DEVICE)
    pred = sa(x)
    print("output shape = " + str(pred.shape))
