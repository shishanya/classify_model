import sys
sys.path.append("../utils")
sys.path.append("../layer")

from utils import create_dataset, Trainer
from layer import Embedding, FeaturesEmbedding, EmbeddingsInteraction, MultiLayerPerceptron
import torch
import torch.nn as nn
import torch.optim as optim


class Attention(nn.Module):
    def __init__(self, embed_dim = 4, t = 4):
        super(Attention, self).__init__()

        self.an = nn.Sequential(
            nn.Linear(embed_dim, t),
            nn.ReLU(),
            nn.Linear(t, 1, bias=False),
            nn.Flatten(),
            nn.Softmax(dim = 1)
        )

    def forward(self, x):
         return self.an(x)


class AFM(nn.Module):
    def __init__(self, field_dim, embed_dim = 4):
        super(AFM, self).__init__()

        num_fields = len(field_dim)

        self.w0 = nn.Parameter(torch.zeros((1,)))
        self.embed1 = FeaturesEmbedding(field_dim, 1)
        self.embed2 = FeaturesEmbedding(field_dim, embed_dim)
        self.interact = EmbeddingsInteraction()
        self.attention = Attention(embed_dim)

        self.p = nn.Parameter(torch.zeros(embed_dim, ))
        nn.init.xavier_uniform_(self.p.unsqueeze(0).data)

    def forward(self, x):

        embeddings = self.embed2(x)
        interact = self.interact(embeddings)
        attr = self.attention(interact)
        # print(interact.mul(attr.unsqueeze(-1)).shape,
        #       interact.mul(attr.unsqueeze(-1)).sum(dim=1).mul(self.p).shape,
        #       interact.mul(attr.unsqueeze(-1)).sum(dim=1).mul(self.p).sum(dim=1, keepdim=True).shape)
        attr_part = interact.mul(attr.unsqueeze(-1)).sum(dim = 1).mul(self.p).sum(dim=1, keepdim=True)

        output = self.w0 + self.embed1(x).sum(dim = 1) + attr_part
        output = torch.sigmoid(output)

        return output

if __name__ == "__main__":
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('Training on [{}].'.format(device))
    dataset = create_dataset('criteo', sample_num=100000, device=device)
    field_dims, (train_X, train_y), (valid_X, valid_y), (test_X, test_y) = dataset.train_valid_test_split()

    EMBEDDING_DIM = 8
    LEARNING_RATE = 1e-4
    REGULARIZATION = 1e-6
    BATCH_SIZE = 4096
    EPOCH = 1000
    TRIAL = 100

    afm = AFM(field_dims, EMBEDDING_DIM).to(device)
    optimizer = optim.Adam(afm.parameters(), lr=LEARNING_RATE, weight_decay=REGULARIZATION)
    criterion = nn.BCELoss()

    trainer = Trainer(afm, optimizer, criterion, BATCH_SIZE)
    trainer.train(train_X, train_y, epoch=EPOCH, trials=TRIAL, valid_X=valid_X, valid_y=valid_y)
    test_loss, test_auc = trainer.test(test_X, test_y)
    print('test_loss:  {:.5f} | test_auc:  {:.5f}'.format(test_loss, test_auc))


















