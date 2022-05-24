from utils.utils import create_dataset, Trainer
from layer.layer import Embedding, FeaturesEmbedding, EmbeddingsInteraction, MultiLayerPerceptron
import torch
import torch.nn as nn
import torch.optim as optim


class FFM(nn.Module):
    def __init__(self, field_dim, embed_dim = 4):
        super(FFM, self).__init__()

        self.field_dim = field_dim
        self.embed_dim = embed_dim

        self.bias = nn.Parameter(torch.zeros((1,)))

        self.embed_linear = FeaturesEmbedding(field_dim, 1)
        self.embed_cross = nn.ModuleList([FeaturesEmbedding(field_dim, embed_dim) for _ in field_dim])

    def forward(self, x):

        num_fields = len(self.field_dim)
        embeddings = [embed(x) for embed in self.embed_cross]
        embeddings = torch.hstack(embeddings)

        i1 = []
        i2 = []
        for i in range(num_fields):
            for j in range(i + 1, num_fields):
                i1.append(j * num_fields + i)
                i2.append(i * num_fields + j)

        embeddings_cross = torch.mul(embeddings[:, i1], embeddings[:, i2]).sum(dim = 2).sum(dim = 1, keepdim = True)
        output = self.embed_linear(x).sum(dim = 1) + self.bias + embeddings_cross
        output = torch.sigmoid(output)
        return output


if __name__ == '__main__':
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

    ffm = FFM(field_dims, EMBEDDING_DIM).to(device)
    optimizer = optim.Adam(ffm.parameters(), lr=LEARNING_RATE, weight_decay=REGULARIZATION)
    criterion = nn.BCELoss()

    trainer = Trainer(ffm, optimizer, criterion, BATCH_SIZE)
    trainer.train(train_X, train_y, epoch=EPOCH, trials=TRIAL, valid_X=valid_X, valid_y=valid_y)
    test_loss, test_auc = trainer.test(test_X, test_y)
    print('test_loss:  {:.5f} | test_auc:  {:.5f}'.format(test_loss, test_auc))