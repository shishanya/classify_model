from utils.utils import create_dataset, Trainer
from layer.layer import Embedding, FeaturesEmbedding, EmbeddingsInteraction, MultiLayerPerceptron
import torch
import torch.nn as nn
import torch.optim as optim


class WideDeep(nn.Module):

    def __init__(self, field_dims, embed_dim=4):
        super(WideDeep, self).__init__()

        self.wide = FeaturesEmbedding(field_dims, 1)
        self.embedding = FeaturesEmbedding(field_dims, embed_dim)
        self.deep = MultiLayerPerceptron([len(field_dims)*embed_dim, 128, 64, 32])
        self.fc = nn.Linear(32 + len(field_dims) * embed_dim, 1)

    def forward(self, x):
        # x shape: (batch_size, num_fields)
        wide_output = self.wide(x)
        embedding_output = self.embedding(x).reshape(x.shape[0], -1)
        deep_output = self.deep(embedding_output)
        concat = torch.hstack([embedding_output, deep_output])
        output = self.fc(concat)
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

    wd = WideDeep(field_dims, EMBEDDING_DIM).to(device)
    optimizer = optim.Adam(wd.parameters(), lr=LEARNING_RATE, weight_decay=REGULARIZATION)
    criterion = nn.BCELoss()

    trainer = Trainer(wd, optimizer, criterion, BATCH_SIZE)
    trainer.train(train_X, train_y, epoch=EPOCH, trials=TRIAL, valid_X=valid_X, valid_y=valid_y)
    test_loss, test_auc = trainer.test(test_X, test_y)
    print('test_loss:  {:.5f} | test_auc:  {:.5f}'.format(test_loss, test_auc))