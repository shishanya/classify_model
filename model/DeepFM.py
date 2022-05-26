import sys
sys.path.append("../utils")
sys.path.append("../layer")

from utils import create_dataset, Trainer
from layer import Embedding, FeaturesEmbedding, EmbeddingsInteraction, MultiLayerPerceptron
import torch
import torch.nn as nn
import torch.optim as optim

class DeepFM(nn.Module):

    def __init__(self, field_dims, embed_dim=4):
        super(DeepFM, self).__init__()

        num_fileds = len(field_dims)
        self.embed1 = FeaturesEmbedding(field_dims, 1)
        self.embed2 = FeaturesEmbedding(field_dims, embed_dim)
        self.fm = EmbeddingsInteraction()
        self.deep = MultiLayerPerceptron([len(field_dims) * embed_dim, 128, 64, 32])
        self.fc = nn.Linear(1 +num_fileds*(num_fileds- 1)//2 + 32, 1)

    def forward(self, x):

        embed1_output = self.embed1(x).sum(dim = 1)
        embedding = self.embed2(x)
        fm_output = self.fm(embedding).sum(dim = -1)
        deep_output = self.deep(embedding.reshape(embedding.shape[0], -1))
        # print(embed1_output.shape, fm_output.shape, deep_output.shape, self.fm(x).shape)

        stacked = torch.hstack([embed1_output, fm_output, deep_output])
        output = self.fc(stacked)
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

    DFM = DeepFM(field_dims, EMBEDDING_DIM).to(device)
    optimizer = optim.Adam(DFM.parameters(), lr=LEARNING_RATE, weight_decay=REGULARIZATION)
    criterion = nn.BCELoss()

    trainer = Trainer(DFM, optimizer, criterion, BATCH_SIZE)
    trainer.train(train_X, train_y, epoch=EPOCH, trials=TRIAL, valid_X=valid_X, valid_y=valid_y)
    test_loss, test_auc = trainer.test(test_X, test_y)
    print('test_loss:  {:.5f} | test_auc:  {:.5f}'.format(test_loss, test_auc))