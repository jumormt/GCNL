import os.path as osp

import torch
import torch.nn.functional as F
from torch.nn import Sequential, Linear, ReLU
from torch_geometric.datasets import TUDataset
from torch_geometric.data import DataLoader
from torch_geometric.nn import GINConv, global_add_pool
from dataset_build import Test787DatasetTest


from torch_geometric.utils import f1_score
from torch_geometric.utils import accuracy
from torch_geometric.utils import true_positive
from torch_geometric.utils import true_negative
from torch_geometric.utils import false_positive
from torch_geometric.utils import false_negative
from torch_geometric.utils import precision
from torch_geometric.utils import recall

path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', 'MUTAG')
# dataset = TUDataset(path, name='MUTAG').shuffle()
dataset = Test787DatasetTest(root="/home/cry/chengxiao/dataset/Test787DatasetTest")
test_dataset = dataset[:len(dataset) // 10]
train_dataset = dataset[len(dataset) // 10:]
test_loader = DataLoader(test_dataset, batch_size=128)
train_loader = DataLoader(train_dataset, batch_size=128)
print("dataset.num_features ",dataset.num_features)
print("dataset.num_classes ",dataset.num_classes)
# for ba in test_loader:
#     print(ba.num_graphs)
#     print(ba.y)


class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        num_features = dataset.num_features# node特征向量维度
        dim = 32

        nn1 = Sequential(Linear(num_features, dim), ReLU(), Linear(dim, dim))
        self.conv1 = GINConv(nn1)
        self.bn1 = torch.nn.BatchNorm1d(dim)

        nn2 = Sequential(Linear(dim, dim), ReLU(), Linear(dim, dim))
        self.conv2 = GINConv(nn2)
        self.bn2 = torch.nn.BatchNorm1d(dim)

        nn3 = Sequential(Linear(dim, dim), ReLU(), Linear(dim, dim))
        self.conv3 = GINConv(nn3)
        self.bn3 = torch.nn.BatchNorm1d(dim)

        nn4 = Sequential(Linear(dim, dim), ReLU(), Linear(dim, dim))
        self.conv4 = GINConv(nn4)
        self.bn4 = torch.nn.BatchNorm1d(dim)

        nn5 = Sequential(Linear(dim, dim), ReLU(), Linear(dim, dim))
        self.conv5 = GINConv(nn5)
        self.bn5 = torch.nn.BatchNorm1d(dim)

        self.fc1 = Linear(dim, dim)
        self.fc2 = Linear(dim, dataset.num_classes)# 分类数

    def forward(self, x, edge_index, batch):
        x = F.relu(self.conv1(x, edge_index))
        x = self.bn1(x)
        x = F.relu(self.conv2(x, edge_index))
        x = self.bn2(x)
        x = F.relu(self.conv3(x, edge_index))
        x = self.bn3(x)
        x = F.relu(self.conv4(x, edge_index))
        x = self.bn4(x)
        x = F.relu(self.conv5(x, edge_index))
        x = self.bn5(x)
        x = global_add_pool(x, batch)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=-1)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Net().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)


def train(epoch):
    model.train()

    if epoch == 51:
        for param_group in optimizer.param_groups:
            param_group['lr'] = 0.5 * param_group['lr']

    loss_all = 0
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        output = model(data.x, data.edge_index, data.batch)
        loss = F.nll_loss(output, data.y)
        loss.backward()
        loss_all += loss.item() * data.num_graphs
        optimizer.step()
    return loss_all / len(train_dataset)


def test(loader):
    model.eval()

    correct = 0
    flag = 0
    preds = None
    datays = None
    for data in loader:
        data = data.to(device)
        output = model(data.x, data.edge_index, data.batch)# 每个data有128张图，输出1×128维向量，通过距离定义分类
        pred = output.max(dim=1)[1]
        correct += pred.eq(data.y).sum().item()
        # print("type pred:", type(pred), "type y: ", type(data.y))
        if(flag == 0):
            preds = pred
            datays = data.y
            flag = 1
        else:
            # print("preds ", preds.size())
            # print("pred ", pred.size())
            preds = torch.cat((preds, pred), 0)
            datays = torch.cat((datays, data.y), 0)
    print("f1socre: {}".format(f1_score(preds, datays, 2)))
    print("true_positive: {}".format(true_positive(preds, datays, 2)))
    print("accuracy: {}".format(accuracy(preds, datays)))
    return correct / len(loader.dataset)


for epoch in range(1, 101):
    train_loss = train(epoch)
    print("train:")
    train_acc = test(train_loader)
    print("test:")
    test_acc = test(test_loader)
    print('Epoch: {:03d}, Train Loss: {:.7f}, '
          'Train Acc: {:.7f}, Test Acc: {:.7f}'.format(epoch, train_loss,
                                                       train_acc, test_acc))