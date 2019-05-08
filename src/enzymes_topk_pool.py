import os.path as osp

import torch
import torch.nn.functional as F
from torch_geometric.datasets import TUDataset
from torch_geometric.data import DataLoader
from torch_geometric.nn import GraphConv, TopKPooling
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
from torch_geometric.nn import GINConv, global_add_pool
from torch.nn import Sequential, Linear, ReLU

from dataset_build import Test787DatasetTest
from torch_geometric.utils import f1_score
from torch_geometric.utils import accuracy
from torch_geometric.utils import true_positive
from torch_geometric.utils import true_negative
from torch_geometric.utils import false_positive
from torch_geometric.utils import false_negative
from torch_geometric.nn import GCNConv, ChebConv  # noqa
from torch_geometric.nn import GATConv
from sklearn import metrics

import logging
import sys
import time
import numpy as np
from sklearn.model_selection import KFold

logger = logging.getLogger(__name__)
logger.setLevel(level=logging.INFO)
file_handler = logging.FileHandler(
    '/home/cry/chengxiao/dataset/tscanc/SARD_119_399/result/log/399f_result.txt')
file_handler.setLevel(level=logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

# StreamHandler
stream_handler = logging.StreamHandler(sys.stdout)
stream_handler.setFormatter(formatter)
stream_handler.setLevel(level=logging.INFO)
logger.addHandler(stream_handler)

path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', 'ENZYMES')
# dataset = Test787DatasetTest(root="/home/cry/chengxiao/dataset/21DatasetTest")
# dataset = Test787DatasetTest(root="/home/cry/chengxiao/dataset/840DatasetTest")
# dataset = Test787DatasetTest(root="/home/cry/chengxiao/dataset/Test691DatasetTest")
# dataset = Test787DatasetTest(root="/home/cry/chengxiao/dataset/BehaviorDatasetTest")
# dataset = Test787DatasetTest(root="/home/cry/chengxiao/dataset/119DatasetTest")
# dataset = Test787DatasetTest(root="/home/cry/chengxiao/dataset/691DatasetTest")
dataset = Test787DatasetTest(root="/home/cry/chengxiao/dataset/399DatasetTest")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        # self.conv1 = GraphConv(dataset.num_features, 128)
        self.conv1 = GCNConv(dataset.num_features, 128, improved=False)
        self.pool1 = TopKPooling(128, ratio=0.8)
        # self.conv2 = GraphConv(128, 128)
        self.conv2 = GCNConv(128, 128, improved=False)
        self.pool2 = TopKPooling(128, ratio=0.8)
        # self.conv3 = GraphConv(128, 128)
        self.conv3 = GCNConv(128, 128, improved=False)
        self.pool3 = TopKPooling(128, ratio=0.8)

        self.lin1 = torch.nn.Linear(256, 128)
        self.lin2 = torch.nn.Linear(128, 64)
        self.lin3 = torch.nn.Linear(64, dataset.num_classes)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        x = F.relu(self.conv1(x, edge_index))
        x, edge_index, _, batch, _ = self.pool1(x, edge_index, None, batch)
        x1 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        x = F.relu(self.conv2(x, edge_index))
        x, edge_index, _, batch, _ = self.pool2(x, edge_index, None, batch)
        x2 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        x = F.relu(self.conv3(x, edge_index))
        x, edge_index, _, batch, _ = self.pool3(x, edge_index, None, batch)
        x3 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        x = x1 + x2 + x3

        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu(self.lin2(x))
        x = F.log_softmax(self.lin3(x), dim=-1)

        return x


def train(epoch, model, train_loader, optimizer, train_dataset):
    model.train()

    loss_all = 0
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        output = model(data)
        # output = model(data.x, data.edge_index, data.batch)
        loss = F.nll_loss(output, data.y)
        loss.backward()
        loss_all += data.num_graphs * loss.item()
        optimizer.step()
    return loss_all / len(train_dataset), model, optimizer


def test(loader, model):
    model.eval()

    flag = 0
    preds = None
    datays = None
    correct = 0
    for data in loader:
        data = data.to(device)
        pred = model(data).max(dim=1)[1]
        # output = model(data.x, data.edge_index, data.batch)  # 每个data有128张图，输出1×128维向量，通过距离定义分类
        # pred = output.max(dim=1)[1]

        correct += pred.eq(data.y).sum().item()
        if (flag == 0):
            preds = pred
            datays = data.y
            flag = 1
        else:
            # print("preds ", preds.size())
            # print("pred ", pred.size())
            preds = torch.cat((preds, pred), 0)
            datays = torch.cat((datays, data.y), 0)
    # logger.info("f1socre: {}".format(f1_score(preds, datays, 2)))
    # print('recall: %.8f' % metrics.recall_score(preds.cpu(), datays.cpu()))
    TP = true_positive(preds, datays, 2).numpy()[1]
    TN = true_negative(preds, datays, 2).numpy()[1]
    FP = false_positive(preds, datays, 2).numpy()[1]
    FN = false_negative(preds, datays, 2).numpy()[1]
    # logger.info("true_positive: {}".format(true_positive(preds, datays, 2)))
    # logger.info("true_negative: {}".format(true_negative(preds, datays, 2)))
    # logger.info("false_positive: {}".format(false_positive(preds, datays, 2)))
    # logger.info("false_negative: {}".format(false_negative(preds, datays, 2)))
    TPR = round(TP / (TP + FN), 10)
    logger.info("tpr: {}".format(TPR))
    FPR = round(FP / (FP + TN), 10)
    logger.info("fpr: {}".format(FPR))
    FNR = round(FN / (TP + FN), 10)
    logger.info("fnr: {}".format(FNR))
    P = round(TP / (TP + FP), 10)
    # logger.info("Precision: {}".format(P))
    F1 = round(2 * P * TPR / (P + TPR), 10)
    logger.info("f1: {}".format(F1))
    logger.info("accuracy: {}".format(accuracy(preds, datays)))
    # print('AUC: %.8f' % metrics.roc_auc_score(preds.cpu(), datays.cpu()))
    # prauc = metrics.precision_recall_curve(datays.cpu(), preds.cpu())
    # logger.info('pr auc: %.8f' % prauc)
    curauc = metrics.roc_auc_score(datays.cpu(), preds.cpu())
    logger.info('AUC: %.8f' % curauc)



    # logger.info('AUC: %.8f' % metrics.precision_recall_curve(datays.cpu(),preds.cpu()))
    return correct / len(loader.dataset), curauc, FPR, F1, FNR, P, model


def main():
    # dataset = dataset.shuffle()
    n = len(dataset) // 5
    train_dataset_list = list()
    test_dataset_list = list()
    test_dataset = dataset[:n]
    train_dataset = dataset[n:]
    test_dataset_list.append(test_dataset)
    train_dataset_list.append(train_dataset)

    test_dataset = dataset[n:2 * n]
    train_dataset = dataset[:n] + dataset[2 * n:]
    test_dataset_list.append(test_dataset)
    train_dataset_list.append(train_dataset)

    test_dataset = dataset[2*n:3*n]
    train_dataset = dataset[:2*n]+dataset[3*n:]
    test_dataset_list.append(test_dataset)
    train_dataset_list.append(train_dataset)

    test_dataset = dataset[3*n:4*n]
    train_dataset = dataset[:3*n]+dataset[4*n:]
    test_dataset_list.append(test_dataset)
    train_dataset_list.append(train_dataset)

    test_dataset = dataset[4*n:]
    train_dataset = dataset[:4*n]
    test_dataset_list.append(test_dataset)
    train_dataset_list.append(train_dataset)

    aucList = list()
    accList = list()
    fprList = list()
    fnrList = list()
    plist = list()
    f1list = list()
    kfcount = 1
    for idx in range(5):
        logger.info("kfcount: {}".format(kfcount))
        kfcount = kfcount + 1
        train_dataset = train_dataset_list[idx]
        test_dataset = test_dataset_list[idx]
        test_loader = DataLoader(test_dataset, batch_size=60)
        train_loader = DataLoader(train_dataset, batch_size=60)
        best_auc = 0
        best_acc = 0
        best_f1 = 0
        best_fpr = 1
        best_fnr = 1
        best_p = 0
        best_tpr = 0

        model = Net().to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)

        for epoch in range(1, 51):
            logger.info("Epoch: {:03d}------------>start".format(epoch))
            loss, model, optimizer = train(epoch, model, train_loader, optimizer, train_dataset)
            logger.info("train---------------:")
            train_acc, trauc, trfpr, trf1, trfnr, trp, model = test(train_loader, model)
            logger.info("test--------------:")
            test_acc, teauc, tefpr, tef1, tefnr, tep, model = test(test_loader, model)
            if (best_auc < teauc):
                best_auc = teauc
                best_fpr = tefpr
                best_fnr = tefnr
                best_acc = test_acc
                best_p = tep
                best_f1 = tef1
                best_tpr = 1 - best_fnr

            # if (best_acc < test_acc):
            #     best_acc = test_acc
            # if(best_f1<tef1):
            #     best_f1 = tef1
            # if(best_fnr>tefnr):
            #     best_fnr = tefnr
            # if(best_fpr>tefpr):
            #     best_fpr = tefpr
            logger.info('Epoch: {:03d}, Loss: {:.5f}, Train Acc: {:.5f}, Test Acc: {:.5f}'.
                        format(epoch, loss, train_acc, test_acc))
        logger.info('best auc: {}'.format(best_auc))
        logger.info('best f1: {}'.format(best_f1))
        logger.info('best fnr: {}'.format(best_fnr))  # 漏报率
        logger.info('best tpr: {}'.format(best_tpr))  # 漏报率
        logger.info('best fpr: {}'.format(best_fpr))
        logger.info('best acc: {}'.format(best_acc))
        # logger.info('best Precision: {}'.format(best_p))
        aucList.append(best_auc)
        fnrList.append(best_fnr)
        fprList.append(best_fpr)
        accList.append(best_acc)
        plist.append(best_p)
        f1list.append(best_f1)
    logger.info("399 cfg result: ")
    logger.info('auc: {}'.format(np.mean(aucList)))
    logger.info('f1: {}'.format(np.mean(f1list)))
    logger.info('fnr: {}'.format(np.mean(fnrList)))  # 漏报率
    logger.info('tpr: {}'.format(1-np.mean(fnrList)))  # 漏报率
    logger.info('fpr: {}'.format(np.mean(fprList)))
    logger.info('acc: {}'.format(np.mean(accList)))
    # logger.info('precision: {}'.format(np.mean(plist)))

if __name__ == '__main__':
    main()
