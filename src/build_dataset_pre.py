# https://blog.csdn.net/john_xyz/article/details/79208564
from gensim.test.utils import common_texts
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
# import torch
# from torch_geometric.data import Data
inputpath = "/Users/chengxiao/Desktop/VulDeepecker/资料/project/CGDSymbolization/src/main/resources/result"
# inputpath = "/home/cry/chengxiao/dataset/SARD.2019-02-28-22-07-31/addswitch/result_sym"
# inputpath = "/Users/chengxiao/Downloads/SARD.2019-02-28-22-07-31/addswitch/result_sym"

# data.x: Node feature matrix with shape [num_nodes, num_node_features]
# data.edge_index: Graph connectivity in COO format with shape [2, num_edges] and type torch.long
# data.edge_attr: Edge feature matrix with shape [num_edges, num_edge_features]
# data.y: Target to train against (may have arbitrary shape)
# data.pos: Node position matrix with shape [num_nodes, num_dimensions]

import os
import json
import random

# data.x: Node feature matrix with shape [num_nodes, num_node_features]
# data.edge_index: Graph connectivity in COO format with shape [2, num_edges] and type torch.long
# data.edge_attr: Edge feature matrix with shape [num_edges, num_edge_features]
# data.y: Target to train against (may have arbitrary shape)
# data.pos: Node position matrix with shape [num_nodes, num_dimensions]
#
# import os
# import json
graphs = dict()

for dirpath, dirnames, filenames in os.walk(inputpath):
    # for dir in dirnames:
    #     fulldir = os.path.join(dirpath,dir)
    #     print(fulldir)

    for file in filenames:  # 遍历完整文件
        fullpath = os.path.join(dirpath, file)
        # print (fullpath)
        with open(fullpath, 'r', encoding="utf-8") as f:
            curjson = json.load(f)
            if ("target" not in curjson.keys()):
                continue
            nodeStrList = curjson["nodes"]

            target = curjson["target"]
            edgeList = curjson["edges"]
            graphs[file] = dict()
            graphs[file]["target"] = target
            graphs[file]["edgeList"] = edgeList
            graphs[file]["nodeStrList"] = nodeStrList

documents = list()
# count = 0
for graph in graphs:
    curGraph = graphs[graph]
    nodes = curGraph["nodeStrList"]
    # curGraph["nodeTagList"] = list()
    for i in range(len(nodes)):
        documents.append(TaggedDocument(nodes[i].split(), [graph + "_tag" + str(i)]))
        # curGraph["nodeTagList"].append(count)
        # count = count+1

model = Doc2Vec(documents, vector_size=128, window=0, min_count=5, workers=4, dm=0, sample=0.0001, alpha=0.025,
                epochs=10)

for graph in graphs:
    curGraph = graphs[graph]
    nodes = curGraph["nodeStrList"]
    nodeVecList = list()
    for i in range(len(nodes)):
        # nodeVecList.append(model.docvecs[str(curGraph["nodeTagList"][i])])
        nodeVecList.append(model.docvecs[graph + "_tag" + str(i)])
    curGraph["nodeVecList"] = nodeVecList

# Read data into huge `Data` list.
data_list = list()

# TODO:解决样本不平衡的问题
from imblearn.combine import SMOTETomek
from imblearn.over_sampling import RandomOverSampler
from collections import Counter

# smote_tomek = RandomOverSampler(random_state=0)
X = list()
Y = list()
X_0 = list()
X_1 = list()
for graphk in graphs:
    curGraph = graphs[graphk]
    # X.append(curGraph)
    y = curGraph["target"]
    if(y==0):
        X_0.append(curGraph)
    else:
        X_1.append(curGraph)
    # Y.append(y)
# X_resampled, y_resampled = smote_tomek.fit_sample(X, Y)
X_resampled, y_resampled = X,Y
a = sorted(Counter(y_resampled).items())
num = abs(len(X_0)-len(X_1))
if(len(X_0)>len(X_1)):
    # 扩展x1

    for i in range(num):
        X_1.append(random.choice(X_1))

else:
    #扩展x0
    for i in range(num):
        X_0.append(random.choice(X_0))

X.extend(X_0)
Y.extend(len(X_0)*[0])
X.extend(X_1)
Y.extend(len(X_1)*[1])
# for graphk in graphs:
for curGraph in X_resampled:
    # curGraph = graphs[graphk]
    # curGraph = curGraph[0]
    edge_index_v = curGraph["edgeList"]

    x_v = curGraph["nodeVecList"]
    # y = torch.tensor([curGraph["target"]], dtype=torch.long)
    #
    # x = torch.tensor(x_v, dtype=torch.float)
    # if (len(edge_index_v) != 0):
    #     edge_index = torch.tensor(edge_index_v, dtype=torch.long)
    #     data = Data(x=x, edge_index=edge_index.t().contiguous(), y=y)
    # else:
    #     edge_index = torch.tensor([], dtype=torch.long)
    #     data = Data(edge_index=edge_index,x=x, y=y)
    # print(edge_index.t().contiguous())
    # print(data)
    # data_list.append(data)