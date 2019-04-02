# https://blog.csdn.net/john_xyz/article/details/79208564
from gensim.test.utils import common_texts
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
# import torch
# from torch_geometric.data import Data
inputpath = "/Users/chengxiao/Desktop/VulDeepecker/资料/project/CGDSymbolization/src/main/resources/result"

# data.x: Node feature matrix with shape [num_nodes, num_node_features]
# data.edge_index: Graph connectivity in COO format with shape [2, num_edges] and type torch.long
# data.edge_attr: Edge feature matrix with shape [num_edges, num_edge_features]
# data.y: Target to train against (may have arbitrary shape)
# data.pos: Node position matrix with shape [num_nodes, num_dimensions]

import os
import json
graphs = dict()

for dirpath,dirnames,filenames in os.walk(inputpath):
    # for dir in dirnames:
    #     fulldir = os.path.join(dirpath,dir)
    #     print(fulldir)

    for file in filenames:#遍历完整文件
        fullpath=os.path.join(dirpath,file)
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
        documents.append(TaggedDocument(nodes[i], [graph+"_tag"+str(i)]))
        # curGraph["nodeTagList"].append(count)
        # count = count+1

model = Doc2Vec(documents, vector_size=5, window=2, min_count=1, workers=4)
for graph in graphs:
    curGraph = graphs[graph]
    nodes = curGraph["nodeStrList"]
    nodeVecList = list()
    for i in range(len(nodes)):
        # nodeVecList.append(model.docvecs[str(curGraph["nodeTagList"][i])])
        nodeVecList.append(model.docvecs[graph+"_tag"+str(i)])
    curGraph["nodeVecList"] = nodeVecList

# Read data into huge `Data` list.
data_list = list()
for graphk in graphs:
    curGraph = graphs[graphk]
    edge_index_v = curGraph["edgeList"]
    x_v = curGraph["nodeVecList"]
    y = curGraph["target"]
    # edge_index = torch.tensor(edge_index_v, dtype=torch.long)
    # x = torch.tensor(x_v, dtype = torch.float)
    # data = Data(x=x, edge_index=edge_index.t().contiguous(), y=y)
    # data_list.append(data)
    print()