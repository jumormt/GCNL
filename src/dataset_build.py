import torch
from torch_geometric.data import Data

#
# edge_index = torch.tensor([[0, 1],
#                            [2, 3],
#                            [4, 5],
#                            [6,7]
#                     ], dtype=torch.long)
# data = Data(edge_index=edge_index.t().contiguous())
from torch_geometric.data import DataLoader
# loader = DataLoader(data, batch_size=2, shuffle=True)
# for i in loader:
#     b = 1

import torch
from torch_geometric.data import InMemoryDataset
from gensim.test.utils import common_texts
from gensim.models.doc2vec import Doc2Vec, TaggedDocument

import os
import json


class Test787DatasetTest(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super(Test787DatasetTest, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ['some_file_1', 'some_file_2']
        # pass

    @property
    def processed_file_names(self):
        return ['data.pt']

    def download(self):
        # Download to `self.raw_dir`.
        pass

    def process(self):
        # Read data into huge `Data` list.
        # from gensim.test.utils import common_texts
        # from gensim.models.doc2vec import Doc2Vec, TaggedDocument
        # import torch
        # from torch_geometric.data import Data
        # inputpath = "/Users/chengxiao/Desktop/VulDeepecker/资料/project/CGDSymbolization/src/main/resources/result"
        inputpath = "/home/cry/chengxiao/dataset/SARD.2019-02-28-22-07-31/addswitch/result_sym"

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
        # from imblearn.combine import SMOTETomek
        # from collections import Counter
        #
        # smote_tomek = SMOTETomek(random_state=0)
        # X = list()
        # Y = list()
        # for graphk in graphs:
        #     curGraph = graphs[graphk]
        #     X.append([curGraph])
        #     y = curGraph["target"]
        #     Y.append(y)
        # X_resampled, y_resampled = smote_tomek.fit_sample(X, Y)
        # # X_resampled, y_resampled = X,Y
        # print(sorted(Counter(y_resampled).items()))

        for graphk in graphs:
        # for curGraph in X_resampled:
            curGraph = graphs[graphk]
            # curGraph = curGraph[0]
            edge_index_v = curGraph["edgeList"]

            x_v = curGraph["nodeVecList"]
            y = torch.tensor([curGraph["target"]], dtype=torch.long)

            x = torch.tensor(x_v, dtype=torch.float)
            if (len(edge_index_v) != 0):
                edge_index = torch.tensor(edge_index_v, dtype=torch.long)
                data = Data(x=x, edge_index=edge_index.t().contiguous(), y=y)
            else:
                edge_index = torch.tensor([], dtype=torch.long)
                data = Data(edge_index=edge_index,x=x, y=y)
            # print(edge_index.t().contiguous())
            print(data)
            data_list.append(data)

        # edge_index = torch.tensor([[0, 1],
        #                            [1,2]
        #                            ], dtype=torch.long)
        # y1 = torch.tensor([0], dtype=torch.long)
        # y2 = torch.tensor([1], dtype=torch.long)
        #
        # x = torch.tensor([[-1], [0], [1]], dtype=torch.float)
        # data1 = Data(x = x,edge_index=edge_index.t().contiguous(), y=y1)
        # data2 = Data(x=x,edge_index=edge_index.t().contiguous(), y=y2)
        #
        # data_list = [data1, data2, data1, data2]

        # edge_index = torch.tensor([[]
        #                            ], dtype=torch.long)
        # y1 = torch.tensor([0], dtype=torch.long)
        # y2 = torch.tensor([1], dtype=torch.long)
        #
        # x = torch.tensor([[-1]], dtype=torch.float)
        # data1 = Data(x=x, edge_index=edge_index.t().contiguous(), y=y1)
        # data2 = Data(x=x, edge_index=edge_index.t().contiguous(), y=y2)
        #
        # data_list = [data1, data2, data1, data2]

        # if self.pre_filter is not None:
        #     data_list [data for data in data_list if self.pre_filter(data)]
        #
        # if self.pre_transform is not None:
        #     data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])


if __name__ == '__main__':
    dataset = Test787DatasetTest(root="/home/cry/chengxiao/dataset/Test787DatasetTest")
    loader = DataLoader(dataset, batch_size=2, shuffle=True)
    # for i in loader:
    #     b = 1
    # edge_index = torch.tensor([[0, 1],
    #                            ], dtype=torch.long)
    # y = torch.tensor([0], dtype=torch.long)
    # data1 = Data(edge_index=edge_index.t().contiguous(), y=y)
    print("end")
