import csv
# https://blog.csdn.net/john_xyz/article/details/79208564
from gensim.test.utils import common_texts
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from sklearn.model_selection import train_test_split
# import torch
# from torch_geometric.data import Data

# data.x: Node feature matrix with shape [num_nodes, num_node_features]
# data.edge_index: Graph connectivity in COO format with shape [2, num_edges] and type torch.long
# data.edge_attr: Edge feature matrix with shape [num_edges, num_edge_features]
# data.y: Target to train against (may have arbitrary shape)
# data.pos: Node position matrix with shape [num_nodes, num_dimensions]

import os
import json

import argparse

from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import recall_score
from sklearn import svm
import numpy as np
from sklearn.model_selection import ShuffleSplit


def getGraphs(inputpath, inputCSVPath):
    '''得到graphs，key为图向量x和标记target

    :param inputpath: graphs-json
    :param inputCSVPath: graphs-vec
    :return: graph
    '''

    graphs = dict()

    # get target
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
                # graphs[file]["edgeList"] = edgeList
                # graphs[file]["nodeStrList"] = nodeStrList

    # get x
    with open(inputCSVPath, encoding="utf-8") as f:
        f_csv = csv.reader(f)
        headers = next(f_csv)
        for row in f_csv:
            name = row[0]
            vec = row[1:]
            graphs[name + ".json"]['x'] = vec
    return graphs


def getXY(graphs):
    '''

    :param graphs: getGraph得到的图
    :return: X，Y-list
    '''
    X = list()
    Y = list()

    for graph in graphs:
        X.append(graphs[graph]['x'])
        Y.append(graphs[graph]['target'])

    return X, Y


def main(args):

    graphs = getGraphs(args.input_json_path, args.input_csv_path)
    X, Y = getXY(graphs)

    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.1, random_state=0)

    cv = ShuffleSplit(n_splits=5, test_size=0.1, random_state=0)
    scoring = ['precision_macro', 'recall_macro']
    clf = svm.SVC(kernel='linear', C=1, random_state=0)
    scores = cross_val_score(clf, X, Y ,cv = cv)

    print()


def parameter_parser():
    """
    A method to parse up command line parameters. By default it gives an embedding of the partial NCI1 graph dataset.
    The default hyperparameters give a good quality representation without grid search.
    Representations are sorted by ID.
    """

    parser = argparse.ArgumentParser(description="Run Graph2Vec.")

    parser.add_argument("--input-json-path",
                        nargs="?",
                        default="/Users/chengxiao/Desktop/VulDeepecker/资料/project/CGDSymbolization/src/main/resources/result",
                        help="Input folder with jsons.")
    parser.add_argument("--input-csv-path",
                        nargs="?",
                        default="/Users/chengxiao/Downloads/graph2vec-master/features/test3.csv",
                        help="Input csv file which contains graphvecs.")
    parser.add_argument("--epochs",
                        type=int,
                        default=10,
                        help="Number of epochs. Default is 10.")



    return parser.parse_args()



if __name__ == '__main__':
    args = parameter_parser()
    main(args)
