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
from sklearn.model_selection import cross_validate
from sklearn.model_selection import cross_val_predict
from sklearn import metrics
from sklearn.model_selection import cross_val_score
from sklearn.metrics.scorer import make_scorer
from sklearn.metrics import recall_score
from sklearn import svm
import numpy as np
from sklearn.model_selection import ShuffleSplit

from collections import Counter
from imblearn.over_sampling import SMOTE, ADASYN  # 过抽样处理库SMOTE
from imblearn.under_sampling import RandomUnderSampler  # 欠抽样处理库RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler  # 欠抽样处理库RandomUnderSampler
from imblearn.ensemble import EasyEnsemble  # 简单集成方法EasyEnsemble

from imblearn.combine import SMOTEENN
from imblearn.combine import SMOTETomek

from sklearn import preprocessing


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
    得到经过均衡处理后的xy，并对x进行预处理
    :param graphs: getGraph得到的图
    :return: X，Y-list
    '''
    X = list()
    Y = list()

    for graph in graphs:
        X.append(graphs[graph]['x'])
        Y.append(graphs[graph]['target'])

    X = np.array(X).astype('float64')
    Y = np.array(Y)

    # 结合采样
    # https://blog.csdn.net/kizgel/article/details/78553009
    smote_tomek = SMOTETomek(random_state=0)
    X_resampled, y_resampled = smote_tomek.fit_sample(X, Y)
    print(sorted(Counter(y_resampled).items()))

    # 预处理 归一化正则化
    scaler = preprocessing.StandardScaler().fit(X_resampled)
    X_train_transformed = scaler.transform(X_resampled)

    return X_train_transformed, y_resampled

def svm_cross_validation(train_x, train_y):
    '''
    svm调参
    :param train_x:
    :param train_y:
    :return:
    '''
    from sklearn.model_selection import GridSearchCV
    from sklearn.svm import SVC
    model = SVC(kernel='rbf', probability=True)
    param_grid = {'C': [1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100, 1000, 10000], 'gamma': [0.01, 0.0001]}
    grid_search = GridSearchCV(model, param_grid, n_jobs = 8, verbose=1)
    grid_search.fit(train_x, train_y)
    best_parameters = grid_search.best_estimator_.get_params()
    for para, val in list(best_parameters.items()):
        print(para, val)
    model = SVC(kernel='rbf', C=best_parameters['C'], gamma=best_parameters['gamma'], probability=True)
    # model.fit(train_x, train_y)
    return model


def main(args):
    graphs = getGraphs(args.input_json_path, args.input_csv_path)
    X, Y = getXY(graphs)

    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.1, random_state=0)

    cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=0)
    # scoring = ['precision_macro', 'recall_macro']

    # clf.fit(X_train, y_train)
    # a = clf.predict(X_test)
    # ros = RandomOverSampler(random_state=0)
    # X_resampled, y_resampled = ros.fit_sample(X, Y)
    # sorted(Counter(y_resampled).items())


    clf = svm_cross_validation(X, Y)

    scores = cross_val_score(clf, X, Y, cv=cv)
    for i in range(len(scores)):
        print("Accuracy%d: %0.5f" % (i, scores[i]))
    print("Accuracy average: %0.5f (+/- %0.5f)" % (scores.mean(), scores.std() * 2))
    # scores = cross_validate(clf, X, Y, scoring=scoring,
    #                         cv=5, return_train_score=True)
    # predicted = cross_val_predict(clf, X, Y, cv=10)

    # print(metrics.accuracy_score(Y, predicted))

    print("end")


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
                        default="/Users/chengxiao/Downloads/graph2vec-master/features/test.csv",
                        help="Input csv file which contains graphvecs.")
    parser.add_argument("--epochs",
                        type=int,
                        default=10,
                        help="Number of epochs. Default is 10.")

    return parser.parse_args()


if __name__ == '__main__':
    args = parameter_parser()
    main(args)
