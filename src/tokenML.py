import json
import os
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from gensim.models import Word2Vec


import logging
import sys

import numpy as np
np.random.seed(1337)  # for reproducibility
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, Embedding, LSTM, TimeDistributed, Input, Bidirectional, Activation
from keras.layers import Conv1D, MaxPooling1D
from keras.utils import np_utils
from keras import backend as K
from keras.preprocessing import sequence

from sklearn import metrics
from sklearn.model_selection import KFold



logger = logging.getLogger(__name__)
logger.setLevel(level=logging.INFO)
file_handler = logging.FileHandler(
    '/home/cry/chengxiao/dataset/svf-related/CWE476/476_tk_result.txt')
file_handler.setLevel(level=logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

# StreamHandler
stream_handler = logging.StreamHandler(sys.stdout)
stream_handler.setFormatter(formatter)
stream_handler.setLevel(level=logging.INFO)
logger.addHandler(stream_handler)

jsonPath = "/Users/chengxiao/Downloads/CWE-691/token_sym/airpcap_loader.c.exm_0_funcline113_sym.json"
# jsonDir = "/Users/chengxiao/Downloads/CWE-840/token_sym/"
# jsonDir = "/home/cry/chengxiao/dataset/tscanc/CWE-21/token_sym/"
jsonDir = "/home/cry/chengxiao/dataset/svf-related/CWE476/token-sym/"

# 全局变量
batch_size = 60
nb_classes = 1
epochs = 12
# input vec dimensions
tk_rows, tk_cols = 1, 64
# number of convolutional filters to use
nb_filters = 32
# size of pooling area for max pooling
pool_size = 4
# convolution kernel size
kernel_size = (3, 3)

HIDDEN_LAYER_SIZE = 64

def getFuncDict(jsonDir):
    '''

    :param jsonDir: func jsons dir path
    :return: {jsonpath:{funcLines:[],target:0 or 1,funcVec:[1*64]}}
    '''
    funcDic = dict()

    for dirpath, dirnames, filenames in os.walk(jsonDir):
        for file in filenames:  # 遍历完整文件
            fullpath = os.path.join(dirpath, file)
            # print (fullpath)
            with open(fullpath, 'r', encoding="utf-8") as f:
                curjson = json.load(f)
                if ("target" not in curjson.keys()):
                    continue
                funcLines = curjson["funcLines"]

                target = curjson["target"]
                funcDic[file] = dict()
                funcDic[file]["target"] = target
                funcDic[file]["funcLines"] = funcLines

    # documents = list()
    #
    # for func in funcDic:
    #     curFunc = funcDic[func]
    #     curFuncLines = curFunc["funcLines"]
    #     curFuncLinesWords = list()
    #     for line in curFuncLines:
    #         curFuncLinesWords.extend(line.split())
    #     documents.append(TaggedDocument(curFuncLinesWords, [func]))
    #
    # model = Doc2Vec(documents, vector_size=64, min_count=5, workers=8, window=8, dm=0, alpha=0.025,
    #                 epochs=50)
    #
    # for func in funcDic:
    #     curFunc = funcDic[func]
    #     curFunc["funcVec"] = model.docvecs[func]

    sentences = list()
    for func in funcDic:
        curFunc = funcDic[func]
        curFuncLines = curFunc["funcLines"]
        curFuncLinesWords = list()
        for line in curFuncLines:
            curFuncLinesWords.extend(line.split())
        sentences.append(curFuncLinesWords)

    model = Word2Vec(sentences, min_count=0, size=1)
    maxindex = len(model.wv.index2entity)

    for func in funcDic:
        curFunc = funcDic[func]
        curFuncLines = curFunc["funcLines"]
        funcVec = list()
        for line in curFuncLines:
            wordL = line.split()
            for word in wordL:
                funcVec.append(model.wv.vocab[word].index + 1)

        curFunc["funcVec"] = funcVec


    return funcDic, maxindex

# TODO(Jumormt): 建立双向lstm模型
def createModel(inputLen, maxIndex, wordSize):
    """创建神经网络.

            创建双向lstm神经网络，包括输入层，循环层，输出层等。

            Args:
                inputLen: 输入向量维度
                maxIndex：词向量最大index
                wordSize: 词向量维度（暂时为1）

            Returns:
                model:网络模型

            Raises:
                IOError: An error occurred accessing the bigtable.Table object.
            """
    # logger.info('创建blstm神经网络...'.encode('utf-8'))
    model = Sequential()
    # model.add(Embedding(maxIndex + 1, wordSize, input_length=inputLen, mask_zero=True))
    model.add(Embedding(maxIndex + 1, wordSize, input_length=inputLen))
    model.add(Bidirectional(LSTM(HIDDEN_LAYER_SIZE, dropout=0.5, recurrent_dropout=0.5)))
    model.add(Dense(1))
    model.add(Activation("sigmoid"))
    model.compile(loss='binary_crossentropy', optimizer='adamax', metrics=['binary_accuracy'])
    # logger.info('创建blstm神经网络成功！'.encode('utf-8'))
    return model

def main():
    funcDic,maxIndex = getFuncDict(jsonDir)
    X = list()
    Y = list()
    for func in funcDic:
        curFunc = funcDic[func]
        X.append(curFunc["funcVec"])
        Y.append(curFunc["target"])
    X = np.array(X)
    Y = np.array(Y)
    X = sequence.pad_sequences(X, 64)

    kf = KFold(n_splits=10, shuffle=True)
    tprList = list()
    fprList = list()
    fnrList = list()
    f1List = list()
    AUCList = list()
    accuracyList = list()
    plist = list()
    kfcount = 1

    for train_idx, test_idx in kf.split(X):
        logger.info("split: {}".format(kfcount))
        kfcount = kfcount + 1

        X_train, Y_train = X[train_idx], Y[train_idx]
        X_test, Y_test = X[test_idx], Y[test_idx]

        # # 根据不同的backend定下不同的格式
        # if K.image_dim_ordering() == 'th':
        #     X_train = X_train.reshape(X_train.shape[0], 1, tk_cols)
        #     X_test = X_test.reshape(X_test.shape[0], 1, tk_cols)
        #     input_shape = (1, tk_cols)
        # else:
        #     X_train = X_train.reshape(X_train.shape[0],  tk_cols, 1)
        #     X_test = X_test.reshape(X_test.shape[0],  tk_cols, 1)
        #     input_shape = ( tk_cols, 1)
        #
        #
        # # 构建模型
        # model = Sequential()
        # """
        # model.add(Convolution2D(nb_filters, kernel_size[0], kernel_size[1],
        #                         border_mode='same',
        #                         input_shape=input_shape))
        # """
        # model.add(Convolution1D(nb_filters, (kernel_size[0], kernel_size[1]),
        #                         padding='same',
        #                         input_shape=input_shape))  # 卷积层1
        # model.add(Activation('relu'))  # 激活层
        # model.add(Convolution1D(nb_filters, (kernel_size[0], kernel_size[1])))  # 卷积层2
        # model.add(Activation('relu'))  # 激活层
        # model.add(MaxPooling1D(pool_size=pool_size))  # 池化层
        # model.add(Dropout(0.25))  # 神经元随机失活
        # model.add(Flatten())  # 拉成一维数据
        # model.add(Dense(128))  # 全连接层1
        # model.add(Activation('relu'))  # 激活层
        # model.add(Dropout(0.5))  # 随机失活
        # model.add(Dense(nb_classes))  # 全连接层2
        # model.add(Activation('sigmoid'))  # Softmax评分
        #
        # # 编译模型
        # model.compile(loss='binary_crossentropy',
        #               optimizer='adamax',
        #               metrics=['binary_accuracy'])
        #
        #
        # # 训练模型
        # model.fit(X_train, Y_train, batch_size=batch_size, epochs=epochs,
        #           verbose=1, validation_data=(X_test, Y_test))
        # # 评估模型
        # score = model.evaluate(X_test, Y_test, verbose=0)
        # print('Test score:', score[0])
        # print('Test accuracy:', score[1])
        from keras.layers import GlobalMaxPooling1D
        model = Sequential()
        model.add(Embedding(maxIndex+1, 64, input_length=64))
        model.add(Conv1D(nb_filters,
                         3,
                         padding='valid',
                         activation='relu',
                         strides=1))
        model.add(GlobalMaxPooling1D())
        model.add(Dense(1))
        model.add(Activation("sigmoid"))
        model.add(Dropout(0.2))
        model.compile(loss='binary_crossentropy', optimizer='adamax', metrics=['binary_accuracy'])
        model.fit(X_train, Y_train, batch_size=64, epochs=50, verbose=2)

        y_result = model.predict_classes(X_test)
        y_result_prob = model.predict_proba(X_test)
        # print('f1: %.8f' % metrics.f1_score(y_test, y_result))
        TP = 0
        TN = 0
        FP = 0
        FN = 0
        for i in range(len(y_result)):
            if (Y_test[i] == 1):
                if (y_result[i] == 1):
                    TP = TP + 1
                else:
                    FN = FN + 1
            else:
                if (y_result[i] == 1):
                    FP = FP + 1
                else:
                    TN = TN + 1
        TPR = round(TP / (TP + FN), 10)
        logger.info("tpr: {}".format(TPR))
        FPR = round(FP / (FP + TN), 10)
        logger.info("fpr: {}".format(FPR))
        FNR = round(FN / (TP + FN), 10)
        logger.info("fnr: {}".format(FNR))
        if (TP + FP != 0):
            P = round(TP / (TP + FP), 10)
            plist.append(P)
            logger.info("Precision:{}".format(P))
            f1 = round(2 * P * TPR / (P + TPR), 10)
            logger.info("f1: {}".format(f1))
            f1List.append(f1)

        accuracy = metrics.accuracy_score(Y_test, y_result)
        logger.info("accuracy: {}".format(accuracy))
        # AUC = metrics.roc_auc_score(Y_test, y_result)
        AUC = metrics.roc_auc_score(Y_test, y_result_prob)
        logger.info('AUC: %.8f' % AUC)
        tprList.append(TPR)
        fprList.append(FPR)
        fnrList.append(FNR)
        accuracyList.append(accuracy)
        AUCList.append(AUC)

    logger.info("119 tk final result: ")
    logger.info("tpr: {}".format(np.mean(tprList)))
    logger.info("fpr: {}".format(np.mean(fprList)))
    logger.info("fnr: {}".format(np.mean(fnrList)))
    logger.info("accuracy: {}".format(np.mean(accuracyList)))
    logger.info("f1: {}".format(np.mean(f1List)))
    logger.info("Precision: {}".format(np.mean(plist)))
    logger.info("AUC: {}".format(np.mean(AUCList)))

    print("end")

if __name__ == '__main__':
    main()