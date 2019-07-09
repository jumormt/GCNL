import os

path = "/Users/chengxiao/Downloads/CWE-691/result/raw_result"

jsonList = list()
for dirpath,dirnames,filenames in os.walk(path):

    for file in filenames:#遍历完整文件
        fullpath=os.path.join(dirpath,file)
        jsonList.append(fullpath)

# import random
# random.shuffle(jsonList)

import json
outputpath="/Users/chengxiao/Downloads/CWE-691/result/raw_result_out/"

for i in range(len(jsonList)):
    curjsonPath = jsonList[i]
    with open(curjsonPath, 'r', encoding="utf-8") as f:
        j = json.load(f)
        outjson = dict()
        outjson["nodes"] = j["nodes"]
        outjson["edges"] = j["edges"]
        outjson["target"] = j["target"]
        with open(outputpath+"{}.json".format(i), "w", encoding="utf-8") as ff:
            json.dump(outjson, ff)

