import os
import json,string,re
from collections import defaultdict
import numpy as np
from numpy import linalg as la


TRAIN_PATH = "data/split/train"
TEST_Path = "data/split/test"
paths = {"train":TRAIN_PATH,"test":TEST_Path}
labels = ["1","2","3","4","5"]

translator = str.maketrans('', '', string.punctuation)
for s in ["train","test"]:
    for l in labels:
        label = s + "_" + l
        print (label)
        f = open("data/Doc2vec/"+label,encoding='utf-8',mode="a")
        with open(os.path.join(paths[s], l+".json"), 'r') as doc:
            for line in doc:
               content = json.loads(line)
               text = content['text'].lower().replace("\n", " ")
               t = re.sub('http\S+', '', text)
               f.write(t.translate(translator)+"\n")







