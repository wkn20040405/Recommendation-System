import numpy as np
import pandas as pd
import random
import time
import os
import pymongo
from math import *
import re
import difflib
import jieba.posseg as pseg
from gensim import corpora, models, similarities
from pprint import pprint #pprint 数据结构更加完整

if __name__ == '__main__':
    Results = pd.read_csv("csv/Results.csv", index_col=0)
    p_dictionary = pd.read_csv("csv/dictionary.csv", index_col=0)
    c_dictionary = pd.read_csv("csv/c_dictionary.csv", index_col="名字")
    c_dictionary.columns = ["product_id"]
    MONGODB_HOST = '117.50.37.209'
    MONGODB_PORT = 27027
    conn = pymongo.MongoClient('{}:{}'.format(MONGODB_HOST, MONGODB_PORT))
    DATEBASE = 'lxj_app'
    TABLENAME = 'Recommendation'
    table = conn[DATEBASE][TABLENAME]

    for index, rows in Results.iterrows():
        insert = {"user_id": index}
        res = table.find_one(insert)
        for j in range(0, Results.shape[1]):
            temp = "Recommend_product{}".format(j+1)
            Recommend_product_id = c_dictionary.loc[p_dictionary.loc[rows.iloc[j], "名字"], "product_id"]
            insert[temp] = Recommend_product_id
            if res is not None:
                res[temp] = Recommend_product_id
        if res is not None:
            result = table.update({"user_id": index}, res)
            print(result)
        else:
            post_id = table.insert_one(insert).inserted_id
            print("post id is:", post_id)
