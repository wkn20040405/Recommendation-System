#-*- coding: UTF-8 -*-
import pymongo
from bson import ObjectId
import pandas as pd

MONGODB_HOST = '117.50.37.209'
MONGODB_PORT = 27027
conn = pymongo.MongoClient('{}:{}'.format(MONGODB_HOST, MONGODB_PORT))

DATEBASE = 'lxj_app'
TABLENAME = 'material'

table = conn[DATEBASE][TABLENAME]

temp = table.find({})

result = table.find({'product_id':{'$exists':1}}, {'product_id':1,'project_name':1})

p_dictionary = pd.DataFrame()
c_dictionary = pd.DataFrame()

for item in result:
    a = item
    pid = str(item['_id'])
    cid = str(item['product_id'])
    name = item['project_name']
    p_dictionary = p_dictionary.append(pd.DataFrame({"名字": [name]}, index=[pid]), sort=True).sort_index()
    c_dictionary = c_dictionary.append(pd.DataFrame({"名字": [name]}, index=[cid]), sort=True).sort_index()

p_dictionary.to_csv("csv/dictionary.csv")
c_dictionary.to_csv("csv/c_dictionary.csv")

