# -*- coding: UTF-8 -*-
import pymongo
from bson import ObjectId

MONGODB_HOST = '117.50.37.209'
MONGODB_PORT = 27027
conn = pymongo.MongoClient('{}:{}'.format(MONGODB_HOST, MONGODB_PORT))

DATEBASE = 'lxj_app'
TABLENAME = 'collector'

table = conn[DATEBASE][TABLENAME]

result = table.find({})

for item in result:
    print(item)


