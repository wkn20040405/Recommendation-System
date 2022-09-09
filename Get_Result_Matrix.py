from LatentFactorModel import LatentFactorModel
from Dataprocessing import DataPreprocessing
import pandas as pd
import numpy as np
from bson import ObjectId


def Recommendation_one_Result(userid, recommend_number, final, corr):
    temp_id = orignial_data.loc[userid] == 0
    temp_data = final.loc[userid][temp_id]
    temp_dictionary = {}
    i = 0
    while i < recommend_number:
        DATA_VALID = True
        temp_id = temp_data.idxmax()
        temp_rating = temp_data[temp_id]
        for j in temp_dictionary.keys():
            if corr.loc[temp_id, temp_dictionary[j][0]] > 0.5:
                DATA_VALID = False
                break
        if DATA_VALID:
            temp_dictionary[i] = [temp_id, temp_rating]
            i += 1
        temp_data[temp_id] = 0
    return temp_dictionary


if __name__ == '__main__':
    recommend_number = 3
    userid = "0056d77609c10ea3c2ba353a0edd0c35251ab25d3d393e10b7496e9bc66c6149"
    final = pd.read_csv("optimization/2020-03-30_16-49-35/final_answer.csv", index_col=0)
    orignial_data = pd.read_csv("csv/rating_original_matrix.csv", index_col=0)
    training_data = pd.read_csv("matrices/random_training", index_col=0)
    user_average = np.load("optimization/2020-03-30_16-49-35/user_average.npy", allow_pickle=True)
    corr = pd.read_csv("csv/corr.csv", index_col=0)
    Q = np.load("optimization/2020-03-30_16-49-35/epoch_19/Q.npy")
    P = np.load("optimization/2020-03-30_16-49-35/epoch_19/P.npy")
    L = pd.read_csv("csv/c_dictionary.csv")
    predict = Q.dot(P)

    record_data = pd.DataFrame(index=final.index)
    for i in range(recommend_number):
        name = "Recommend-product_{}".format(i + 1)
        record_data.loc[:, name] = ""
        #record_data[name] = "fuck_now" #这里不能写record_data.loc[:, name] = pd.Series()

    for j in range(0, record_data.shape[0]):
        temp_dictionary = Recommendation_one_Result(record_data.index[j], recommend_number, final, corr)
        print("Great.{}".format(j))
        for k in range(record_data.shape[1]):
            s = temp_dictionary[k][0]
            record_data.iloc[j, k] = s
            #极力避免使用record_data.iloc[j][k]
            #或者使用record_data.loc[行名，列名]
    record_data.to_csv("csv/Results.csv", encoding='utf_8_sig')
    print(record_data)