import pandas as pd
import numpy as np
from math import *
import re
import matplotlib.pyplot as plt
import difflib
from LatentFactorModel import LatentFactorModel
import warnings

warnings.filterwarnings("ignore")

import jieba.posseg as pseg
from gensim import corpora, models, similarities
from pprint import pprint

try:
    from tqdm import tqdm  # tqdm是一个反应进程的函数

    HAS_TQDM = True
except ModuleNotFoundError as e:
    HAS_TQDM = False

TEST_MODE = False
READ_CORR = False


############################# text process funcs ###############################

def seg_sentence(sentence, stop_words):
    stop_flag = ['x', 'c', 'u', 'd', 'p', 't', 'uj', 'f', 'r']
    sentence_seged = pseg.cut(sentence)
    # sentence_seged = set(sentence_seged)
    outstr = []
    for word, flag in sentence_seged:
        if word not in stop_words and flag not in stop_flag:
            outstr.append(word)
    return outstr


def tfidf_corr(dictionary):
    stop_words = ['「', '」', '（', '）', '(', ')', '-', '•']
    texts = [seg_sentence(seg, stop_words) for seg in dictionary['名字']]
    dictionary = corpora.Dictionary(texts)
    feature_cnt = len(dictionary.token2id.keys())
    # print('dictionary.token2id:')
    # print(dictionary.token2id)
    bow_corpus = [dictionary.doc2bow(text) for text in texts]
    tfidf = models.TfidfModel(bow_corpus)
    tf_corpus = tfidf[bow_corpus]
    index = similarities.SparseMatrixSimilarity(tf_corpus, num_features=feature_cnt)
    print(index[tf_corpus])
    return index


########################### Pre-process funcs ##################################

def calculate_product_correlation(dictionary, sim, extend=True, threshold=0, amplifier=1.0):
    product_correlation = pd.DataFrame()
    if dictionary is not None:
        product_correlation = pd.DataFrame(data=np.eye(len(dictionary)), index=dictionary.index,
                                           columns=dictionary.index)
        if extend:
            if HAS_TQDM:
                progress = tqdm(iterable=dictionary.index, total=len(dictionary.index), desc='Similarity', ncols=100)
            else:
                progress = dictionary.index
            for Id in progress:
                if not HAS_TQDM: print('#', end='', flush=True)
                for col in dictionary.index:
                    match = sim(Id, col)
                    if match > 0.99:
                        product_correlation.loc[Id, col] = 1.0
                    else:
                        product_correlation.loc[Id, col] = max((match - threshold) * amplifier, 0)
            if HAS_TQDM: progress.close()
    return product_correlation


def rectifier(x):
    # return exp(x) / (1.0 + exp(x))
    return log(x + 1)


def similarity(str1, str2):
    return difflib.SequenceMatcher(lambda x: x in ' 「」（）()-•', str1, str2).quick_ratio()


def add_record(data, token, pids, push=True, p_dictionary=None, c_dictionary=None, corr=None):
    '''
    Add record of token:pid to data of type dict
    '''
    # print('#', end='', flush =True)
    ### translate compare id to push id
    if not push and (p_dictionary is not None) and (c_dictionary is not None):
        names = c_dictionary.loc[pids]
        real_pids = p_dictionary.index[p_dictionary['名字'].isin(names['名字'])]
    else:
        real_pids = pids
    try:
        tmp = np.max(corr.loc[real_pids, :], axis=0)
        #PUSH的时候加上一行对应一个元素为1的行向量
        #COMPARE的时候加上一行对应多个元素为1的行向量
        if token not in data.index:
            data.loc[token] = tmp
        else:
            data.loc[token] += tmp

    except KeyError:
        pass
    # print(tmp)


def records_to_table(df_records, p_dictionary=None, c_dictionary=None, corr=None):
    '''
    p_dictionary: dictionary for push action
    c_dictionary: dictionary for compare action
    '''
    df_ret = pd.DataFrame(columns=p_dictionary.index)
    re_cmps = re.compile('5[a-z0-9]{20,30}')

    for i, line in df_records.iterrows():
        act = line['actionid']
        tok = line['token']
        product = line['data'][15:-2]
        raw_compare = line['data'][18:-3]
        if (not pd.isna(tok)) and (tok != '0'):
            if act[0:4] == 'PUSH':
                add_record(df_ret, tok, [product], p_dictionary=p_dictionary, c_dictionary=c_dictionary, corr=corr)
            else:
                cmps = re.findall(re_cmps, raw_compare)
                add_record(df_ret, tok, cmps, push=False, p_dictionary=p_dictionary, c_dictionary=c_dictionary,
                           corr=corr)

    ### Rectify scores
    # df_ret = df_ret.applymap(rectifier)
    return df_ret


def expand_table(df, corr):
    for i, row in df.iterrows():
        vals = row.loc[row > 0.0]
        # print(vals)
        idx = row.index[row > 0.0]
        # print(idx)
        candidates = corr.loc[idx]
        # print(candidates.values.shape, vals.values.shape)
        result = np.max(candidates.values.T * vals.values, axis=1)
        # print(row.shape, result.shape, result)
        row.iloc[:] = result
        # print(row)
    return df.applymap(rectifier)


########################### Data Preprocess ####################################

collector_meets_condition = pd.read_csv("csv/collector_2020_03_25.csv", encoding='ISO-8859-1')
#collector = collector.replace(np.nan, '0').astype(str)
#collector_meets_condition_pageid = collector.loc[
                                   #(collector["pageid"].str.contains("VIEW_PRODUCT_DETAIL|VIEW_COMPARE", regex=True)),
                                   #:]
#collector_meets_condition = collector_meets_condition_pageid.loc[
                            #(collector["actionid"].str.contains("PUSH|CLICK_COMPARE", regex=True)), :]

p_dictionary = pd.read_csv("dictionary.csv").set_index('ID')
c_dictionary = pd.read_csv("c_dictionary.csv").set_index('ID')
# print('c_dictionary', c_dictionary)

print('Num of records that can be used: ', len(collector_meets_condition))

# file = open("data.csv", 'r', encoding='UTF-8')  # 记得读取文件时加‘r’， encoding='UTF-8'
# file.readline()

if READ_CORR:
    corr = pd.read_csv('corr.csv')
    corr.set_index('ID', inplace=True)
else:
    corr = pd.DataFrame(tfidf_corr(p_dictionary)).rename(index=lambda x: p_dictionary.index[x],
                                                         columns=lambda x: p_dictionary.index[x])
    # corr = pd.DataFrame(tfidf_corr(p_dictionary)).rename(index = lambda x: p_dictionary.iloc[x].values[0],
    # columns = lambda x: p_dictionary.iloc[x].values[0])
    # corr = calculate_product_correlation(p_dictionary, similarity)
    corr.to_csv('corr.csv', encoding='utf_8_sig')

print(corr)

### Calculate corr distribution
#corr_flat = corr.values.flatten() * 100.0
#plt.hist(corr_flat, bins=np.arange(0, 101, 1))
#plt.show()

### Corr in Chinese & English names
# corr_out = corr.copy()
# corr_out = corr_out.rename(index = lambda x: p_dictionary.loc[x].values[0],
#     columns = lambda x: p_dictionary.loc[x].values[0])
# corr_out.to_csv('corr_out.csv', encoding='utf_8_sig')

# print(corr)
corr_eye = calculate_product_correlation(p_dictionary, similarity, extend=False)

# df1 = records_to_table(collector_meets_condition)
df1 = records_to_table(collector_meets_condition, p_dictionary, c_dictionary, corr_eye)
df1 = expand_table(df1, corr)  # 这一步视是否有dictionary而定

df1 = df1.fillna(0)
df1.to_csv('data_decomposition1_SUM.csv', encoding='UTF-8', )
print('\ndf1: ')
print(df1)

df_train = records_to_table(collector_meets_condition.iloc[:5000, :], p_dictionary, c_dictionary, corr)
df_train = df_train.fillna(0)

df_train_eye = records_to_table(collector_meets_condition.iloc[:5000, :], p_dictionary, c_dictionary, corr_eye)
df_train_eye = df_train_eye.fillna(0)

df_test = records_to_table(collector_meets_condition.iloc[5000:, :], p_dictionary, c_dictionary, corr)
df_test = df_test.fillna(0)

df_test_eye = records_to_table(collector_meets_condition.iloc[5000:, :], p_dictionary, c_dictionary, corr_eye)
df_test_eye = df_test_eye.fillna(0)

print(len(df_train), len(df_test))

"""
计算任何两位用户之间的相似度
"""


########################### recommendation funcs ###############################

# 计算某两个用户的相似度 
def Euclidean(user1, user2, df=df1):
    # 取出两位用户评论过的产品和评分
    # user1_data = data[user1]
    # user2_data = data[user2]
    # distance = 0
    # # 找到两位用户都买过的产品，并计算欧式距离
    # for key in user1_data.keys():
    #     if key in user2_data.keys():
    #         # 注意，distance越小表示两者越相似
    #         distance += pow(float(user1_data[key]) - float(user2_data[key]), 2)
    user1_data = df.loc[user1, :]
    user2_data = df.loc[user2, :]
    distance = np.sum((user1_data - user2_data) ** 2)
    return 1 / (1 + sqrt(distance))  # 这里返回值越大，相似度越大


def Pearson(user1, user2, df=df1):
    user1_data = df.loc[user1, :]
    user2_data = df.loc[user2, :]
    # print(user1, user2, np.corrcoef(user1_data, user2_data))
    # input('stall')
    return np.corrcoef(user1_data, user2_data)[0, 1]


def Cosinual(user1, user2, df=df1):
    user1_data = df.loc[user1, :]
    user2_data = df.loc[user2, :]
    return np.inner(user1_data, user2_data) / (np.linalg.norm(user1_data) * np.linalg.norm(user2_data))


def Jaccard(user1, user2, df=df1):
    user1_data = df.loc[user1, :]
    user2_data = df.loc[user2, :]
    cnt = 0
    cnt_1 = 0
    cnt_2 = 0
    for i in range(0, len(user1_data)):
        if user1_data[i] != 0:
            cnt_1 += 1
        if user2_data[i] != 0:
            cnt_2 += 1
        if user1_data[i] != 0 and user2_data[i] != 0:
            cnt += 1
    return cnt / (cnt_1 + cnt_2)


def SVD_decomposition(userid, df=df1, k=100):
    if not user in df.index:
        return []
    u, s, v = np.linalg.svd(df)
    # print(u,s,v)
    u_tilder = u[:, :k]
    s_tilder = s[:k]
    v_tilder = v[:k, :]
    # print(u_tilder.shape,s_tilder.shape,v_tilder.shape)
    A_tilder = u_tilder * s_tilder @ v_tilder
    # print(A_tilder)

    RMSE_test(A_tilder, df)

    user_data = df.loc[userid, :]
    predict_user_data = A_tilder[df.index.get_loc(userid), :]
    # print(user_data)
    for i, pref in enumerate(predict_user_data):
        if user_data[i] != 0:
            predict_user_data[i] = 0
    idx = np.argsort(predict_user_data)[::-1]
    # print(idx[:5], predict_user_data[idx[:5]], df.columns[idx[:5]])
    return list(df.columns[idx[:5]])


def RMSE_test(predicted, ref=df1):
    '''
    Test RMSE error of A_tilder as a result of SVD decomp, against df1.
    '''
    error = 0
    for usr in predicted.index:
        tmp_error = 0
        tmp_cnt = 0
        for item in predicted.columns:
            if ref.loc[usr, item] != 0:
                tmp_error += (ref.loc[usr, item] - predicted.loc[usr, item]) ** 2
                # print(tmp_error)
                tmp_cnt += 1
        error += tmp_error / tmp_cnt
    error = error / predicted.shape[0]

    return error


def LFM_train(df=df1, k=100):
    lfm = LatentFactorModel(20, k, 0.005, 1)
    lfm.run_new_model()


# 计算对于某个用户而言，与之最相似的m个用户，m可自行定义
def top_simliar(userID, df, m):
    res = []
    # for userid in data.keys():
    for userid in df.index:
        # 排除与自己计算相似度
        if not userid == userID:
            simliar = Rev_distance(userID, userid, df)
            res.append((userid, simliar))

    res.sort(key=lambda val: val[1], reverse=True)
    # res = res[::-1]
    # print('res[:m]', res[:m])
    return res[:m]


Rev_distance = Euclidean


# RES = top_simliar('9659fdb49a5cab698e8c058d472e9c8aaf69e22c20a7fc2995964eccd8fbbdfc', 4)
# print(RES)


########################################################################
# 根据用户推荐产品给其他人

def weighted_recommend(user, df, df_ref=None):
    if not user in df.index:
        return []
    # 相似度最高的用户
    top_sim_user = top_simliar(user, df, 4)[0:4]
    # print('top_sim_user', top_sim_user)
    ws = np.sum(np.exp([x[1] for x in top_sim_user]))
    weights = np.exp([x[1] for x in top_sim_user]) / ws
    # print(weights)
    # weighted_preference = weights.T @ df.loc[[x[0] for x in top_sim_user], :]
    weighted_preference = pd.DataFrame(columns=['preference'])
    weighted_preference['preference'] = np.sum(df.loc[[x[0] for x in top_sim_user], :].T * weights, axis=1)
    # print(df.loc[[x[0] for x in top_sim_user], :].T)
    # print(weighted_preference)

    # if df_ref is not None:
    #     weighted_preference.loc[df_ref.loc[user] > 0.99] = 0

    return weighted_preference

    # weighted_preference = weighted_preference.sort_values(by='preference', ascending = False)
    # return weighted_preference.index[:5].values


########################### tests ##############################################

# def test_

########################### main ###############################################

if __name__ == '__main__':

    if TEST_MODE:

        # Recommendations = weighted_recommend('9659fdb49a5cab698e8c058d472e9c8aaf69e22c20a7fc2995964eccd8fbbdfc', df_train)
        # print('recommendation:')
        # print(Recommendations)

        # now test
        four_FP = 0
        four_TP = 0
        first_FP = 0
        first_TP = 0
        first_precision = 0
        test_cnt = 0

        predicted = pd.DataFrame(columns=p_dictionary.index)

        # tmp_rmse = 0
        if HAS_TQDM:
            progress = tqdm(iterable=df_test.index, total=len(df_test.index), desc='Test Reco', ncols=100)
        else:
            progress = df_test.index
        for user in progress:
            if not HAS_TQDM: print('#', end='')

            weighted_preference = weighted_recommend(user, df_train, df_train_eye)
            if len(weighted_preference) == 0:
                continue
            # print('weighted_preference', weighted_preference)
            # weighted_preference = weighted_preference.sort_values(by='preference', ascending = False)
            recos = weighted_preference.sort_values(by='preference', ascending=False)['preference'].index[0:5]
            weighted_preference.columns = [user]
            predicted = predicted.append(weighted_preference.T)

            # recos = recos.index[:5].values

            # print('recos', recos)
            # recos = SVD_decomposition(user, df_train)
            if len(recos) == 0:
                continue
            test_cnt += np.sum(df_test.loc[user, :])
            for rec in recos:
                if rec not in df_test.columns:
                    four_FP += 1
                    continue
                if df_test.loc[user, rec] != 0:
                    four_TP += 1
                else:
                    four_FP += 1
            rec_1 = recos[0]
            if rec_1 not in df_test.columns:
                first_FP += 1
                continue
            if df_test.loc[user, rec_1] != 0:
                first_TP += 1
            else:
                first_FP += 1
        if HAS_TQDM: progress.close()
        print("\nPrecision evaluated by all four recommendations = ", four_TP / (four_TP + four_FP))
        print("Precision evaluated by first recommendations = ", first_TP / (first_TP + first_FP))
        print("Hit Ratio = ", four_TP / test_cnt)

        # print(predicted)
        # print('----------------------------------------')
        print('RMSE: ', RMSE_test(predicted, df_test))
        # SVD_decomposition('9659fdb49a5cab698e8c058d472e9c8aaf69e22c20a7fc2995964eccd8fbbdfc')

        # flat = df1.values.flatten()
        # flat_filtered = flat[flat > 0]
        # plt.hist(flat_filtered, bins = np.arange(0,21,1))
        # plt.show()
        # # print('flat', list(flat))
        # values = sorted(list(flat_filtered))
        # print(values)
        # values_stat = Counter(values)
        # print(values_stat)

        # similarities = []
        # for u1 in df1.index:
        #     print(f'Processing user {u1}') 
        #     for u2 in df1.index:
        #         if not u1 == u2:
        #             sim = Rev_distance(u1,u2)
        #             similarities.append(sim)
        # plt.hist(similarities)
        # plt.show()

        # print(Counter(similarities))

        # print(top_ten_idx.shape)
        # for top_idx in top_ten_idx:
        #     print(df1.loc[top_idx])
        # top_ten_value = df1.loc[top_ten_idx]
        # print('top_ten_value:', top_ten_value)


    else:

        Recommendations = weighted_recommend('9659fdb49a5cab698e8c058d472e9c8aaf69e22c20a7fc2995964eccd8fbbdfc')
        print('recommendation:')
        print(Recommendations)
