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
from pprint import pprint  # pprint 数据结构更加完整


class DataPreprocessing:
    def __init__(self):
        start_time = time.time()
        print("Initializing Data Preprocessing")
        self.arbitrary_training_filepath = 'csv/arbitrary_training.csv'
        self.arbitrary_testing_filepath = 'csv/arbitrary_test.csv'
        self.random_training_filepath = 'csv/random_training.csv'
        self.random_testing_filepath = 'csv/random_test.csv'
        self.original_data_filepath = 'csv/collector.csv'
        self.row_data_filepath = 'csv/data_decomposition.csv'
        self.basic_matrix_filepath = 'csv/basic_matrix.csv'
        self.npz_path = 'matrices/'
        self.collector = pd.DataFrame()
        self.basic_matrix = None
        self.READ_CORR = False
        self.have_dictionary = True
        self.row_data = pd.DataFrame()
        #self.updata_collector = False  # 此选项可更改，表示是否更新数据并下载到本地
        self.updata_collector = True

        if not os.path.isdir('csv/'):
            print("Ensure that there exists the csv folder in current working directory")
            exit(1)
        if not os.path.isdir(self.npz_path):
            os.mkdir(self.npz_path)
        if not self.updata_collector:
            self.collector = pd.read_csv("csv/collector.csv", index_col=0)
        else:
            self.get_data()
            print("Finish downloading Collector and Dictionary Data from Mongo Database")

        print("Now trying to get data_decomposition and row data matrix!")
        self.after_process()
        print("Now trying to get basic matrix!")
        self.get_basic_matrix()
        self.rating_data = self.load_csv_as_pd_Dataframe(self.row_data_filepath)
        print("Data Preprocessing initialization done in {} seconds".format(time.time() - start_time))

    def get_data(self):
        MONGODB_HOST = '117.50.37.209'
        MONGODB_PORT = 27027
        conn = pymongo.MongoClient('{}:{}'.format(MONGODB_HOST, MONGODB_PORT))
        DATEBASE = 'lxj_app'
        TABLENAME = 'collector'
        TABLENAME_dictionary = 'material'
        table = conn[DATEBASE][TABLENAME]
        result = table.find({"actionid": {"$in": ["CLICK_COMPARE", "PUSH_PRODUCT_DETAIL_FROM_HOT",
                                                  "PUSH_PRODUCT_DETAIL_FROM_PRODUCTS",
                                                  "PUSH_PRODUCT_DETAIL_FROM_RECOMMEND",
                                                  "PUSH_PRODUCT_DETAIL_FROM_SEARCH"]}}, no_cursor_timeout=True)
        collector = pd.DataFrame(columns=["type", "stay_time", "pageid", "actionid", "udid", "platform", "token",
                                          "sid", "lastid", "data", "start_time", "end_time"])
        count = 0
        for item in result:
            collector = collector.append(pd.DataFrame([item], index=[count]), sort=True)
            count += 1
            if count % 10000 == 0:
                print("Current count :{}".format(count))
        collector.fillna(0, inplace=True)
        collector.sort_values(axis=0, by="_id").reset_index(drop=True)
        collector.to_csv(self.original_data_filepath)
        self.collector = pd.read_csv("csv/collector.csv", index_col=0)
        table_dictionary = conn[DATEBASE][TABLENAME_dictionary]
        result_dictionary = table_dictionary.find({'product_id': {'$exists': 1}}, {'product_id': 1, 'project_name': 1})
        p_dictionary = pd.DataFrame()
        c_dictionary = pd.DataFrame()
        for item in result_dictionary:
            pid = str(item['_id'])
            cid = str(item['product_id'])
            name = item['project_name']
            if "测试" not in name:
                p_dictionary = p_dictionary.append(pd.DataFrame({"名字": [name]}, index=[pid]), sort=True).sort_index()
                c_dictionary = c_dictionary.append(pd.DataFrame({"名字": [name]}, index=[cid]), sort=True).sort_index()
        p_dictionary.to_csv("csv/dictionary.csv", encoding="utf_8_sig")
        c_dictionary.to_csv("csv/c_dictionary.csv", encoding="utf_8_sig")

    def seg_sentence(self, sentence, stop_words):
        stop_flag = ['x', 'c', 'u', 'd', 'p', 't', 'uj', 'f', 'r']
        sentence_seged = pseg.cut(sentence)
        outstr = []
        for word, flag in sentence_seged:
            if word not in stop_words and flag not in stop_flag:
                outstr.append(word)
        return outstr

    def tfidf_corr(self, dictionary):
        stop_words = ['「', '」', '（', '）', '(', ')', '-', '•']
        texts = [self.seg_sentence(seg, stop_words) for seg in dictionary['名字']]
        dictionary = corpora.Dictionary(texts)
        feature_cnt = len(dictionary.token2id.keys())
        bow_corpus = [dictionary.doc2bow(text) for text in texts]
        tfidf = models.TfidfModel(bow_corpus)
        tf_corpus = tfidf[bow_corpus]
        index = similarities.SparseMatrixSimilarity(tf_corpus, num_features=feature_cnt)
        return index

    def rectifier(self, x):
        # return exp(x) / (1.0 + exp(x))
        return log(x + 1)

    def calculate_product_correlation(self, dictionary, sim, extend=True, threshold=0, amplifier=1.0):
        product_correlation = pd.DataFrame()
        if dictionary is not None:
            product_correlation = pd.DataFrame(data=np.eye(len(dictionary)), index=dictionary.index,
                                               columns=dictionary.index)
            if extend:
                progress = dictionary.index
                for Id in progress:
                    print('#', end='', flush=True)  # flush 清空缓存，立刻显示 end 结尾不换行
                    for col in dictionary.index:
                        match = sim(Id, col)
                        if match > 0.99:
                            product_correlation.loc[Id, col] = 1.0
                        else:
                            product_correlation.loc[Id, col] = max((match - threshold) * amplifier, 0)
        return product_correlation

    def similarity(self, str1, str2):
        return difflib.SequenceMatcher(lambda x: x in ' 「」（）()-•', str1, str2).quick_ratio()

    def add_record(self, data, token, pids, push=True, p_dictionary=None, c_dictionary=None, corr=None):
        '''
        Add record of token:pid to data of type dict
        '''
        # print('#', end='', flush =True)
        ### translate compare id to push id
        if not push and (p_dictionary is not None) and (c_dictionary is not None):
            names = c_dictionary.reindex(pids)
            real_pids = p_dictionary.index[p_dictionary['名字'].isin(names['名字'])]
        else:
            real_pids = pids
        try:
            tmp = np.max(corr.loc[real_pids, :], axis=0)
            if token not in data.index:
                data.loc[token] = tmp
            else:
                data.loc[token] += tmp

        except KeyError:
            print("Error happens: Cannot find compare_pids in c_dictionary")
            pass

    def records_to_table(self, df_records, p_dictionary=None, c_dictionary=None, corr=None):
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
                    self.add_record(df_ret, tok, [product], p_dictionary=p_dictionary, c_dictionary=c_dictionary,
                                    corr=corr)
                else:
                    cmps = re.findall(re_cmps, raw_compare)
                    self.add_record(df_ret, tok, cmps, push=False, p_dictionary=p_dictionary, c_dictionary=c_dictionary,
                                    corr=corr)
        return df_ret

    def expand_table(self, df, corr):
        df1 = pd.DataFrame(columns=["user_id", "product_id", "rating"])
        count = 0
        for i, row in df.iterrows():
            # row是一个list
            vals = row.loc[row > 0.0]
            idx = row.index[row > 0.0]
            candidates = corr.loc[idx]
            result = np.max(candidates.values.T * vals.values, axis=1)  # 每行均乘以一个vals的list（元素对应相乘），然后取每行的最大值
            row.iloc[:] = result
            idx = row.index[row > 0.0]
            for j in idx:
                temp = pd.DataFrame({"user_id": [i],
                                     "product_id": [j],
                                     "rating": [self.rectifier(row.loc[j])]}, index=[count])
                count += 1
                if count % 10000 == 0:
                    print("Current count {}".format(count))
                df1 = df1.append(temp)
        df1.sort_values(axis=0, by="user_id").reset_index(drop=True)
        self.row_data = df1
        df1.to_csv(self.row_data_filepath, encoding='UTF-8')
        print("Get data_decomposition.csv")
        print(df.shape)
        return df.applymap(self.rectifier)

    def load_table(self, temporary):
        df1 = pd.DataFrame(columns=["user_id", "product_id", "rating"])
        count = 0
        for user in temporary.index:
            for product in temporary.columns:
                if temporary.loc[user][product] != 0:
                    count += 1
                    temp = pd.DataFrame({"user_id": [user],
                                         "product_id": [product],
                                         "rating": [temporary.loc[user][product]]}, index=[count])
                    df1 = df1.append(temp)
                    if count % 10000 == 0:
                        print("Current count {}".format(count))
        df1.to_csv(self.row_data_filepath, encoding='UTF-8')
        print("Get data_decomposition.csv")

    def after_process(self):
        p_dictionary = pd.read_csv("csv/dictionary.csv", index_col=0)
        c_dictionary = pd.read_csv("csv/c_dictionary.csv", index_col=0)
        print('Num of records that can be used: ', len(self.collector))
        if self.READ_CORR:
            corr = pd.read_csv('csv/corr.csv', index_col=0)
        else:
            corr = pd.DataFrame(self.tfidf_corr(p_dictionary)).rename(index=lambda x: p_dictionary.index[x],
                                                                      columns=lambda x: p_dictionary.index[
                                                                          x])  # 此处语法不需要写x的范围
            corr.to_csv('csv/corr.csv', encoding='utf_8_sig')
        corr_eye = self.calculate_product_correlation(p_dictionary, self.similarity, extend=False)
        df1 = self.records_to_table(self.collector, p_dictionary, c_dictionary, corr_eye)
        df1 = df1.fillna(0)
        df1.sort_index(inplace=True)
        df1.to_csv('csv/rating_original_matrix.csv', encoding='UTF-8')
        if self.have_dictionary:
            df1 = self.expand_table(df1, corr)  # 这一步视是否有dictionary而定
        else:
            df1.applymap(self.rectifier)
            self.load_table(df1)
        df1.sort_index(inplace=True)
        df1.to_csv('csv/rating_before_decomposition.csv', encoding='UTF-8')
        print("Get user-product matrix!")

    def get_basic_matrix(self):
        temporary = pd.read_csv("csv/rating_before_decomposition.csv", index_col=0)
        df2 = temporary.copy(deep=True)
        df2.iloc[:, :] = 0
        df2 = df2.sort_index()
        self.basic_matrix = df2
        df2.to_csv(self.basic_matrix_filepath, encoding='UTF-8')
        print("Get basic_matrix.csv")

    def save_csv_data(self, file_name, data_csv):
        data_csv.to_csv(file_name, encoding='UTF-8')

    def load_csv_as_pd_Dataframe(self, filepath):
        # 注意这里一定要声明index_col=0，否则index也会被当成一列
        data = pd.read_csv(filepath, delimiter=',', dtype=str, index_col=0, header=0)
        return data

    def random_sample(self, data, seed=42, percent_test=0.2):
        random.seed(seed)
        total_size = data.shape[0]
        test_size = int(total_size * percent_test)
        test_list = random.sample(range(total_size), test_size)
        test_list.sort(key=lambda val: val)
        train_list = list(range(total_size))
        for ix in test_list:
            train_list.remove(ix)
        test_data = data.iloc[test_list, :]
        training_data = data.iloc[train_list, :]
        return training_data, test_data,

    def row_data_to_matrix(self, data):
        matrix = self.basic_matrix.copy(deep=True)
        for index in range(0, data.shape[0]):
            matrix.loc[data.iloc[index, 0]][data.iloc[index, 1]] = data.iloc[index, 2]
        return matrix

    def random_split(self, rating_data):
        training_dst = self.npz_path + 'random_training'
        testing_dst = self.npz_path + 'random_test'
        # Randomly split the data into an 80-20 training/test set respectively
        training_data, test_data = self.random_sample(rating_data)

        training_matrix = self.row_data_to_matrix(training_data)
        test_matrix = self.row_data_to_matrix(test_data)

        print("Saving Data")
        self.save_csv_data(training_dst, training_matrix)
        self.save_csv_data(testing_dst, test_matrix)
        print("Done")

    def arbitrary_split(self, rating_data):
        training_dst = self.npz_path + 'arbitrary_training'
        testing_dst = self.npz_path + 'arbitrary_test'
        # Find the 80-20 split point from the loaded data
        split_delimiter = int(.8 * rating_data.shape[0])
        training_data = rating_data.iloc[0:split_delimiter, :]
        test_data = rating_data.iloc[split_delimiter:, :]
        training_matrix = self.row_data_to_matrix(training_data)
        test_matrix = self.row_data_to_matrix(test_data)
        print("Saving Data")
        self.save_csv_data(training_dst, training_matrix)
        self.save_csv_data(testing_dst, test_matrix)
        print("Done")

    def run_random_split(self):
        start_time = time.time()
        print("Running random split")
        self.random_split(rating_data=self.rating_data)
        print("Random split finished in {} seconds".format(time.time() - start_time))

    def run_arbitrary_split(self):
        start_time = time.time()
        print("Running arbitrary split")
        self.arbitrary_split(rating_data=self.rating_data)
        print("Arbitrary split finished in {} seconds".format(time.time() - start_time))


if __name__ == '__main__':
    start_whole = time.time()
    preprocess = DataPreprocessing()
    preprocess.run_random_split()
    end = time.time()
    print("Time to run program " + str((end - start_whole)))
