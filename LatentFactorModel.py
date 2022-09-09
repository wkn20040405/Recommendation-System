import itertools
import numpy as np
import math
from collections import defaultdict
import time
import os
import datetime
import pandas as pd
from scipy.sparse.linalg import svds

class LatentFactorModel:
    def __init__(self, epochs, k, learning_rate, lambda_reg):
        # Load the sparse matrix from a file
        self.training_filepath = 'matrices/{}_training'.format('random')
        self.testing_filepath = 'matrices/{}_test'.format('random')
        #Here I can change 'random' or 'arbitary'
        self.training = self.load_csv_matrix(self.training_filepath)
        self.test = self.load_csv_matrix(self.testing_filepath)
        # get sparse matrix
        self.P = None
        self.Q = None
        self.final = pd.DataFrame()
        self.epochs = epochs
        self.current_epoch = 0
        self.number_of_training_ratings = 0
        self.k = k
        self.learning_rate = learning_rate
        self.lambda_reg = lambda_reg
        self.user_average = {}
        self.global_mean = 0.0
        self.model_directory = None
        self.model_loaded = False
        self.training_id = np.array(self.training != 0)
        self.test_id = np.array(self.test != 0)

        self.calculate_mean_user_rating()
        self.training = self.center_matrix_user(self.training, self.user_average)

    def get_useful_id(self, matrix):
        temp = []
        for i in range(0, matrix.shape[0]):
            for j in range(0, matrix.shape[1]):
                if matrix.iloc[i][j] != 0:
                    temp.append((i,j))
        return temp

    def center_matrix_user(self, matrix, user_average):
        #centered_matrix = self.basic_matrix.copy(deep=True)
        num_users = matrix.shape[0]
        # Create vectors of centered ratings
        for user in range(num_users):
            matrix.iloc[user, list(matrix.iloc[user,:] != 0)] = matrix.iloc[user, list(matrix.iloc[user,:] != 0)] - user_average[user]
        return matrix

    def load_csv_matrix(self, file_name):
        return pd.read_csv(file_name, index_col=0)

    def calculate_global_baseline_rating(self):
        summed_movie_rating = 0
        number_of_ratings = 0
        for i in range(0, self.training.shape[0]):
            for j in range(0, self.training.shape[1]):
                if self.training.iloc[i][j] != 0:
                    number_of_ratings += 1
                    summed_movie_rating = summed_movie_rating + self.training.iloc[i][j]
        self.number_of_training_ratings = number_of_ratings
        self.global_mean = float(summed_movie_rating) / number_of_ratings

    def calculate_mean_user_rating(self):
        user_sums = self.training.sum(axis=1)
        user_rating_counts = {}
        # Calculate the number of ratings for each user
        for i in range(self.training.shape[0]):
            user_rating_counts[i] = (self.training.iloc[i, :] != 0).sum()
        # Loop through each user
        for index in range(0, self.training.shape[0]):
            # Check to see if the user has not rated
            if user_rating_counts[index] == 0:
                user_average = 0
            else:
                user_average = float(user_sums[index]) / user_rating_counts[index]
            self.user_average[index] = user_average

    def run_svd(self):
        u, s, v = svds(self.training, k=self.k, which='LM')
        #注意此处由Dataframe变成了np.array
        #u_tilder = u[:, :self.k]
        #s_tilder = s[:self.k]
        #v_tilder = v[:self.k, :]
        #这样挑选奇异值对吗？可能需要random
        #\讨论结论：取MAX的奇异值
        self.Q = u
        diag_matrix = np.diag(s)
        self.P = diag_matrix.dot(v)

    def predicted_value(self, user_new_id, product_new_id):
        col = self.P[:, product_new_id]
        row = self.Q[user_new_id, :]
        return row.dot(col)

    def error(self, user_new_id, product_new_id):
        actual_value = self.training.iloc[user_new_id, product_new_id]
        predicted_value = self.predicted_value(user_new_id, product_new_id)
        return actual_value - predicted_value

    def square_error_train(self, user_new_id, product_new_id):
        actual_value = self.training.iloc[user_new_id, product_new_id]
        predicted_value = self.predicted_value(user_new_id, product_new_id)
        return math.pow(actual_value - predicted_value, 2)

    def square_error_test(self, user_new_id, product_new_id):
        actual_value = self.test.iloc[user_new_id, product_new_id]
        #
        predicted_value = self.predicted_value(user_new_id, product_new_id) + self.user_average[user_new_id]
        return math.pow(actual_value - predicted_value, 2)

    def calculate_test_rmse(self):
        summed_error = 0
        test_dataset_size = (self.test != 0).sum().sum()
        # Loop through each entry in the test dataset
        for user_new_id in range(self.test.shape[0]):
            for product_new_id in range(self.test.shape[1]):
                if self.test.iloc[user_new_id, product_new_id] != 0:
                    summed_error += self.square_error_test(user_new_id, product_new_id)
        # Calculate the number of entries in the test set
        rmse = math.sqrt(float(summed_error) / test_dataset_size)
        return rmse

    def calculate_training_rmse(self):
        summed_error = 0
        training_dataset_size = (self.training != 0).sum().sum()
        # Loop through each entry in the test dataset
        for user_new_id in range(self.training.shape[0]):
            for product_new_id in range(self.training.shape[1]):
                if self.training.iloc[user_new_id, product_new_id] != 0:
                    summed_error += self.square_error_train(user_new_id, product_new_id)
        # Calculate the number of entries in the test set
        rmse = math.sqrt(float(summed_error) / training_dataset_size)
        return rmse

    def save_hyperparameters(self):
        hyper_param_file = '{}hyperparams.txt'.format(self.model_directory)
        params = 'Learning rate: {} \nRegularization rate: {} \nNumber of factors (k): {} \n# of epochs: {}'.format(self.learning_rate, self.lambda_reg, self.k, self.epochs)
        f = open(hyper_param_file, "w+")
        f.write(params)
        f.close()

    def save_rmse_file(self, directory, rmse_training, rmse_test):
        rmse_file = directory + "RMSE.txt"
        rmse_info = 'RMSE Training: {} \nRMSE Test: {}'.format(rmse_training, rmse_test)
        f = open(rmse_file, "w+")
        f.write(rmse_info)
        f.close()

    def save_user_average(self, directory):
        matrix = "{}{}".format(directory, "user_average.npy")
        np.save(arr=self.user_average, file=matrix)
        #dictionary 也可以储存成为npy文件，读取时最后加上.item（）

    def save_matrices(self, directory):
        p_matrix = "{}{}".format(directory, "P.npy")
        q_matrix = "{}{}".format(directory, "Q.npy")
        np.save(arr=self.P, file=p_matrix)
        np.save(arr=self.Q, file=q_matrix)

    def save_model(self, epoch, rmse_test, rmse_training):
        # Only create the hyperparameter file once
        if epoch == 0:
            self.model_directory = 'optimization/{}/'.format(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
            directory = '{}epoch_{}/'.format(self.model_directory, epoch)
            os.makedirs(self.model_directory)
            self.save_hyperparameters()
            self.save_user_average(self.model_directory)
        else:
            directory = '{}epoch_{}/'.format(self.model_directory, epoch)
        if not os.path.exists(directory):
            os.makedirs(directory)
            self.save_matrices(directory=directory)
            self.save_rmse_file(directory=directory, rmse_training=rmse_training, rmse_test=rmse_test)
        else:
            print("Error: directory already exists")

    def find_current_epoch(self, model_directory):
        highest_epoch = -1
        for directory in os.listdir(model_directory):
            # Check that it's actually a directory
            if os.path.isdir(model_directory + directory):
                temp, current_epoch = directory.split('_')
                current_epoch = int(current_epoch)
                if current_epoch > highest_epoch:
                    highest_epoch = current_epoch
        return highest_epoch

    def load_hyperparameters(self, path_to_hyperparam_file):
        with open(path_to_hyperparam_file) as f:
            lines = f.readlines()
            temp, learning_rate = lines[0].split(':')
            temp, reg_rate = lines[1].split(':')
            temp, num_factors = lines[2].split(':')
            temp, epochs = lines[3].split(':')
            self.learning_rate = float(learning_rate.strip())
            self.lambda_reg = float(reg_rate.strip())
            self.k = int(num_factors.strip())
            self.epochs = int(epochs.strip())

    def load_model(self, model_directory):
        self.model_loaded = True
        # Find the last epoch that was saved
        epoch = self.find_current_epoch(model_directory=model_directory)
        if epoch >= 0:
            self.current_epoch = epoch
        else:
            print("Failed to find epoch folder")
            exit(1)
        path_to_model = model_directory + 'epoch_{}/'.format(epoch)
        path_to_user_average = model_directory + 'user_average.npy'
        path_to_hyperparam = model_directory + 'hyperparams.txt'
        self.model_directory = model_directory
        # Check that hyperparamter file exists
        if os.path.exists(path_to_hyperparam):
            self.load_hyperparameters(path_to_hyperparam)
            self.user_average = np.load(path_to_user_average, allow_pickle=True).item()
        else:
            print("Failed to find the hyperparameter file.")
            exit(1)
        path_to_matrix_P = path_to_model + 'P.npy'
        path_to_matrix_Q = path_to_model + 'Q.npy'
        # Check that the models exist
        if os.path.exists(path_to_model):
            self.P = np.load(path_to_matrix_P)
            self.Q = np.load(path_to_matrix_Q)
        else:
            print("Failed to load model.")
            exit(1)

    def calculate_epoch_error(self, epoch):
        #print "Movie 4830, user 47914, true rating: 6. Predicted rating: " + str( self.predicted_value(4830, 47914) + self.user_average[47914])
        start = time.time()
        rmse_test = self.calculate_test_rmse()
        end = time.time()
        print("Time to calculate RMSE test: {}".format(end - start))
        start = time.time()
        rmse_training = self.calculate_training_rmse()
        end = time.time()
        print("Time to calculate RMSE training: {}".format(end - start))
        print("Training RMSE for epoch {}: {}".format(epoch, rmse_training))
        print("Test RMSE for epoch {}: {}".format(epoch, rmse_test))
        return rmse_test, rmse_training

    def optimize_matrices(self):
        model_already_tested_and_saved = self.model_loaded
        epoch_count = 1
        for epoch in range(self.current_epoch, self.epochs):
            if not model_already_tested_and_saved:
                rmse_test, rmse_training = self.calculate_epoch_error(epoch)
                #training 的error竟然比test的要大
                self.save_model(epoch=epoch, rmse_test=rmse_test, rmse_training=rmse_training)
                print("Epoch {} model saved".format(epoch))
            else:
                model_already_tested_and_saved = False
            #若一开始model_already_tested_and_saved为True， 则直接第一次开始梯度下降，而不是录入模型
            self.learning_rate *= 0.9 #循环次数越多，learning_rate越低
            print("Reduce Learning_rate")
            epoch_count += 1
            count = 0
            start = time.time()
            # Loop through each entry in the training dataset
            for user in range(0,self.training.shape[0]):
                for product in range(0,self.training.shape[1]):
                    if self.training_id[user, product]:
                        if count % 10000 == 0:
                            print("Current count {}".format(count))
                            end = time.time()
                            print("Time taken {}".format(end - start))
                            start = end
                        count = count + 1
                        # Loop through every latent factor
                        for k in range(self.k):
                            gradient_q = self.learning_rate * (2 * self.error(user, product) * self.P[k, product] - 2 *
                                                               self.lambda_reg * self.Q[user, k])
                            self.Q[user, k] = self.Q[user, k] + gradient_q
                            gradient_p = self.learning_rate * (2 * self.error(user, product) * self.Q[user, k] - 2 *
                                                               self.lambda_reg * self.P[k, product])
                            self.P[k, product] = self.P[k, product] + gradient_p

    def run_new_model(self):
        self.run_svd()
        self.optimize_matrices()

    def run_old_model(self, model_directory):
        self.load_model(model_directory=model_directory)
        self.optimize_matrices()

    def Get_Result(self):
        model_directory = self.model_directory
        self.load_model(model_directory=model_directory)
        original = self.Q.dot(self.P)
        final = pd.read_csv("csv/basic_matrix.csv", index_col=0)
        for user in range(original.shape[0]):
            final.iloc[user, :] = original[user, :] + self.user_average[user]
            print("Now computing...")
        self.final = final
        #print(final)
        final.to_csv(model_directory+"rating_PQ.csv")
        final.to_csv("csv/rating_PQ.csv")

    def Recommendation_one_Result(self, userid, recommend_number, final, corr):
        orignial_data = pd.read_csv("csv/rating_original_matrix.csv", index_col=0)
        temp_id = orignial_data.loc[userid] == 0
        temp_data = final.loc[userid, temp_id]
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

    def Recommendation_total_Result(self, recommend_number, final):
        record_data = pd.DataFrame(index=self.training.index)
        corr = pd.read_csv("csv/corr.csv", index_col=0)
        for i in range(recommend_number):
            name = "Recommend-product_{}".format(i + 1)
            record_data.loc[:, name] = ""
            # record_data[name] = "fuck_now" #这里不能写record_data.loc[:, name] = pd.Series()
        for j in range(record_data.shape[0]):
            temp_dictionary = self.Recommendation_one_Result(record_data.index[j], recommend_number, final, corr)
            print("Great.{}".format(j))
            for k in range(record_data.shape[1]):
                s = temp_dictionary[k][0]
                record_data.iloc[j, k] = s
                # 极力避免使用record_data.iloc[j][k]
                # 或者使用record_data.loc[行名，列名]
        record_data.to_csv("csv/Results.csv", encoding='utf_8_sig')
        #print(record_data)
        return record_data


if __name__ == '__main__':
    start_whole = time.time()
    print("Initializing the latent factor model.")
    latent_model = LatentFactorModel(epochs=10, k=20, learning_rate=0.001, lambda_reg=0.01)
    print("Beginning the long training process...")
    latent_model.run_new_model()
    print("Now preparing to get final result for this model")
    latent_model.Get_Result()
    print("Get Results")
    final = pd.read_csv("csv/rating_PQ.csv", index_col=0)
    latent_model.Recommendation_total_Result(recommend_number=3, final=final)
    print("Get Result Matrix!")
    end = time.time()
    print("Time to run program " + str((end - start_whole)))