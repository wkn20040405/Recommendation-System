from LatentFactorModel import LatentFactorModel
from Dataprocessing import DataPreprocessing


if __name__ == '__main__':
    print("Welcome to the User_Product Recommender System.")

    #***************Data Preprocessing***************
    print("If this is your first time running the program? You'll need to create the necessary matrices "
          "if it is.")
    initialization = input("Create matrices if they don't already exist? (yes or no) ")
    while initialization != 'yes' and initialization != 'no':
        initialization = input("Create matrices if they don't already exist? (yes or no) ")
    if initialization == 'yes':
        preprocess = DataPreprocessing()
        preprocess.run_random_split()
        preprocess.run_arbitrary_split()
        #上面这一步默认选no，因为已经初始化好

    # ***************Latent Factor***************
    run = input("Do you want to run the latent factor model recommendation? (yes or no) ")
    while run != 'yes' and run != 'no':
        run = input("Do you want to run the latent factor model recommendation? (yes or no) ")
    if run == 'yes':
        model_type = input("Do you want to start a new model or load an old model? (old or new)\nWarning: "
                               "training a new model is very slow, it's recommended that you use the default model provided. ")
        while model_type != 'new' and model_type != 'old':
            model_type = input("Do you want to start a new model or load an old model? (old or new) ")
            #上面这一步默认选old，即用已生成的模型，模型存储在optimization文件夹里面
        if model_type == 'new':
            parameters = input("Do you want to use the default parameters? (yes or no) ")
            while parameters != 'yes' and parameters != 'no':
                parameters = input("Do you want to use the default parameters? (yes or no) ")
            if parameters == 'yes':
                print("Initializing the latent factor model.")
                latent_model = LatentFactorModel(epochs=15, k=20, learning_rate=0.006, lambda_reg=0.06)
                print("Beginning the long training process...")
                latent_model.run_new_model()
            elif parameters == 'no':
                epochs = input("Enter the number of epochs to train for: ")
                k = input("Enter the number of latent factors: ")
                learning_rate = input("Enter the learning rate: ")
                lambda_reg = input("Enter the lambda regularization value: ")
                print("Initializing the latent factor model.")
                latent_model = LatentFactorModel(epochs=epochs, k=k, learning_rate=learning_rate, lambda_reg=lambda_reg)
                print("Beginning the long training process...")
                latent_model.run_new_model()
        elif model_type == 'old':
            default_model = input("Do you want to load the default model? (yes or no)\nNote: it's recommended "
                                          "to load the default model. ")
            while default_model != 'yes' and default_model != 'no':
                default_model = input("Do you want to load the default model? (yes or no)\nNote: it's recommended "
                                          "to load the default model. ")
            #最好用defaultmodel
            if default_model == 'yes':
                print("Initializing the latent factor model.")
                latent_model = LatentFactorModel(epochs=15, k=20, learning_rate=0.006, lambda_reg=0.06)
                latent_model.load_model(model_directory='optimization/2020-03-13_17-29-44/')
                print("Calculating the random split test RMSE.")
                test_rmse = latent_model.calculate_test_rmse()
                print("Random split test RMSE is: " + str(test_rmse))
            elif default_model == 'no':
                directory = input("Enter the model directory (e.g. 'optimization/2020-03-13_17-29-44/'): ")
                print("Initializing the latent factor model.")
                latent_model = LatentFactorModel(epochs=15, k=20, learning_rate=0.006, lambda_reg=0.06)
                latent_model.load_model(model_directory=directory)
                print("Calculating the random split test RMSE.")
                test_rmse = latent_model.calculate_test_rmse()
                print("Random split test RMSE is: " + str(test_rmse))

#下面即可从最后获得的模型（这里是epoch_14）当中，通过Q和P矩阵相乘并加上latent_model.user_average即为最后的结果
#我将过程放到了Get_Result 当中
#npp = np.load("optimization/2020-03-13_17-29-44/epoch_14/P.npy")
#npq = np.load("optimization/2020-03-13_17-29-44/epoch_14/Q.npy")
#final = npq.dot(npp) + latent_model.user_average.item()
#for user in range(final.shape[0]):
#    final[user, :] = final[user, :] + latent_model.user_average[user]
#注意：默认模型中这一段跑不出来，因为我忘加user_average了，需要重新跑一遍模型才行，但是Q*P可以算出来
#造成最后结果不良的影响：1、k取得太小或者太大。2、数据量不足只有4000个。3、SVD中选取k个奇异值的策略。4、在SVD分解之前是否需要进行减去user_averafge的操作