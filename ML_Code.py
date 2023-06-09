import pandas as pd
import numpy as np
#Importing the linear regression model, most of this code came from lab 1
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score
#Normalizing the features using MinMaxScaler
#https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MinMaxScaler.html 
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import PolynomialFeatures
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.utils import shuffle
from scipy.stats import pearsonr
import optuna

def get_df():
    df=pd.read_excel("GameDataCleaned.xlsx")    #read file.
    #Checking if the dataframe contains empty cell values.
    df.isnull().values.any() #Cleans out any null values in the dataframe
    return df

def linear_regression(df, num_splits):
    X1=df[["Year_of_Release", "Publisher", "Genre", "Critic_Score", "Critic_Count", "User_Score", "User_Count", "Published_in_NA",
    "Published_in_EU", "Published_in_JP", "Published_in_Other"]] #Sets up the dataframe 
    #Output label
    y=df['Global_Sales']
    #print(X1)

    scaler = MinMaxScaler()
    X = scaler.fit_transform(X1)
    #print(X)

    #sample code
    lin_reg = LinearRegression()
    np.random.seed(40)  # to make this code example reproducible

    kfold = KFold(n_splits=num_splits, shuffle=True, random_state=42) #--->here k=3 with 3-fold cross validation.
    #random_state controls randomness of each fold.
    scores = []
    for i, (train, test) in enumerate(kfold.split(X, y)):#--->splitting into train and test set.
    #Enter your code here. You can refer lab 1 and assignment 1 document for the implementation of
    #fit and score.
        lin_reg.fit(X[train], y[train])
        score = lin_reg.score(X[test], y[test])
        scores.append(score)

    count = 0
    avg = 0
    for i in scores:
        avg += i
        count += 1

    #print(scores)
    #print(count)
    avg = avg/count
    #print(avg)
    return avg

def polynomial_regression(df, degree, num_splits):
    X1=df[["Year_of_Release", "Publisher", "Genre", "Critic_Score", "Critic_Count", "User_Score", "User_Count", "Published_in_NA",
    "Published_in_EU", "Published_in_JP", "Published_in_Other"]] #Sets up the dataframe 
    #Output label
    y=df['Global_Sales']
    #print(X1)

    scaler = MinMaxScaler()
    X = scaler.fit_transform(X1)
    #print(X)

    #sample code
    lin_reg = LinearRegression()
    np.random.seed(40)  # to make this code example reproducible
    scores_poly = []
    #Sample code
    kfold = KFold(n_splits=num_splits, shuffle=True, random_state=42)  #------>In this example, k=3
    for i, (train, test) in enumerate(kfold.split(X, y)):

            poly_model = PolynomialFeatures(degree)  #polynomialFeatures class to transform our training data, adding the higher degree 'd' of each feature in the training set as a new feature
            #Enter you code here-->This is similar to k-fold cross validation for linear regression except that now, we are
            #using polynomial features.
            X_poly = poly_model.fit_transform(X)
            #poly_model.fit_transform(X[train], y[train])
            lin_reg.fit(X_poly[train], y[train])
            score = lin_reg.score(X_poly[test], y[test])
            scores_poly.append(score)

    count = 0
    avg_poly = 0
    for i in scores_poly:
        avg_poly += i
        count += 1

    #print(scores_poly)
    #print(count)
    avg_poly = avg_poly/count
    return avg_poly

def Neural_Network(df):
    X1=df[["Year_of_Release", "Publisher", "Genre", "Critic_Score", "Critic_Count", "User_Score", "User_Count", "Published_in_NA",
    "Published_in_EU", "Published_in_JP", "Published_in_Other"]] #Sets up the dataframe 
    #Output label
    y=df['Global_Sales']
    #print(X1)
    X_train, X_test, y_train, y_test = train_test_split(X1, y, test_size = 0.30)
    
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.fit_transform(X_test)
    model = tf.keras.Sequential([tf.keras.layers.Dense(units=10, activation='relu'), 
                    tf.keras.layers.Dense(units=20, activation='relu'),
                    tf.keras.layers.Dense(units=10, activation='relu'), 
                    tf.keras.layers.Dense(units=1, activation='relu')])

    model.compile(loss='mean_absolute_error', optimizer=tf.keras.optimizers.Adam(0.001))
    model.fit(X_train_scaled,y_train,epochs=150)
    preds = model.predict(X_test_scaled) #use model to predict x test values
    print('R score is :', r2_score(y_test, preds))

# #Objective function for Optuna from https://medium.com/optuna/using-optuna-to-optimize-tensorflow-hyperparameters-57b6d4d316a2
# def objective(trial):
#     classifier = create_classifier(trial)
#     df = get_df()

#     optuna_pruning_hook = optuna.integration.TensorFlowPruningHook(
#         trial=trial,
#         estimator=classifier,
#         metric="accuracy",
#         run_every_steps=50
#     )

#     train_spec = tf.estimator.TrainSpec(
#         input_fn=, max_steps=1000, hooks=[optuna_pruning_hook]
#     )

#     eval_spec = tf.estimator.EvalSpec(input_fn=, start_delay_secs=0, throttle_secs=0)

#     eval_results, _ = tf.estimator.train_and_evaluate(classifier, train_spec, eval_spec)

#     return float(eval_results["accuracy"])

# #Classifier function from Optuna from https://medium.com/optuna/using-optuna-to-optimize-tensorflow-hyperparameters-57b6d4d316a2
# def create_classifier(trial):
#     # We optimize the numbers of layers and their units.

#     n_layers = trial.suggest_int("n_layers", 1, 3)
#     hidden_units = []
#     for i in range(n_layers):
#         n_units = trial.suggest_int("n_units_l{}".format(i), 1, 128)
#         hidden_units.append(n_units)


def KMeans_Clustering(df):
    X=df[["Year_of_Release", "Publisher", "Genre", "Critic_Score", "Critic_Count", "User_Score", "User_Count", "Published_in_NA",
    "Published_in_EU", "Published_in_JP", "Published_in_Other"]] #Sets up the dataframe 
    #Output label
    y=df['Global_Sales']
    
    X, y = shuffle(X, y, random_state=42)
    model = KMeans(n_clusters=3, random_state=42, n_init=10) #n_init needs to be defined
    game_kmeans = model.fit(X)
    print(game_kmeans.labels_)

def Feature_Correlation(df):
    #only dataframe that is a numeric value with equal length with global sales
    X1=df[["Year_of_Release", "Critic_Score", "Critic_Count", "User_Score", "User_Count"]] #Sets up the dataframe 
    #Output label
    y=df['Global_Sales']

    scaler = MinMaxScaler()
    X = scaler.fit_transform(X1)

    correlation_result_year = pearsonr(df["Year_of_Release"], y)
    correlation_result_CS = pearsonr(df["Critic_Score"], y)
    correlation_result_CC = pearsonr(df["Critic_Count"], y)
    correlation_result_US = pearsonr(df["User_Score"], y)
    correlation_result_UC = pearsonr(df["User_Count"], y)

    print(correlation_result_year) 
    print(correlation_result_CS) 
    print(correlation_result_CC) 
    print(correlation_result_US) 
    print(correlation_result_UC)

def main():
    df = get_df()
    print("Linear Regression: " + str(linear_regression(df, 5)))
    print("Polynomial Regression: " + str(polynomial_regression(df, 2, 5)))
    Neural_Network(df)
    KMeans_Clustering(df)
    Feature_Correlation(df)
    #matplotlib works here, example of using it, this should open up a new window with the given graph
    # plt.scatter(df["Critic_Score"], df["Global_Sales"])
    # plt.xlabel("Critic Score")
    # plt.ylabel("Global Sales")
    # plt.show()

    # #optuna analysis from https://medium.com/optuna/using-optuna-to-optimize-tensorflow-hyperparameters-57b6d4d316a2
    # study = optuna.create_study(direction="maximize")
    # study.optimize(objective, n_trials=25)
    # pruned_trials = [t for t in study.trials if t.state == optuna.structs.TrialState.PRUNED]
    # complete_trials = [t for t in study.trials if t.state == optuna.structs.TrialState.COMPLETE]
    # print("Study statistics: ")
    # print("  Number of finished trials: ", len(study.trials))
    # print("  Number of pruned trials: ", len(pruned_trials))
    # print("  Number of complete trials: ", len(complete_trials))

    # print("Best trial:")
    # trial = study.best_trial

    # print("  Value: ", trial.value)

    # print("  Params: ")
    # for key, value in trial.params.items():
    #     print("    {}: {}".format(key, value))
    
if __name__ == "__main__":
    main()