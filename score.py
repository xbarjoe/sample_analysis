import tensorflow as tf
import keras
import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense
from sklearn.model_selection import StratifiedKFold, RepeatedStratifiedKFold
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_auc_score, make_scorer
from sklearn.utils import class_weight
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif
import matplotlib.pyplot as plt
import seaborn as sns
import itertools
import sys



def run_model(test_data):
    """
    Returns the predicted values when test_data is passed to model.h5
    model.h5 is a simple logistic regression model.
    """
    
    #Load the saved model and pass the features through it, getting the output probabilities.
    model = load_model("model.h5")
    y_pred = model.predict(test_data)
    
    return y_pred



def pre_process(dataset):
    """
    Returns the testing data, cleaned in a way that it will be digestable by the logistic regression model.
    Involves feature selection, KNN Inputer for missing values, and a MinMax Scaler for the normalization of linear values
    """

    #Isolate the desired features from the original dataset
    isolated_features = dataset[['DataSource1_Feature1','DataSource2_Feature1','DataSource3_Feature2','DataSource3_Feature3','DataSource4_Feature6','DataSource4_Feature5']]
    isolated_features = pd.get_dummies(isolated_features, columns=['DataSource1_Feature1','DataSource4_Feature6','DataSource3_Feature3'], dtype=float)

    #Use K-NN to fill in any missing values in the dataset
    imputer = KNNImputer(n_neighbors=5)
    isolated_features = pd.DataFrame(imputer.fit_transform(isolated_features), columns = isolated_features.columns)

    #Scale the linear values so they end up normalized
    scaler = MinMaxScaler()
    isolated_features = pd.DataFrame(scaler.fit_transform(isolated_features), columns = isolated_features.columns)

    #Convert DataFrame to Numpy array and return the array
    X = isolated_features.to_numpy()
    return X
    
    
def save_results(data, Y):
    """
    Extracts the ID from the processed dataset, extracts the 'ID' column, and appends the predicted values from the logistic regression model.
    The resulting DataFrame is then saved to the disk as a csv.
    Returns: Nothing
    """
    
    #Extract the ID from the original dataset.
    df = pd.DataFrame(data['ID'])
    
    #Append the values from the logistic regression model to the dataframe and save it to the disk.
    df.insert(1, "Score", Y)
    df.to_csv("./score.csv", sep=',')
    
    
def main():
    """
    Main driver code for the classifier.
    Loads the data, and then passes the required pieces of data around to be processed / saved.
    """
    
    #Grab dataset path from command line arguments
    data = pd.read_csv(str(sys.argv[1]), delimiter = ",")
    
    X = pre_process(data)
    Y = run_model(X)
    save_results(data,Y)
    
if __name__ == "__main__":
    main()