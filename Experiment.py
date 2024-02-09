# Name: Experiment.py
# Auth: Kareem T
# Date: 2/8/24
# Desc: Experiment implementation
import pandas as pd
from time import time
from sklearnex import patch_sklearn # Patch SKLearn using Intel extension for PERFORMANCE BOOST
patch_sklearn()
from sklearn import svm
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.preprocessing import StandardScaler

class Experiment:

    @staticmethod
    def load_data(filepath):
        """ Load data, partition into testing/training data"""
        return pd.read_csv(filepath)

    @staticmethod
    def process_data(df, outliers = False):
        '''Convert a DataFrame with catageorical labels to one_hot encoded DataFrame'''
        if outliers:
            df = df[(df != 0).all(axis=1)]
            df = df[df['price'] > df['price'].quantile(0.03)]
            df = df[df['price'] < df['price'].quantile(0.97)]

        transmission = pd.get_dummies(df.transmission) # convert categorical variables to one-hot encoding
        models = pd.get_dummies(df.model)
        fuelType = pd.get_dummies(df.fuelType)

        onehot = pd.concat([transmission, fuelType, models], axis=1) # one_hot df with all categorical vars converted

        df = df.drop(['model', 'transmission', 'fuelType'], axis=1) # Drop the existing categorical labels to be replaced
        df = pd.concat([df, onehot], axis=1)                        # With one_hot encoding
        df = df.dropna()
        return df

    @staticmethod
    def train_test_split_data(df):
        '''Split a one_hot encoded DataFrame into Train/Test splits'''
        train = df.sample(frac=0.8)
        test = df[~df.index.isin(train.index)]
        x_train = train.drop('price', axis=1).values
        x_test = test.drop('price', axis=1).values
        y_train = train.iloc[:,1].values
        y_test = test.iloc[:,1].values

        scaler = StandardScaler().fit(x_train)
        x_train = scaler.transform(x_train)
        x_test = scaler.transform(x_test)
        return x_train, x_test, y_train, y_test

    @staticmethod
    def fit(x,y):
        '''Fit training data to Support Vector Machine'''
        model = svm.SVR(kernel='linear')
        model.fit(x,y)
        return model

    @staticmethod
    def predict(model, x, y_test):
        '''Make prediction using SVM model, print accuracy'''
        y_pred = model.predict(x).astype(int)
        Experiment.accuracy(y_test, y_pred)
        return y_pred

    @staticmethod
    def run(file_path):
        """Load data, set up/train ML model, predict from model using unseen data, assess and print model accuracy"""
        start_time = time()
        df = Experiment.load_data(file_path)
        onehot = Experiment.process_data(df)
        train_X, test_X, train_y, test_y = Experiment.train_test_split_data(onehot)
        model = Experiment.fit(train_X, train_y)
        Experiment.predict(model, test_X, test_y)
        print(f'Runtime: {time() - start_time:.3f} sec')
        print()

    @staticmethod
    def accuracy(act, pred):
        '''Print MAPE statistics'''
        mape = mean_absolute_percentage_error(act, pred)
        print(f"Some model predictions: {pred[0:2]}")
        print(f"Some actual predictions: {act[0:2]}")
        print("Mean absolute percentage error: ", mape)
        print(f"The model's prediction are roughly {(1-mape)*100:.2f}% accurate")

class Grand_Experiment(Experiment):

    def __init__(self):
        self.df = pd.DataFrame()

    def add_data(self, file_name):
        df = self.load_data(file_name)
        if 'tax(£)' in df.columns: df.rename(columns = {'tax(£)': 'tax'}, inplace = True)
        self.df = pd.concat([self.df, df], axis = 0, ignore_index = True)

    def run(self, debug = False):
        """Load data, set up/train ML model, predict from model using unseen data, assess and print model accuracy"""
        start_time = time.time()
        df = self.df # Model Training
        onehot = self.process_data(df)
        if debug: self.debug(df)
        train_X, test_X, train_y, test_y = self.train_test_split_data(onehot)

        model = self.fit(train_X, train_y) # Model Prediction
        self.predict(model, test_X, test_y)
        print(f'Runtime: {time() - start_time:.3f} sec')
        print()

    def debug(df):
        '''Prints the distribution (shape and null values) for each column'''
        # Iterate over columns and get shape of each array
        for column_name, column_data in df.items():
            column_array = column_data.values
            column_shape = column_array.shape
            print(column_data.isnull().sum())
            print(f"Column {column_name} array shape: {column_shape}")
        print(df.isnull().sum())
