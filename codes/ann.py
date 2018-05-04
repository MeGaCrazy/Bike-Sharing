import numpy as np
import pandas as pd
import tensorflow as tf
from keras import backend as k
from keras.layers import Dense
from keras.models import Sequential
from keras.models import load_model
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from keras.optimizers import SGD, Adam, RMSprop


def dataSetAnalysis(df):
    # view starting values of data set
    print("DataSet Head")
    print(df.head(3))
    print("=" * 30)
    # View features in data set
    print("Dataset Features Details")
    print(df.info())
    print("=" * 30)
    # view distribution of numerical features across the data set
    print("Dataset Numerical Features")
    print(df.describe())
    print("=" * 30)
    # view distribution of categorical features across the data set
    # print("Dataset Categorical Features")
    # print(df.describe(include=['O']))
    # print("=" * 30)


def solve():
    # Read the data
    tr = pd.read_csv('../Dataset/reg_train.csv', index_col=['instant', 'dteday'])
    test = pd.read_csv('../Dataset/reg_test.csv', index_col=['instant', 'dteday'])
    # Initialize the X and the y
    X = tr.drop(['cnt', 'casual', 'registered'], axis=1)
    y = tr['cnt']
    # Build the model
    model = Sequential()
    model.add(Dense(2048, activation='relu', input_shape=(X.shape[1],)))
    model.add(Dense(256, activation='relu'))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(1))
    # Compile the Model
    model.compile(optimizer=Adam(lr=0.0001), loss='mse', metrics=['accuracy'])
    model.fit(X, y, validation_split=0.2, verbose=1, epochs=30, shuffle=True,
              callbacks=[EarlyStopping(patience=3)])
    # Get the predictions
    predictions = np.array(model.predict(test), dtype='int')
    # model summary
    model.summary()
    # reformat the output
    test = test.reset_index(level=1, drop=True)
    test = test.drop(
        ['season', 'yr', 'mnth', 'hr', 'holiday', 'weekday', 'workingday', 'weathersit', 'temp', 'atemp', 'hum',
         'windspeed'], axis=1)
    test['cnt'] = predictions
    # Save the data to csv format
    test.to_csv('../output.csv', sep=',')


if __name__ == '__main__':
    dataset = pd.read_csv('../Dataset/reg_train.csv')
    dataSetAnalysis(dataset)
    solve()
