import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import pandas as pd
import tensorflow as tf
import keras
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from keras.wrappers.scikit_learn import KerasClassifier

from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import KFold,cross_val_score
from keras import models,layers

class Automobile:
    def __init__(self, data_path, name_path):
        self.data_path = data_path
        self.name_path = name_path
        self.feature_names = self.get_names()
        self.data, self.target = self.preprocess_data()

    def preprocess_data(self):
        file = open(self.data_path, 'r').readlines()
        file = [x.strip() for x in file]

        full_data, target = [], []

        for f in file:
            # skip the samples with the missing values
            if "?" in f:
                continue
            split_f = f.split(',')
            target.append(float(split_f[1]))
            full_data.append(self.str2int(split_f))

        normalized = preprocessing.normalize(full_data)

        # target = np.reshape(target,newshape=[None,1])

        print("Data samples: {} labels: {}".format(np.shape(full_data), np.shape(target)))

        return normalized, target


    def get_names(self):
        lines = open('feature.txt','r').readlines()
        names = {}
        for line in lines:
            name = line.split(':')[0].split()[1]

            data = line.split(':')[1]
            if not 'continuous' in data:
                data = data.strip().split(', ')
            names[name] = data

        return names

    def str2int(self,arr):
        '''
        :param arr:
        :return: string to integers
        '''
        info = self.get_names()
        data_arr = []
        for data,(key,value) in zip(arr, info.items()):
            try:
                data = float(data)
            except:
                step = 1/len(value)
                data = value.index(data) * float(step)

            data_arr.append(data)

        assert len(data_arr) == 26
        return data_arr[2:]

def build_model():
    model = keras.Sequential([
        layers.Dense(20, activation='relu', input_shape =[24]),
        layers.Dense(10, activation='relu'),
        layers.Dense(1)
    ])
    opt = keras.optimizers.Adam(learning_rate=0.001)
    model.compile(loss='mse', optimizer=opt, metrics=['mse'])

    return model

def main():
    filedata, filenames = 'imports-85.data', 'imports-85.names'

    auto = Automobile(filedata, filenames)
    x_data, x_test, y_data, y_test =  train_test_split(auto.data, auto.target, test_size=0.1)

    # x_data, y_data = auto.data, auto.target
    y_data = np.array(y_data)
    model = build_model()
    # model.summary()

    N = 10
    kf = KFold(N,shuffle=False)
    mse = []
    for train_set, test_set in kf.split(x_data):
        x_train, x_test = x_data[train_set], x_data[test_set]
        y_train, y_test = y_data[train_set], y_data[test_set]

        model.fit(x_train, y_train, epochs=200)
        y_pred = model.predict(x_test)

        mse_fold = np.mean(np.square(y_pred-y_test))
        mse.append(mse_fold)

    y_pred = model.predict(x_test)

    print("\n{}-Fold loss : {}".format(N, np.mean(mse)))
    for i,j in zip(y_pred, y_test):
        print("test: pred {:.4f} target {} ".format(i[0],j))


if __name__=='__main__':
    main()