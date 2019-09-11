#  Copyright (c) 2019. Lorem ipsum dolor sit amet, consectetur adipiscing elit.
#  Morbi non lorem porttitor neque feugiat blandit. Ut vitae ipsum eget quam lacinia accumsan.
#  Etiam sed turpis ac ipsum condimentum fringilla. Maecenas magna.
#  Proin dapibus sapien vel ante. Aliquam erat volutpat. Pellentesque sagittis ligula eget metus.
#  Vestibulum commodo. Ut rhoncus gravida arcu.
#  Created by LEEJUNKI
#  Copyright © 2019 LEEJUNKI. All rights reserved.
#  github :: https://github.com/ljk423

import numpy as np
import xlearn as xl
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, log_loss
import category_encoders as ce
import sys
import time

def load_criteo(dataset_folder='dataset/'): #If use label encoded file

    with np.load(dataset_folder + 'criteo/kaggle_processed.npz') as data:
        X_int = data["X_int"]
        X_cat = data["X_cat"]
        y = data["y"]
        counts = data["counts"]

    temp_cat = pd.DataFrame(X_cat)
    ce_hash = ce.hashing.HashingEncoder(cols=temp_cat.columns.tolist())
    hashed = ce_hash.fit_transform(temp_cat)
    X_cat = hashed.to_numpy()

    indices = np.arange(len(y))
    indices = np.array_split(indices, 7)
    for i in range(len(indices)):
        indices[i] = np.random.permutation(indices[i])

    train_indices = np.concatenate(indices[:-1])
    test_indices = indices[-1]
    val_indices, test_indices = np.array_split(test_indices, 2)
    train_indices = np.random.permutation(train_indices)

    raw_data = dict()

    raw_data['counts'] = counts
    mms = MinMaxScaler(feature_range=(0, 1))
    mms_cat = MinMaxScaler(feature_range=(0, 1))
    # X_cat = mms_cat.fit_transform(X_cat)

    raw_data['X_cat_train'] = X_cat[train_indices].astype(np.int32)
    raw_data['X_int_train'] = np.log(X_int[train_indices] + 1).astype(np.float32)
    raw_data['X_int_train'] = mms.fit_transform(raw_data['X_int_train']).astype(np.float32)
    raw_data['y_train'] = y[train_indices].astype(np.float32)

    raw_data['X_cat_val'] = X_cat[val_indices]
    raw_data['X_int_val'] = np.log(X_int[val_indices] + 1).astype(np.float32)
    raw_data['X_int_val'] = mms.transform(raw_data['X_int_val']).astype(np.float32)
    raw_data['y_val'] = y[val_indices]

    raw_data['X_cat_test'] = X_cat[test_indices]
    raw_data['X_int_test'] = np.log(X_int[test_indices] + 1).astype(np.float32)
    raw_data['X_int_test'] = mms.transform(raw_data['X_int_test']).astype(np.float32)
    raw_data['y_test'] = y[test_indices]

    pd.DataFrame(raw_data).to_csv('./hashed_train.txt', index=False, header=False)
    sys.exit()
    return raw_data

def convert_to_ffm(df, type, numerics, categories, features):
    currentcode = len(numerics)
    catdict = {}
    catcodes = {}
    # Flagging categorical and numerical fields
    for x in numerics:
        catdict[x] = 0
    for x in categories:
        catdict[x] = 1

    nrows = df.shape[0]
    ncolumns = len(features)
    with open(str(type) + "_ffm.txt", "w") as text_file:

    # Looping over rows to convert each row to libffm format
        for n, r in enumerate(range(nrows)):
            datastring = ""
            datarow = df.iloc[r].to_dict()
            datastring += str(int(datarow['label']))
            # For numerical fields, we are creating a dummy field here
            for i, x in enumerate(catdict.keys()):
                if (catdict[x] == 0):
                    datastring = datastring + " " + str(i) + ":" + str(i) + ":" + str(datarow[x])
                else:
                    # For a new field appearing in a training example
                    if (x not in catcodes):
                        catcodes[x] = {}
                        currentcode += 1
                        catcodes[x][datarow[x]] = currentcode  # encoding the feature
                    # For already encoded fields
                    elif (datarow[x] not in catcodes[x]):
                        currentcode += 1
                        catcodes[x][datarow[x]] = currentcode  # encoding the feature
                    code = catcodes[x][datarow[x]]
                    datastring = datastring + " " + str(i) + ":" + str(int(code)) + ":1"

            datastring += '\n'
            text_file.write(datastring)

def LR(X_train, y_train, X_val, y_val):
    linear_model = xl.LRModel(task='binary',
                              epoch=100, lr=0.005,
                              reg_lambda=1.0, opt='adagrad', nthread=8,metric='auc')

    linear_model.fit(X_train, y_train,
                     eval_set=[X_val, y_val])
    y_pred = linear_model.predict(X_val)
    return y_pred

def FM(X_train, y_train, X_val, y_val):
    fm_model = xl.FMModel(task='binary',
                          epoch=100, lr=0.05, reg_lambda=0.00002,
                          k=40, opt='adagrad', nthread=8,metric='auc')  # k = latent factor size

    fm_model.fit(X_train, y_train,
                 eval_set=[X_val, y_val], is_instance_norm=True)

    y_pred = fm_model.predict(X_val)
    return y_pred

def FFM():
    ffm_model = xl.create_ffm()
    ffm_model.setTrain("Train_ffm.txt")
    ffm_model.setValidate("Test_ffm.txt")
    # ffm_model.disableEarlyStop()
    param = {'task': 'binary',  # ‘binary’ for classification, ‘reg’ for Regression
             'k': 4,  # Size of latent factor
             'lr': 0.2,  # Learning rate for GD
             'opt' : 'adagrad',
             'lambda': 0.00002,  # L2 Regularization Parameter
             'metric': 'auc',  # Metric for monitoring validation set performance
             'epoch': 100,  # Maximum number of Epochs
             'nthread': 8,
             'stop_window': 3
             }
    # ffm_model.setSigmoid()
    ffm_model.fit(param, "model.out")
    ffm_model.setTest("./Test_ffm.txt")
    y_pred = ffm_model.predict("./model.out", "./output.txt")
    return y_pred

if __name__ == '__main__':
    # raw_data = load_criteo('/Users/hza582/PycharmProjects/Xlearn/dataset/')

# raw data preprocessing
    start = time.time()
    data = pd.read_csv('./train.txt')

    sparse_features = ['C' + str(i) for i in range(1, 27)]
    dense_features = ['I' + str(i) for i in range(1, 14)]
#
# Handling Missing values
    data[sparse_features] = data[sparse_features].fillna('-1', )
    data[dense_features] = data[dense_features].fillna(0, )
    target = ['label']
#
# # ## Preprocessing for FM,FFM (Hashing Trick)
# # hashing
    data[sparse_features] = pd.DataFrame(data[sparse_features])
    ce_hash = ce.hashing.HashingEncoder(n_components=20, cols=data[sparse_features].columns.tolist())
    x_cat = ce_hash.fit_transform(data[sparse_features])
    data = data.drop(sparse_features, axis=1)
    data = pd.concat([data,x_cat], axis=1)
# # log transform
    data[data[dense_features]<0] = 0   # For log transform
    data[dense_features] = np.log(data[dense_features] + 1).astype(np.float32)  # log transformation
#

# train test split
    indices = np.arange(len(data))
    print('# of datapoints : {}'.format(len(data)))
    indices = np.array_split(indices, 7)
    for i in range(len(indices)):
        indices[i] = np.random.permutation(indices[i])

    train_indices = np.concatenate(indices[:-1])
    test_indices = indices[-1]
    val_indices, test_indices = np.array_split(test_indices, 2)
    train_indices = np.random.permutation(train_indices)

    train = data.loc[train_indices]
    test = data.loc[test_indices]

    y_train = train['label']
    X_train = train.drop(['label'], axis=1)
    y_val = test['label']
    X_val = test.drop(['label'], axis=1)
    end = time.time()
    print('Preprocessing time is {} seconds'.format(round(end-start,2)))
# # save preprocessed files
#     full_train = pd.concat([y_train, X_train], axis=1)
#     full_test = pd.concat([y_val, X_val], axis=1)
#     full_train.to_csv('full_train.csv', index=False)
#     full_test.to_csv('full_test.csv', index=False)

# load preprocessed files
#     train = pd.read_csv('full_train.csv')
#     test = pd.read_csv('full_test.csv')
#     y_train = train['label']
#     X_train = train.drop(['label'], axis=1)
#     y_val = test['label']
#     X_val = test.drop(['label'], axis=1)

# Make libfm format file
#     y_train = raw_data['y_train']
#     X_train = np.column_stack([raw_data['X_int_train'],raw_data['X_cat_train']])
#     y_val = raw_data['y_val']
#     X_val = np.column_stack([raw_data['X_int_val'],raw_data['X_cat_val']])

    train = pd.concat([y_train, X_train], axis=1)
    test = pd.concat([y_val, X_val], axis=1)
    categories = train.columns[14:]
    numerics = train.columns[1:14]
    features = train.columns[1:]

    convert_to_ffm(train,'Train',numerics,categories,features)
    convert_to_ffm(test, 'Test', numerics, categories, features)

    pred_LR = LR(X_train,y_train,X_val, y_val)
    pred_FM = FM(X_train,y_train,X_val, y_val)
    pd.DataFrame(pred_LR).to_csv('./pred_LR')
    pd.DataFrame(pred_FM).to_csv('./pred_FM')
    FFM()
    pred_FFM = np.loadtxt('./output.txt')
    pd.DataFrame(y_val).to_csv('./y_true')
    print('LR Scores :::: Loss : {} ACC : {}, AUC : {}'.format(
        log_loss(y_val,pred_LR),
        accuracy_score(y_val, pred_LR.round()),
        roc_auc_score(y_val, pred_LR)
    ))
    print('FM Scores :::: Loss : {} ACC : {}, AUC : {}'.format(
        log_loss(y_val, pred_FM),
        accuracy_score(y_val, pred_FM.round()),
        roc_auc_score(y_val, pred_FM.round()),
    ))
    print('FFM Scores :::: ACC : {}, AUC : {}, Precision : {}, Recall : {}'.format(
        accuracy_score(y_val, pred_FFM.round()),
        roc_auc_score(y_val, pred_FFM),
        precision_score(y_val, pred_FFM.round()),
        recall_score(y_val, pred_FFM.round())
    ))
