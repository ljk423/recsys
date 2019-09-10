import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.metrics import log_loss, roc_auc_score, accuracy_score, recall_score, precision_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, Normalizer

from deepCTR.models import DeepFM
from deepCTR.models import DCN
from deepCTR.models import xDeepFM
from deepCTR.inputs import  SparseFeat, DenseFeat,get_fixlen_feature_names

if __name__ == "__main__":
    # original preprocessing part
    data = pd.read_csv('./train.txt')

    sparse_features = ['C' + str(i) for i in range(1, 27)]
    dense_features = ['I' + str(i) for i in range(1, 14)]

    data[sparse_features] = data[sparse_features].fillna('-1', )
    data[dense_features] = data[dense_features].fillna(0, )
    target = ['label']

    # 1.Label Encoding for sparse features,and do simple Transformation for dense features
    for feat in sparse_features:
        lbe = LabelEncoder()
        data[feat] = lbe.fit_transform(data[feat])
    # mms = MinMaxScaler(feature_range=(0, 1))
    # nor = Normalizer()
    # data[dense_features] = mms.fit_transform(data[dense_features])
    data[data[dense_features]<0] = 0                                           #For log transform
    data[dense_features] = np.log(data[dense_features]+1).astype(np.float32)   #log transformation
    # data[dense_features] = nor.fit_transform(data[dense_features])           #Normalize

    # 2.count #unique features for each sparse field,and record dense feature field name

    fixlen_feature_columns = [SparseFeat(feat, data[feat].nunique())
                           for feat in sparse_features] + [DenseFeat(feat, 1,)
                          for feat in dense_features]
    dnn_feature_columns = fixlen_feature_columns
    linear_feature_columns = fixlen_feature_columns

    fixlen_feature_names = get_fixlen_feature_names(linear_feature_columns + dnn_feature_columns)

    # 3.generate input data for model

    # train, test = train_test_split(data, test_size=0.2)
    indices = np.arange(len(data))
    print(len(data))
    indices = np.array_split(indices, 7)
    for i in range(len(indices)):
        indices[i] = np.random.permutation(indices[i])

    train_indices = np.concatenate(indices[:-1])
    test_indices = indices[-1]
    val_indices, test_indices = np.array_split(test_indices, 2)
    train_indices = np.random.permutation(train_indices)

    train = data.loc[train_indices]
    test = data.loc[test_indices]

    train_model_input = [train[name] for name in fixlen_feature_names]
    test_model_input = [test[name] for name in fixlen_feature_names]

    deepFM model
    model = DeepFM(linear_feature_columns, dnn_feature_columns, embedding_size = 5,
                   dnn_dropout=0.9, dnn_hidden_units=(200,200,200), task='binary')
    model.compile("adam", "binary_crossentropy",
                  metrics=['binary_crossentropy'])
    callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)
    history = model.fit(train_model_input, train[target].values,callbacks=[callback],
                        batch_size=1024, epochs=10, verbose=2, validation_split=0.2)
    pred_ans = model.predict(test_model_input, batch_size=1024)
    print("test LogLoss", round(log_loss(test[target].values, pred_ans), 4))
    print("test AUC", round(roc_auc_score(test[target].values, pred_ans), 4))
    print("test ACC", round(accuracy_score(test[target].values, pred_ans.round()), 4))
    print("test Precision", round(precision_score(test[target].values, pred_ans.round()), 4))
    print("test Recall", round(recall_score(test[target].values, pred_ans.round()), 4))
    pd.DataFrame(test[target].values).to_csv('./y_true')
    pd.DataFrame(pred_ans).to_csv('./y_pred')

    # DCN model
    # model =  DCN(dnn_feature_columns, dnn_hidden_units = (1024,1024), dnn_dropout=0.9)
    # model.compile("adam", "binary_crossentropy",
    #               metrics=['binary_crossentropy'])
    # callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)
    # history = model.fit(train_model_input, train[target].values, callbacks=[callback],
    #                     batch_size=512, epochs=10, verbose=2, validation_split=0.2)
    # pred_ans = model.predict(test_model_input, batch_size=512)
    # print("test LogLoss", round(log_loss(test[target].values, pred_ans), 4))
    # print("test AUC", round(roc_auc_score(test[target].values, pred_ans), 4))
    # print("test ACC", round(accuracy_score(test[target].values, pred_ans.round()), 4))

    #XdeepFM model
    model = xDeepFM(linear_feature_columns, dnn_feature_columns, embedding_size = 8,
                    dnn_dropout=0.5, dnn_hidden_units=(200,200,200), cin_layer_size=(200,200,200), task='binary',
                    init_std=0.0001, l2_reg_linear=0.0001, l2_reg_embedding=0.0001)
    model.compile("adam", "binary_crossentropy",
                  metrics=['binary_crossentropy'])
    callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)
    history = model.fit(train_model_input, train[target].values, callbacks=[callback],
                        batch_size=1024, epochs=10, verbose=2, validation_split=0.2)
    pred_ans = model.predict(test_model_input, batch_size=1024)
    print("test LogLoss", round(log_loss(test[target].values, pred_ans), 4))
    print("test AUC", round(roc_auc_score(test[target].values, pred_ans), 4))
    print("test ACC", round(accuracy_score(test[target].values, pred_ans.round()), 4))
    print("test Precision", round(precision_score(test[target].values, pred_ans.round()), 4))
    print("test Recall", round(recall_score(test[target].values, pred_ans.round()), 4))
