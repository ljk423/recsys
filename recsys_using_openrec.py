#  Copyright (c) 2019. Lorem ipsum dolor sit amet, consectetur adipiscing elit.
#  Morbi non lorem porttitor neque feugiat blandit. Ut vitae ipsum eget quam lacinia accumsan.
#  Etiam sed turpis ac ipsum condimentum fringilla. Maecenas magna.
#  Proin dapibus sapien vel ante. Aliquam erat volutpat. Pellentesque sagittis ligula eget metus.
#  Vestibulum commodo. Ut rhoncus gravida arcu.
#  Created by LEEJUNKI
#  Copyright © 2019 LEEJUNKI. All rights reserved.
#  github :: https://github.com/ljk423

from tensorflow.data import Dataset
from openrec import DLRM
from tensorflow.keras import optimizers
from tqdm import tqdm
import pandas as pd
import tensorflow as tf
import dataloader

raw_data = dataloader.load_criteo('/Users/hza582/PycharmProjects/dlrm_tensorflow/dataset/')
dim_embed = 16 #default = 4
bottom_mlp_size = [512,256,64,16] #default = 8,4
top_mlp_size = [512,256,1] #default = 128,64,1
total_iter = int(1e5) # default = 1e5
batch_size = 2048
eval_interval = 100
save_interval = eval_interval
tr_it = len(raw_data['y_train'])//batch_size
ts_it = len(raw_data['y_val'])//batch_size

# Sample 1000 batches for training
train_dataset = Dataset.from_tensor_slices({
    'dense_features': raw_data['X_int_train'][:batch_size * tr_it], #default=1000
    'sparse_features': raw_data['X_cat_train'][:batch_size * tr_it], #default=1000
    'label': raw_data['y_train'][:batch_size * tr_it] #default=1000
}).batch(batch_size).prefetch(1).shuffle(5 * batch_size).repeat(20)

# Sample 100 batches for validation
val_dataset = Dataset.from_tensor_slices({
    'dense_features': raw_data['X_int_val'][:batch_size * ts_it], #default=100
    'sparse_features': raw_data['X_cat_val'][:batch_size * ts_it], #default=100
    'label': raw_data['y_val'][:batch_size * ts_it] #default=100
}).batch(batch_size)

optimizer = optimizers.Adam() ##default = adam ,lr=0.001

dlrm_model = DLRM(
    m_spa=dim_embed,
    ln_emb=raw_data['counts'],
    ln_bot=bottom_mlp_size,
    ln_top=top_mlp_size
)

auc = tf.keras.metrics.AUC()
acc = tf.keras.metrics.BinaryAccuracy()

@tf.function
def train_step(dense_features, sparse_features, label):
    with tf.GradientTape() as tape:
        loss_value = dlrm_model(dense_features, sparse_features, label)
    gradients = tape.gradient(loss_value, dlrm_model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, dlrm_model.trainable_variables))
    return loss_value


@tf.function
def eval_step(dense_features, sparse_features, label):
    pred = dlrm_model.inference(dense_features, sparse_features)
    auc.update_state(y_true=label, y_pred=pred)
    acc.update_state(y_true=label, y_pred=pred)


average_loss = tf.keras.metrics.Mean()
import time
for train_iter, batch_data in enumerate(train_dataset):
    if train_iter // tr_it == 0 and train_iter % tr_it == 0:
        start = time.time()
    loss = train_step(**batch_data)
    average_loss.update_state(loss)
    print('%d iter training.' % train_iter, end='\r')
    if train_iter % tr_it == 0 and train_iter//tr_it != 0:
        end = time.time()
        print("************************ Epoch {} Completed!, {} sec ************************".format(train_iter//tr_it, round(end-start,2)))
        start = time.time()
    if train_iter % tr_it == 0: # % eval_interval
        for eval_batch_data in tqdm(val_dataset,
                                    leave=False): #desc='%d iter evaluation' % train_iter
            eval_step(**eval_batch_data)
        print("Iter: %d, Loss: %.4f, AUC: %.4f, ACC: %.4f" % (train_iter,
                                                   average_loss.result().numpy(),
                                                   auc.result().numpy(), acc.result()))

        average_loss.reset_states()
        auc.reset_states()
        acc.reset_states()

pred_ = dlrm_model.inference(raw_data['X_int_test'], raw_data['X_cat_test'])
pd.DataFrame(pd.DataFrame(raw_data['y_test'])).to_csv('./y_true')
pd.DataFrame(pd.DataFrame(pred_)).to_csv('./y_pred')
