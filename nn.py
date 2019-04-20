from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.normalization import BatchNormalization
from keras.callbacks import Callback
from keras.optimizers import Adam
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm
from statistics import mean
import numpy as np
import gc
# import utils

from logging import getLogger, StreamHandler, Formatter, FileHandler, DEBUG, INFO

TRAIN_FILE = '../input/custom/201808231336_train_norm.csv'
TEST_FILE = '../input/custom/201808231336_test_norm.csv'
SUBMISSION_FILE = '../result/submission.csv'
LOG_FILE = '../log/nn.py.log'

KEY_COL = 'SK_ID_CURR'
TGT_COL = 'TARGET'

# Log Settings
logger = getLogger(__name__)
log_fmt = Formatter('%(asctime)s %(name)s %(lineno)d [%(levelname)s][%(funcName)s] %(message)s')
log_handler = StreamHandler()
log_handler.setLevel(INFO)
log_handler.setFormatter(log_fmt)
logger.addHandler(log_handler)
log_handler = FileHandler(LOG_FILE, 'a')
log_handler.setFormatter(log_fmt)
logger.setLevel(DEBUG)
logger.addHandler(log_handler)

logger.info('=== start nn.py ===')

logger.info('--- load data ---')
df_train = pd.read_csv(TRAIN_FILE).sort_values(by=KEY_COL).reset_index(drop=True)
logger.info('df_train: {}'.format(df_train.shape))
df_test = pd.read_csv(TEST_FILE).sort_values(by=KEY_COL).reset_index(drop=True)
logger.info('df_test: {}'.format(df_test.shape))

logger.info('--- X-Y split ---')
x_train = df_train.drop(columns=[KEY_COL, TGT_COL])
logger.info('x_train: {}'.format(x_train.shape))
y_train = df_train[TGT_COL]
logger.info('y_train: {}'.format(y_train.shape))
del df_train

keys = df_test[KEY_COL]
x_test = df_test.drop(columns=[KEY_COL])
logger.info('x_test: {}'.format(x_test.shape))
del df_test

# logger.info('--- reduce memory usage ---')
# x_train = utils.reduce_mem_usage(x_train, logger)
# x_test = utils.reduce_mem_usage(x_test, logger)
# gc.collect()

logger.info('--- definite NN model ---')

M = len(x_train.columns)        # 入力データの次元
N = 1                           # 隠れ層の数
U = int(len(x_train.columns))        # 隠れ層の次元数
K = 1                           # 出力クラス数

kernel_initializer = 'truncated_normal'     # 重みの初期化方法
ih_activation = 'tanh'                      # 入力層・隠れ層の活性化関数
o_activation = 'sigmoid'                    # 出力層の活性化関数
dropout = 0.5                               # 入力層・隠れ層のドロップアウト率

model = Sequential()                        # モデルの初期化

# 入力層 - 隠れ層を生成
model.add(Dense(input_dim=M, units=U, kernel_initializer=kernel_initializer))
model.add(BatchNormalization())
model.add(Activation(ih_activation))
model.add(Dropout(dropout))

# 隠れ層 - 隠れ層を生成
for i in range(N - 1):
    model.add(Dense(input_dim=U, units=U, kernel_initializer=kernel_initializer))
    model.add(BatchNormalization())
    model.add(Activation(ih_activation))
    model.add(Dropout(dropout))

# 隠れ層 - 出力層を生成
model.add(Dense(input_dim=U, units=K, kernel_initializer=kernel_initializer))
model.add(Activation(o_activation))

loss = 'binary_crossentropy'                      # 誤差関数
lr = 0.001                          # Adamパラメータ lr
beta_1 = 0.9                        # Adamパラメータ beta_1
beta_2 = 0.999                      # Adamパラメータ beta_2
metrics = 'accuracy'                   # 計測するメトリクス

model.compile(loss=loss, optimizer=Adam(lr=lr, beta_1=beta_1, beta_2=beta_2), metrics=[metrics])
logger.info(model.summary())

logger.info('--- cross validation ---')
batch_size = 1000   # バッチサイズ
epochs = 10         # エポック数

kfold = StratifiedKFold(n_splits=3, shuffle=True, random_state=0)

list_auc_trn = []
list_auc_val = []
for trn, val in kfold.split(x_train, y_train):
    x_trn = x_train.iloc[trn, :]
    x_val = x_train.iloc[val, :]
    y_trn = y_train[trn]
    y_val = y_train[val]

    logger.info('fitting to trn data...')
    model.fit(x_trn, y_trn, epochs=epochs, batch_size=batch_size, verbose=1)

    logger.info('predicting trn data...')
    pred_trn = model.predict_proba(x_trn, batch_size=1000, verbose=1).flatten()
    auc_trn = roc_auc_score(y_trn, pred_trn)
    list_auc_trn.append(auc_trn)
    logger.info('auc_trn: {}'.format(auc_trn))

    logger.info('predicting val data...')
    pred_val = model.predict_proba(x_val, batch_size=1000, verbose=1).flatten()
    auc_val = roc_auc_score(y_val, pred_val)
    list_auc_val.append(auc_val)
    logger.info('auc_val: {}'.format(auc_val))

mean_auc_trn = mean(list_auc_trn)
logger.info('mean auc of trn: {}'.format(mean_auc_trn))
mean_auc_val = mean(list_auc_val)
logger.info('mean auc of val: {}'.format(mean_auc_val))

logger.info('--- predict test data ---')
logger.info('fitting to train data...')
model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, verbose=1)
logger.info('predicting test data...')
pred_test = model.predict_proba(x_test, batch_size=1000, verbose=1).flatten()

logger.info('--- save submission file ---')
df_submission = pd.DataFrame({
                            KEY_COL: keys,
                            TGT_COL: pred_test
                        })
logger.info('saving: ' + SUBMISSION_FILE)
df_submission.to_csv(SUBMISSION_FILE, index=False)

logger.info('=== end nn.py ===')

