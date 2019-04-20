from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.normalization import BatchNormalization
from keras.callbacks import Callback
from keras.optimizers import Adam
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_score, StratifiedKFold
from keras.wrappers.scikit_learn import KerasClassifier
from tqdm import tqdm
from statistics import mean
import numpy as np
import gc
from logging

# import utils
TRAIN_FILE = '../input/custom/201808231336_train_norm_debug.csv'
TEST_FILE = '../input/custom/201808231336_test_norm_debug.csv'
SUBMISSION_FILE = '../result/submission.csv'
LOG_FILE = '../log/train_nn.py.log'

KEY_COL = 'SK_ID_CURR'
TGT_COL = 'TARGET'

# Logger Settings.
fmt = '%(asctime)s %(levelname)s [%(module)s#%(funcName)s] :%(message)s'
logging.basicConfig(level=logging.DEBUG, format=fmt)
logger = logging.getLogger('Main')

def gen_model():
    """
    generate keras model
    :return: model
    """
    M = len(x_train.columns)        # 入力データの次元
    N = 1                           # 隠れ層の数
    U = len(x_train.columns)        # 隠れ層の次元数
    K = 1                           # 出力クラス数

    kernel_initializer = 'truncated_normal'     # 重みの初期化方法
    ih_activation = 'relu'                      # 入力層・隠れ層の活性化関数
    o_activation = 'sigmoid'                    # 出力層の活性化関数
    dropout = 0.5                               # 隠れ層のドロップアウト率

    model = Sequential()                        # モデルの初期化

    # 入力層 - 隠れ層を生成
    model.add(Dense(input_dim=M, units=U, kernel_initializer=kernel_initializer))
    model.add(BatchNormalization())
    model.add(Activation(ih_activation))

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
    return model
    

if __name__ == '__main__':

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

	logger.info('--- cross validation ---')
	epochs = 20
	batch_size = 1000
	clf = KerasClassifier(build_fn=gen_model, epochs=epochs, batch_size=batch_size)
	kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
	cv_auc = cross_val_score(clf, x_train, y_train, cv=kfold, scoring='roc_auc', verbose=3)
	logger.info('auc of each cv: {}'.format(cv_auc))
	mean_auc = mean(cv_auc)
	logger.info('mean auc: {}'.format(mean_auc))

	logger.info('fitting to train data...')
	clf.fit(x_train, y_train)

	logger.info('predicting test data...')
	pred_test = clf.predict_proba(x_test, batch_size=1000, verbose=1)[:, 1]

	logger.info('--- save submission file ---')
	df_submission = pd.DataFrame({
	                            KEY_COL: keys,
	                            TGT_COL: pred_test
	                        })
	logger.info('saving: ' + SUBMISSION_FILE)
	df_submission.to_csv(SUBMISSION_FILE, index=False)

	logger.info('=== end nn.py ===')

