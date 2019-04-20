from tqdm import tqdm
import numpy as np
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
from logging import getLogger, StreamHandler, Formatter, FileHandler, DEBUG, INFO

IN_DIR = '../sql_result/'
IN_TRAIN_FILES = ['201808231336_train-000000000000.csv',
#                 '201808231336_train-000000000001.csv',
#                 '201808231336_train-000000000002.csv',
#                 '201808231336_train-000000000003.csv',
#                 '201808231336_train-000000000004.csv',
#                 '201808231336_train-000000000005.csv',
#                 '201808231336_train-000000000006.csv',
#                 '201808231336_train-000000000007.csv',
#                 '201808231336_train-000000000008.csv',
                 '201808231336_train-000000000009.csv']
IN_TEST_FILE = '201808231336_test.csv'

OUT_DIR = '../input/custom/'
OUT_TRAIN_FILE = '201808231336_train_norm.csv'
OUT_TEST_FILE= '201808231336_test_norm.csv'

KEY_COL = 'SK_ID_CURR'
TGT_COL = 'TARGET'


LOG_DIR = '../log/'
LOG_FILE = 'data_prep.py.log'

# Log Settings
logger = getLogger(__name__)
log_fmt = Formatter('%(asctime)s %(name)s %(lineno)d [%(levelname)s][%(funcName)s] %(message)s')
log_handler = StreamHandler()
log_handler.setLevel(INFO)
log_handler.setFormatter(log_fmt)
logger.addHandler(log_handler)
log_handler = FileHandler(LOG_DIR + LOG_FILE, 'a')
log_handler.setFormatter(log_fmt)
logger.setLevel(DEBUG)
logger.addHandler(log_handler)

logger.info('start data_prep.py')

logger.info('--- load data ---')
logger.info('loading: ' + IN_TRAIN_FILES[0])
train = pd.read_csv(IN_DIR + IN_TRAIN_FILES[0])
del IN_TRAIN_FILES[0]
for in_train_file in tqdm(IN_TRAIN_FILES):
    logger.info('loading: ' + in_train_file)
    tmp_train = pd.read_csv(IN_DIR + in_train_file)
    train = pd.concat([train, tmp_train], axis=0, sort=False, ignore_index=True)
logger.info('in_train: {}'.format(train.shape))

logger.info('in_test file: ' + IN_TEST_FILE)
test = pd.read_csv(IN_DIR + IN_TEST_FILE)
logger.info('in_test: {}'.format(test.shape))

logger.info('--- fill NaN ---')
train = train.fillna(0)
logger.info('train: {}'.format(train.shape))
test = test.fillna(0)
logger.info('test: {}'.format(test.shape))

logger.info('--- normalization ---')
key_train = train[KEY_COL]
target_train = train[TGT_COL]
x_train = train.drop(columns=[KEY_COL, TGT_COL])
columns = x_train.columns
logger.info(KEY_COL + ' and ' + TGT_COL + ' removed from train: {}'.format(x_train.shape))

key_test = test[KEY_COL]
x_test = test.drop(columns=[KEY_COL])
logger.info(KEY_COL + ' removed from test: {}'.format(x_test.shape))

scaler = MinMaxScaler()
logger.info('fitting and transforming x_train...')
train = pd.DataFrame(scaler.fit_transform(x_train), columns=columns)
logger.info('transforming x_test...')
test = pd.DataFrame(scaler.transform(x_test), columns=columns)

train[KEY_COL] = key_train
train[TGT_COL] = target_train
logger.info(KEY_COL + ' and ' + TGT_COL + ' added to train: {}'.format(train.shape))
test[KEY_COL] = key_test
logger.info(KEY_COL + ' added to test: {}'.format(test.shape))


logger.info('--- save data---')
logger.info('saving: ' + OUT_DIR + OUT_TRAIN_FILE)
train.to_csv(OUT_DIR + OUT_TRAIN_FILE, index=False)
logger.info('saving: ' + OUT_DIR + OUT_TEST_FILE)
test.to_csv(OUT_DIR + OUT_TEST_FILE, index=False)

logger.info('end data_prep.py')
