from tqdm import tqdm
import numpy as np
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
from logging import getLogger, StreamHandler, Formatter, FileHandler, DEBUG, INFO

IN_DIR = '../sql_result/'
IN_TRAIN_FILES = ['201808141735_train-000000000000.csv',
                 '201808141735_train-000000000001.csv',
                 '201808141735_train-000000000002.csv',
                 '201808141735_train-000000000003.csv',
                 '201808141735_train-000000000004.csv',
                 '201808141735_train-000000000005.csv']
IN_TEST_FILE = '201808141735_test.csv'

OUT_DIR = '../input/custom/'
OUT_TRAIN_FILE = '201808141735_train.csv'
OUT_TEST_FILE= '201808141735_test.csv'

KEY_COL = 'SK_ID_CURR'
TGT_COL = 'TARGET'


LOG_DIR = '../log/'
LOG_FILE = 'data_prep_custom2.py.log'

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

# pandas settings
pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", None)
np.set_printoptions(threshold=9999)

logger.info('start')

logger.info('--- load data ---')
logger.info('train file: ' + IN_TRAIN_FILES[0])
train = pd.read_csv(IN_DIR + IN_TRAIN_FILES[0])
del IN_TRAIN_FILES[0]
for in_train_file in tqdm(IN_TRAIN_FILES):
    logger.info('train file: ' + in_train_file)
    tmp_train = pd.read_csv(IN_DIR + in_train_file)
    train = pd.concat([train, tmp_train], axis=0, sort=False)
logger.info('in_train: {}'.format(train.shape))

logger.info('in_test file: ' + IN_TEST_FILE)
test = pd.read_csv(IN_DIR + IN_TEST_FILE)
logger.info('in_test: {}'.format(test.shape))

logger.info('--- fill NaN ---')
train = train.fillna(0)
logger.info('train: {}'.format(train.shape))
test = test.fillna(0)
logger.info('test: {}'.format(test.shape))

logger.info('--- drop target col from test data ---')
test = test.drop(columns=[TGT_COL])
logger.info('test: {}'.format(test.shape))

logger.info('--- write data---')
train.to_csv(OUT_DIR + OUT_TRAIN_FILE, index=False)
logger.info('data written: ' + OUT_DIR + OUT_TRAIN_FILE)
test.to_csv(OUT_DIR + OUT_TEST_FILE, index=False)
logger.info('data written: ' + OUT_DIR + OUT_TEST_FILE)

logger.info('end')
