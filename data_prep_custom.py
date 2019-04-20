import numpy as np
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
from logging import getLogger, StreamHandler, Formatter, FileHandler, DEBUG, INFO

IN_TRAIN = '../sql_result/results_20180813234900_train.csv'
IN_TEST = '../sql_result/results_20180813234900_test.csv'

OUT_DIR = '../input/custom/'
OUT_TRAIN_FILE = '20180813234900_train.csv'
OUT_TEST_FILE= '20180813234900_test.csv'

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

# pandas settings
pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", None)
np.set_printoptions(threshold=9999)

logger.info('start')

logger.info('--- load data ---')
in_train = pd.read_csv(IN_TRAIN)
logger.info('in_train: {}'.format(in_train.shape))
in_test = pd.read_csv(IN_TEST)
logger.info('in_test: {}'.format(in_test.shape))
all_data = pd.concat([in_train, in_test], axis=0, sort=False)
del in_train
del in_test
logger.info('all_data: {}'.format(all_data.shape))

logger.info('--- one-hot encoding ---')
all_data = pd.get_dummies(all_data, dummy_na=True)
logger.info('all_data: {}'.format(all_data.shape))

logger.info('--- fill NaN ---')
all_data = all_data.fillna(0)
logger.info('all_data: {}'.format(all_data.shape))

logger.info('--- train test split ---')
train = all_data.iloc[:307511]
logger.info('train: {}'.format(train.shape))
test = all_data.iloc[307511:]
test = test.drop(columns=[TGT_COL])
logger.info('test: {}'.format(test.shape))

logger.info('--- write data---')
train.to_csv(OUT_DIR + OUT_TRAIN_FILE, index=False)
logger.info('data written: ' + OUT_DIR + OUT_TRAIN_FILE)
test.to_csv(OUT_DIR + OUT_TEST_FILE, index=False)
logger.info('data written: ' + OUT_DIR + OUT_TEST_FILE)

logger.info('end')
