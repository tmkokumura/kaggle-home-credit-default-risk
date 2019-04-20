import load_data
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split, StratifiedKFold, ParameterGrid, GridSearchCV
from sklearn.metrics import roc_auc_score
from load_data import load_test_data, load_test_data
from encode import label_encode, one_hot_encode
from logging import getLogger, StreamHandler, Formatter, FileHandler, DEBUG, INFO
import xgboost as xgb

LOG_DIR = '../result/'
SUBMISSION_DIR = '../result/'
ID_LABEL = 'SK_ID_CURR'
TARGET_LABEL = 'TARGET'

logger.info('start')

# log setting
logger = getLogger(__name__)
log_fmt = Formatter('%(asctime)s %(name)s %(lineno)d [%(levelname)s][%(funcName)s] %(message)s')
log_handler = StreamHandler()
log_handler.setLevel(INFO)
log_handler.setFormatter(log_fmt)
logger.addHandler(log_handler)

log_handler = FileHandler(LOG_DIR + 'main.py.log', 'a')
log_handler.setFormatter(log_fmt)
logger.setLevel(DEBUG)
logger.addHandler(log_handler)

# load data
train_data = load_data.load_train_data(debug=False, logger)
logger.info('Train data Loading end: {}'.format(train_data.shape))

test_data = load_data.load_test_data(logger)
logger.info('Test data Lording end : {}'.format(test_data.shape))

# union
train_test = pd.concat([train_data, test_data])

# get categorical column
categorical_cols = [col for col in train_test.columns if not pd.api.types.is_numeric_dtype(train_test[col].dtype)]
logger.debug('categorical column: {}'.format(categorical_cols))
df_categorical_col_num = app_train_test[categorical_cols].nunique(dropna=False)))
logger.debug('categorical column num: {}'.format(df_categorical_col_num))

train_data = label_encode(train_data)
test_data = label_encode(test_data)
logger.info('label encoding end')

train_data, test_data = one_hot_encode(train_data, test_data, TARGET_LABEL)
logger.info('one-hot-encoding end')

train_data = train_data.fillna(0.0)
test_data = test_data.fillna(0.0)
logger.info('zero filling end')

train_x = train_data.drop(TARGET_LABEL, axis=1)
train_y = train_data[TARGET_LABEL].values
use_cols = train_x.columns.values

logger.debug('train_colmuns: {} {}'.format(use_cols.shape, use_cols))

test_x = test_data[use_cols].sort_values('SK_ID_CURR')

logger.info('data preparation end {}'.format(train_x.shape))

logger.info('grid search start')
all_params = {
        'max_depth': [3, 5, 7],
        'min_child_weight': [3, 5, 10],
        'learning_rate': [0.1],
        'n_estimators': [100],
        'colsample_bytree': [0.8, 0.9],
        'colsample_bylevel': [0.8, 0.9],
        'reg_alpha': [0.0, 0.1],
        'max_delta_step': [0.0, 0.1],
        'seed': [0]
        }

clf = xgb.sklearn.XGBClassifier()
grid = GridSearchCV(clf, param_grid=all_params, scoring="roc_auc", return_train_score=True, n_jobs=-1)
grid.fit(train_x, train_y)

logger.info('grid search end')
logger.info('max_auc: {}'.format(grid.best_score_))
logger.info('max_params: {}'.format(grid.best_params_))
logger.debug(pd.DataFrame(grid.cv_results_))

proba = grid.predict_proba(test_x)[:, 1]
submission = pd.DataFrame({
        ID_LABEL : test_x[ID_LABEL],
        TARGET_LABEL : proba
    })
submission.to_csv(SUBMISSION_DIR + 'submission.csv', index=False)
logger.info('end')

