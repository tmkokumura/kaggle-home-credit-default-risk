import load_data
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split, StratifiedKFold, ParameterGrid
from sklearn.metrics import roc_auc_score
from load_data import load_test_data, load_test_data
from encode import label_encode, one_hot_encode
from logging import getLogger, StreamHandler, Formatter, FileHandler, DEBUG, INFO
import xgboost as xgb

LOG_DIR = '../result/'
SUBMISSION_DIR = '../result/'
ID_LABEL = 'SK_ID_CURR'
TARGET_LABEL = 'TARGET'

'''
Log Settings
'''
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

logger.info('start')
train_data = load_data.load_train_data(debug=True)
logger.info('Train data Loading end: {}'.format(train_data.shape))

test_data = load_data.load_test_data()
logger.info('Test data Lording end : {}'.format(test_data.shape))

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
        'n_estimators': [10000],
        'colsample_bytree': [0.8, 0.9],
        'colsample_bylevel': [0.8, 0.9],
        'reg_alpha': [0.0, 0.1],
        'max_delta_step': [0.0, 0.1],
        'seed': [0]
        }

max_auc = 0.0
max_params = None 
for params in tqdm(list(ParameterGrid(all_params))):

    logger.debug('  params: {}'.format(params) )
    cv = StratifiedKFold(n_splits=2, shuffle=True, random_state=0)

    auc_val_list = []
    best_iteration_list = []
    for train_idx, test_idx in cv.split(train_x, train_y):
        trn_x = train_x.iloc[train_idx, :]
        val_x = train_x.iloc[test_idx, :]

        trn_y = train_y[train_idx]
        val_y = train_y[test_idx]

        clf_grid = xgb.sklearn.XGBClassifier(**params)
        clf_grid.fit(
                trn_x, 
                trn_y, 
                eval_set=[(val_x, val_y)], 
                early_stopping_rounds=100,
                eval_metric='auc')

        pred_grid = clf_grid.predict_proba(val_x, ntree_limit=clf_grid.best_iteration+ 1)[:, 1]
        auc_val = roc_auc_score(val_y, pred_grid)
        auc_val_list.append(auc_val)
        best_iteration_list.append(clf_grid.best_iteration)
        logger.debug('    auc_val: {}'.format(auc_val))

    params['n_estimators'] = int(np.mean(best_iteration_list))
    avg_auc = np.mean(auc_val_list)

    if avg_auc > max_auc:
        max_auc = avg_auc
        max_params = params

    logger.debug('  cross validation end. avg_auc: {}'.format(avg_auc))
    logger.info('  max_auc: {}'.format(max_auc))
    logger.info('  max_params: {}'.format(max_params))

logger.info('grid search end')


logger.info('max_auc: {}'.format(max_auc))
logger.info('max_params: {}'.format(max_params))

logger.info('test data fit start')
clf = xgb.sklearn.XGBClassifier(**max_params)
clf.fit(train_x, train_y)
logger.info('test data fit end')

proba = clf.predict_proba(test_x)[:, 1]
submission = pd.DataFrame({
        ID_LABEL : test_x[ID_LABEL],
        TARGET_LABEL : proba
    })
submission.to_csv(SUBMISSION_DIR + 'submission.csv', index=False)
logger.info('end')

