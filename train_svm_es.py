import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, ParameterGrid
from sklearn.metrics import roc_auc_score
from sklearn.svm import SVC
from logging import getLogger, StreamHandler, Formatter, FileHandler, DEBUG, INFO
import gc
import utils
from statistics import mean

SUBMISSION_DIR = '../result/'
KEY_COL = 'SK_ID_CURR'
TGT_COL = 'TARGET'

TRAIN = '../input/custom/201808280210_train_joined_debug.csv'
TEST = '../input/custom/201808280210_test_joined_debug.csv'

LOG_DIR = '../log/'
LOG_FILE = 'train_svm_es.py.log'

WORK_DIR = '../work/'

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

logger.info('=== start train_svm_es.py ===')

logger.info('--- load data ---')
logger.info('train file name:' + TRAIN)
train = pd.read_csv(TRAIN)
logger.info('train: {}'.format(train.shape))
logger.info('test file name:' + TEST)
test = pd.read_csv(TEST)
logger.info('test: {}'.format(test.shape))

logger.info('--- X-Y split ---')
train_x = train.drop(columns=[KEY_COL, TGT_COL])
features = train_x.columns
logger.info('train_x: {}'.format(train_x.shape))
train_y = train[TGT_COL]
logger.info('train_y: {}'.format(train_y.shape))
del train

keys = test[KEY_COL]
test = test.drop(columns=[KEY_COL])
logger.info('test_x: {}'.format(test.shape))

# logger.info('--- reduce memory usage ---')
# train_x = utils.reduce_mem_usage(train_x, logger)
# test = utils.reduce_mem_usage(test, logger)
gc.collect()

logger.info('--- grid search ---')
param_grid = {
        'kernel': ['rbf'],
        'C': [1.0],
        'gamma': [0.001],
        'probability': [True],
        'verbose': [True]
        'random_state': [42]
}

folds = 3
best_tst_auc = 0
best_param = {}
logger.info('start grid search')
for ngrid, params in enumerate(ParameterGrid(param_grid)):
    logger.info('  grid search #{}'.format(ngrid + 1))
    logger.info('  params: {}'.format(params))

    clf = SVC(**params)
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=42)

    list_trn_auc = []
    list_tst_auc = []

    logger.info('  start closs validation(total {} folds)'.format(folds))
    for nfold, (trn_idx, tst_idx) in enumerate(skf.split(train_x, train_y)):
        logger.info('    cross validation: {} / {} fold'.format(nfold + 1, folds))
        
        x_trn = train_x.iloc[trn_idx]
        y_trn = train_y.iloc[trn_idx]
        x_tst = train_x.iloc[tst_idx]
        y_tst = train_y.iloc[tst_idx]

        clf.fit(x_trn, y_trn)
        
        trn_auc = roc_auc_score(y_trn, clf.predict_proba(x_trn)[:, 1])
        list_trn_auc.append(trn_auc)
        logger.info('    trn_auc: {}'.format(trn_auc))
        
        tst_auc = roc_auc_score(y_tst, clf.predict_proba(x_tst)[:, 1])
        list_val_auc.append(tst_auc)
        logger.info('    tst_auc: {}'.format(tst_auc))
    
    mean_trn_auc = mean(list_trn_auc)
    logger.info('  mean_trn_auc: {}'.format(mean_trn_auc))
    
    mean_tst_auc = mean(list_tst_auc)
    logger.info('  mean_tst_auc: {}'.format(mean_tst_auc))
    
    if best_tst_auc < mean_tst_auc:
        best_tst_auc = mean_tst_auc
        best_params = params
    
    logger.info('  current best_tst_auc: {}'.format(best_tst_auc))
    logger.info('  current best_params: {}'.format(best_params))

logger.info('final best_tst_auc: {}'.format(best_tst_auc))
logger.info('final best_params: {}'.format(best_params))

logger.info('--- save submission file ---')
logger.info('fitting to all train data with best params...')
clf = SVClassifier(**best_params)
clf.fit(train_x, train_y)

logger.info('predicting test data...')
proba = clf.predict_proba(test)[:, 1]

logger.info('saving: ' + SUBMISSION_DIR + 'submission_svm.csv')
submission = pd.DataFrame({KEY_COL: keys, 'TARGET': proba})
submission.to_csv(SUBMISSION_DIR + 'submission_svm.csv', index=False)

logger.info('=== end train_svm_es.py ===')
