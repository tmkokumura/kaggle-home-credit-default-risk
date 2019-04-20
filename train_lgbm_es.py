import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, ParameterGrid
from sklearn.metrics import roc_auc_score
from logging import getLogger, StreamHandler, Formatter, FileHandler, DEBUG, INFO
import lightgbm as lgbm
import gc
import utils
from statistics import mean

# Logger Settings.
fmt = '%(asctime)s %(levelname)s [%(module)s#%(funcName)s] :%(message)s'
logging.basicConfig(level=logging.DEBUG, format=fmt)
logger = logging.getLogger('Main')

SUBMISSION_DIR = '../result/'
KEY_COL = 'SK_ID_CURR'
TGT_COL = 'TARGET'
TRAIN = '../input/custom/201808280210_train_joined.csv'
TEST = '../input/custom/201808280210_test_joined.csv'
WORK_DIR = '../work/'

if __name__ == '__main__':
	logger.info('start')

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
	        'num_leaves': [8],
	        'max_depth': [-1],
	        'learning_rate': [0.1],
	        'n_estimators': [10000],
	        'subsample_for_bin': [500],
	        'min_child_weight': [1e-7],
	        'min_child_samples': [20],
	        'subsample': [0.8],
	        'colsample_bytree': [0.8],
	        'reg_alpha': [1.0],
	        'reg_lambda': [150.0, 200.0, 250.0],
	        'random_state': [42]
	}

	folds = 3
	best_val_auc = 0
	best_param = {}
	logger.info('start grid search')
	for ngrid, params in enumerate(ParameterGrid(param_grid)):
	    logger.info('  grid search #{}'.format(ngrid + 1))
	    logger.info('  params: {}'.format(params))

	    clf = lgbm.sklearn.LGBMClassifier(**params)
	    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=42)

	    list_best_iteration = []
	    list_trn_auc = []
	    list_val_auc = []

	    logger.info('  start closs validation(total {} folds)'.format(folds))
	    for nfold, (trn_idx, val_idx) in enumerate(skf.split(train_x, train_y)):
	        logger.info('    cross validation: {} / {} fold'.format(nfold + 1, folds))
	        
	        x_trn = train_x.iloc[trn_idx]
	        y_trn = train_y.iloc[trn_idx]
	        x_val = train_x.iloc[val_idx]
	        y_val = train_y.iloc[val_idx]

	        clf.fit(x_trn, y_trn, eval_set=[(x_val, y_val)], eval_metric='auc', early_stopping_rounds=100, verbose=1)
	        
	        list_best_iteration.append(clf.best_iteration_)
	        logger.info('    best iteration: {}'.format(clf.best_iteration_))
	        
	        trn_auc = roc_auc_score(y_trn, clf.predict_proba(x_trn, num_iteration=clf.best_iteration_)[:, 1])
	        list_trn_auc.append(trn_auc)
	        logger.info('    trn_auc: {}'.format(trn_auc))
	        
	        val_auc = roc_auc_score(y_val, clf.predict_proba(x_val, num_iteration=clf.best_iteration_)[:, 1])
	        list_val_auc.append(val_auc)
	        logger.info('    val_auc: {}'.format(val_auc))
	    
	    mean_best_iteration = round(mean(list_best_iteration))
	    logger.info('  mean_best_iteration: {}'.format(mean_best_iteration))

	    mean_trn_auc = mean(list_trn_auc)
	    logger.info('  mean_trn_auc: {}'.format(mean_trn_auc))
	    
	    mean_val_auc = mean(list_val_auc)
	    logger.info('  mean_val_auc: {}'.format(mean_val_auc))
	    
	    if best_val_auc < mean_val_auc:
	        best_val_auc = mean_val_auc
	        params['n_estimators'] = mean_best_iteration + 1
	        best_params = params
	    
	    logger.info('  current best_val_auc: {}'.format(best_val_auc))
	    logger.info('  current best_params: {}'.format(best_params))

	logger.info('final best_val_auc: {}'.format(best_val_auc))
	logger.info('final best_params: {}'.format(best_params))

	logger.info('--- save submission file ---')
	logger.info('fitting to all train data with best params...')
	clf = lgbm.sklearn.LGBMClassifier(**best_params)
	clf.fit(train_x, train_y, verbose=1)

	logger.info('predicting test data...')
	proba = clf.predict_proba(test)[:, 1]

	logger.info('saving: ' + SUBMISSION_DIR + 'submission_lgbm.csv')
	submission = pd.DataFrame({KEY_COL: keys, 'TARGET': proba})
	submission.to_csv(SUBMISSION_DIR + 'submission_lgbm.csv', index=False)

	logger.info('--- save feature importance ---')
	logger.info('saving: ' + WORK_DIR + 'feature_importance_lgbm.csv')
	feature_importance = pd.DataFrame({'feature': features,'importance': clf.feature_importances_})
	feature_importance.sort_values(by='importance', axis=0, ascending=False, inplace=True)
	feature_importance.to_csv(WORK_DIR + 'feature_importance_lgbm.csv', index=False)

	logger.info('=== end train_xgb_es.py ===')
