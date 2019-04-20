import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import roc_auc_score
from logging import getLogger, StreamHandler, Formatter, FileHandler, DEBUG, INFO
import lightgbm as lgbm
import gc
import utils

# Logger Settings.
fmt = '%(asctime)s %(levelname)s [%(module)s#%(funcName)s] :%(message)s'
logging.basicConfig(level=logging.DEBUG, format=fmt)
logger = logging.getLogger('Main')

# Constants
SUBMISSION_DIR = '../result/'
KEY_COL = 'SK_ID_CURR'
TGT_COL = 'TARGET'
TRAIN = '../input/custom/201808210821_2_train.csv'
TEST = '../input/custom/201808210821_2_test.csv'
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

	logger.info('--- reduce memory usage ---')
	# train_x = utils.reduce_mem_usage(train_x, logger)
	# test = utils.reduce_mem_usage(test, logger)
	gc.collect()

	logger.info('--- grid search ---')
	param_grid = {
	        'num_leaves': [32], 
	        'max_depth': [-1],
	        'learning_rate': [0.1],
	        'n_estimators': [200],
	        'subsample_for_bin':[500],
	        'min_child_weight': [1e-7],
	        'min_child_samples': [20],
	        'subsample' : [0.8],
	        'colsample_bytree':[0.8] ,
	        'reg_alpha': [1.0],
	        'reg_lambda': [100.0]}

	clf = lgbm.sklearn.LGBMClassifier()
	grid = GridSearchCV(clf, param_grid=param_grid, cv=2, scoring="roc_auc", return_train_score=True, verbose=3, n_jobs=-1)
	grid.fit(train_x, train_y)

	logger.info('grid search end')
	logger.info('max_auc: {}'.format(grid.best_score_))
	logger.info('max_params: {}'.format(grid.best_params_))
	logger.debug(pd.DataFrame(grid.cv_results_))

	logger.info('--- save submission file ---')
	proba = grid.predict_proba(test)[:, 1]
	submission = pd.DataFrame({
	                            KEY_COL : keys,
	                            'TARGET' : proba
	                        })
	logger.info('saving: ' + SUBMISSION_DIR + 'submission.csv')
	submission.to_csv(SUBMISSION_DIR + 'submission.csv', index=False)

	logger.info('--- save feature importance ---')
	feature_importance = pd.DataFrame({
	    'feature': features,
	    'importance': grid.best_estimator_.feature_importances_
	    })
	feature_importance.sort_values(by='importance', axis=0, ascending=False, inplace=True)
	logger.info('saving: ' + WORK_DIR + 'feature_importance.csv')
	feature_importance.to_csv(WORK_DIR + 'feature_importance.csv', index=False)
	logger.info('end')
