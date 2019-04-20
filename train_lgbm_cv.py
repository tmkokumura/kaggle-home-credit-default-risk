import pandas as pd
from sklearn.model_selection import train_test_split
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

	logger.info('--- cross validation ---')
	x_trn, x_val, y_trn, y_val = train_test_split(train_x, train_y, test_size=0.25, random_state=42)

	params = {
	        'num_leaves': 32,
	        'max_depth': -1,
	        'learning_rate': 0.1,
	        'n_estimators': 10000,
	        'subsample_for_bin': 500,
	        'min_child_weight': 1e-7,
	        'min_child_samples': 20,
	        'subsample': 0.8,
	        'colsample_bytree': 0.8,
	        'reg_alpha': 1.0,
	        'reg_lambda': 100.0,
	        'random_state': 42
	}
	clf = lgbm.sklearn.LGBMClassifier(**params)
	clf.fit(x_trn, y_trn, eval_set=[(x_val, y_val)], eval_metric='roc_auc', early_stopping_rounds=100, verbose=1)
	logger.info('best iteration: {}'.format(clf.best_iteration_))
	trn_auc = roc_auc_score(y_trn, clf.predict_proba(x_trn, num_iteration=clf.best_iteration_)[:, 1])
	logger.info('trn_auc: {}'.format(trn_auc))
	val_auc = roc_auc_score(y_val, clf.predict_proba(x_val, num_iteration=clf.best_iteration_)[:, 1])
	logger.info('val_auc: {}'.format(trn_auc))

	logger.info('--- save submission file ---')
	params['n_estimators'] = clf.best_iteration_
	clf = lgbm.sklearn.LGBMClassifier(**params)
	clf.fit(train_x, train_y, verbose=1)
	proba = clf.predict_proba(test)[:, 1]
	submission = pd.DataFrame({
	                            KEY_COL: keys,
	                            'TARGET': proba
	                        })
	logger.info('saving: ' + SUBMISSION_DIR + 'submission.csv')
	submission.to_csv(SUBMISSION_DIR + 'submission.csv', index=False)

	logger.info('--- save feature importance ---')
	feature_importance = pd.DataFrame({
	    'feature': features,
	    'importance': clf.feature_importances_
	    })
	feature_importance.sort_values(by='importance', axis=0, ascending=False, inplace=True)
	logger.info('saving: ' + WORK_DIR + 'feature_importance.csv')
	feature_importance.to_csv(WORK_DIR + 'feature_importance.csv', index=False)
	logger.info('end')
