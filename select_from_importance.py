import pandas as pd
from logging import getLogger, StreamHandler, Formatter, FileHandler, DEBUG, INFO


# define constant
IN_TRAIN_FILE = '../input/custom/201808231336_train.csv'
IN_TEST_FILE = '../input/custom/201808231336_test.csv'

IN_IMPORTANCE_FILE = '../work/feature_importance_20180823_1.csv'

OUT_TRAIN_FILE = '../input/custom/201808232103_train_selected.csv'
OUT_TEST_FILE = '../input/custom/201808232103_test_selected.csv'

OUT_MASK_FILE = '../work/mask.csv'

LOG_DIR = '../log/'
LOG_FILE = 'select_from_importance.py.log'

KEY_COL = 'SK_ID_CURR'
TGT_COL = 'TARGET'

RANK = 100

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

logger.info('start select_from_importance.py')

logger.info('--- load data ---')
logger.info('loading: ' + IN_TRAIN_FILE)
df_train = pd.read_csv(IN_TRAIN_FILE)
logger.info('df_train: {}'.format(df_train.shape))

logger.info('loading: ' + IN_TEST_FILE)
df_test = pd.read_csv(IN_TEST_FILE)
logger.info('df_test: {}'.format(df_test.shape))

logger.info('loading: ' + IN_IMPORTANCE_FILE)
df_importance = pd.read_csv(IN_IMPORTANCE_FILE)
logger.info('df_importance: {}'.format(df_importance.shape))

logger.info('--- drop unimportant feature ---')
# 1行目はヘッダのため除く
drop_features = df_importance.iloc[RANK:, 0].values.tolist()
df_train.drop(columns=drop_features, inplace=True)
logger.info('df_train: {}'.format(df_train.shape))
df_test.drop(columns=drop_features, inplace=True)
logger.info('df_test: {}'.format(df_test.shape))

logger.info('--- save data ---')
logger.info('saving: ' + OUT_TRAIN_FILE)
df_train.to_csv(OUT_TRAIN_FILE, index=False)
logger.info('saving: ' + OUT_TEST_FILE)
df_test.to_csv(OUT_TEST_FILE, index=False)


