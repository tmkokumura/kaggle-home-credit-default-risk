import pandas as pd
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from logging import getLogger, StreamHandler, Formatter, FileHandler, DEBUG, INFO


# define constant
IN_TRAIN_FILE = '../input/custom/201808201329_train.csv'
IN_TEST_FILE = '../input/custom/201808201329_test.csv'

OUT_TRAIN_FILE = '../input/custom/201808201329_train_selected.csv'
OUT_TEST_FILE = '../input/custom/201808201329_test_selected.csv'

OUT_MASK_FILE = '../work/mask.csv'

LOG_DIR = '../log/'
LOG_FILE = 'select_from_model.py.log'

KEY_COL = 'SK_ID_CURR'
TGT_COL = 'TARGET'

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

logger.info('start')

logger.info('--- load data ---')
logger.info('loading: ' + IN_TRAIN_FILE)
train = pd.read_csv(IN_TRAIN_FILE)
logger.info('train: {}'.format(train.shape))

logger.info('loading: ' + IN_TEST_FILE)
test = pd.read_csv(IN_TEST_FILE)
logger.info('train: {}'.format(test.shape))

logger.info('--- fill NaN ---')
train = train.fillna(0)
logger.info('train: {}'.format(train.shape))
test = test.fillna(0)
logger.info('test: {}'.format(test.shape))

logger.info('--- X-Y split ---')
train_x = train.drop(columns=[KEY_COL, TGT_COL])
train_key = train[KEY_COL]
features = train_x.columns
logger.info('train_x: {}'.format(train_x.shape))
train_y = train[TGT_COL]
logger.info('train_y: {}'.format(train_y.shape))

test_key = test[KEY_COL]
test = test.drop(columns=[KEY_COL])
logger.info('test: {}'.format(test.shape))

selector = SelectFromModel(
    RandomForestClassifier(n_estimators=100, random_state=0, n_jobs=-1),
    threshold='median'
)

logger.info('--- select from model ---')
logger.info('fitting...')
selector.fit(train_x, train_y)
mask = selector.get_support()

df_mask = pd.DataFrame({
    'features': features,
    'mask': mask
    })
df_mask.to_csv(OUT_MASK_FILE, index=false)

logger.info('features: {}'.format(features))
logger.info('mask: {}'.format(mask))

logger.info('transforming...')
train = pd.DataFrame(selector.transform(train_x))
train[KEY_COL] = train_key
train[TGT_COL] = train_y
logger.info('train: {}'.format(train.shape))

test = pd.DataFrame(selector.transform(test))
test[KEY_COL] = test_key
logger.info('test: {}'.format(test.shape))

logger.info('--- save data ---')
logger.info('saving: ' + OUT_TRAIN_FILE)
train.to_csv(OUT_TRAIN_FILE, index=False)
logger.info('saving: ' + OUT_TEST_FILE)
test.to_csv(OUT_TEST_FILE, index=False)


