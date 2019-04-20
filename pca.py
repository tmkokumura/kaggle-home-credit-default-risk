import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from logging import getLogger, StreamHandler, Formatter, FileHandler, DEBUG, INFO

IN_DIR = '../input/custom/'
IN_TRAIN_FILE = '201808231336_train_tmp.csv' 
IN_TEST_FILE = '201808231336_test.csv'

OUT_DIR = '../input/custom/'
OUT_TRAIN_FILE = '201808231336_train_pca.csv'
OUT_TEST_FILE= '201808231336_test_pca.csv'

KEY_COL = 'SK_ID_CURR'
TGT_COL = 'TARGET'


LOG_DIR = '../log/'
LOG_FILE = 'pca.py.log'

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

logger.info('start tsne.py')

logger.info('--- load data ---')
logger.info('loading: ' + IN_DIR + IN_TRAIN_FILE)
df_train = pd.read_csv(IN_DIR + IN_TRAIN_FILE)
train_rows = len(df_train)
logger.info('df_train: {}'.format(df_train.shape))

logger.info('loading: ' + IN_DIR + IN_TEST_FILE)
df_test = pd.read_csv(IN_DIR + IN_TEST_FILE)
logger.info('df_test: {}'.format(df_test.shape))

logger.info('--- X-Y split ---')
key_train = df_train[KEY_COL]
x_train = df_train.drop(columns=[KEY_COL, TGT_COL])
logger.info('x_train: {}'.format(x_train.shape))
y_train = df_train[TGT_COL]
logger.info('y_train: {}'.format(y_train.shape))
key_test = df_test[KEY_COL]
x_test = df_test.drop(columns=[KEY_COL])
logger.info('x_test: {}'.format(x_test.shape))

logger.info('--- x_train - x_test concatenate ---')
x = pd.concat([x_train, x_test], axis=0, sort=False)
logger.info('x: {}'.format(x.shape))

logger.info('--- t-SNE ---')
logger.info('fitting...')
pca = PCA(
        n_components=100, 
        copy=True,
        whiten=False,
        svd_solver='auto',
        tol=0.0,
        iterated_power='auto',
        random_state=0,
        )
x_embedded = pca.fit_transform(x)
logger.info('x_embedded: {}'.format(x_embedded.shape))

logger.info('--- x_train - x_test split ---')
df_train = x_embedded[:train_rows]
logger.info('df_train: {}'.format(df_train.shape))
df_test = x_embedded[train_rows:]
logger.info('df_test: {}'.format(df_test.shape))

logger.info('--- X-Y concatenate ---')
df_train[KEY_COL] = key_train
logger.info('df_train: {}'.format(df_train.shape))
df_test[KEY_COL] = key_test
logger.info('df_test: {}'.format(df_test.shape))

logger.info('--- save data ---')
logger.info('saving: ' + OUT_DIR + OUT_TRAIN_FILE)
df_train.to_csv(OUT_DIR + OUT_TRAIN_FILE, index=False)
logger.info('saving: ' + OUT_DIR + OUT_TEST_FILE)
df_test.to_csv(OUT_DIR + OUT_TEST_FILE, index=False)

logger.info('end tsne.py')

