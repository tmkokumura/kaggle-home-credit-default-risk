import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
from logging import getLogger, StreamHandler, Formatter, FileHandler, DEBUG, INFO

IN_DIR = '../input/custom/'
IN_TRAIN_FILE = '201808231336_train_tmp.csv' 
IN_TEST_FILE = '201808231336_test.csv'

OUT_DIR = '../input/custom/'
OUT_TRAIN_FILE = '201808231336_train_tsne.csv'
OUT_TEST_FILE= '201808231336_test_tsne.csv'

KEY_COL = 'SK_ID_CURR'
TGT_COL = 'TARGET'


LOG_DIR = '../log/'
LOG_FILE = 'tsne.py.log'

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
tsne = TSNE(
        n_components=2, 
        perplexity=30.0, 
        early_exaggeration=12.0,
        learning_rate=200.0, 
        n_iter=1000, 
        n_iter_without_progress=300,
        min_grad_norm=1e-07,
        metric='euclidean',
        init='pca',
        verbose=3,
        random_state=0,
        method='barnes_hut',
        angle=0.5
        )
x_embedded = tsne.fit_transform(x)
logger.info('x_embedded: {}'.format(x_embedded.shape))

logger.info('--- x_train - x_test split ---')
df_train = x[:train_rows]
logger.info('df_train: {}'.format(df_train.shape))
df_test = x[train_rows:]
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

