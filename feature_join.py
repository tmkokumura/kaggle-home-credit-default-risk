from tqdm import tqdm
import pandas as pd
from logging import getLogger, StreamHandler, Formatter, FileHandler, DEBUG, INFO


IN_TRAIN_FILES = [
        '../input/custom/201808231336_train.csv',
        '../input/custom/201808231336_train_pca.csv',
        '../input/custom/201808231336_train_tsne.csv'
        ]

IN_TEST_FILES = [
        '../input/custom/201808231336_test.csv',
        '../input/custom/201808231336_test_pca.csv',
        '../input/custom/201808231336_test_tsne.csv'
        ]

OUT_TRAIN_FILE = '../input/custom/201808280210_train_joined.csv'
OUT_TEST_FILE= '../input/custom/201808280210_test_joined.csv'

KEY_COL = 'SK_ID_CURR'
TGT_COL = 'TARGET'

LOG_DIR = '../log/'
LOG_FILE = 'feature_join.py.log'

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


def load(list_file_name):
    """
    load files
    :param list_file_name: list of file name 
    :return: list of data frame
    """
    logger.info('loading: ' + list_file_name[0])
    list_df = [pd.read_csv(list_file_name[0]).sort_values(by=KEY_COL).reset_index(drop=True)]
    logger.info('df: {}'.format(list_df[0].shape))
    del list_file_name[0]
    for file_name in tqdm(list_file_name):
        logger.info('loading: ' + file_name)
        list_df.append(pd.read_csv(file_name).sort_values(by=KEY_COL).reset_index(drop=True))
        logger.info('df: {}'.format(list_df[-1].shape))
    return list_df

def join(list_df, drop_target=False):
    """
    join data frames
    :param list_df: list of data frames
    :return: joined data frame
    """
    df = list_df[0].copy()
    del list_df[0]
    for df_tmp in tqdm(list_df):
        if drop_target:
            df_tmp.drop(columns=[TGT_COL], inplace=True)
        df = df.join(df_tmp.set_index(KEY_COL), how='inner', on=KEY_COL)
    del list_df
    logger.info('df: {}'.format(df.shape))
    return df


logger.info('start feature_join.py')

logger.info('--- load train data ---')
list_df_train = load(IN_TRAIN_FILES)

logger.info('--- join train data ---')
df_train = join(list_df_train, drop_target=True)

logger.info('--- load test data ---')
list_df_test = load(IN_TEST_FILES)

logger.info('--- join test data ---')
df_test = join(list_df_test, drop_target=False)

logger.info('--- save data---')
logger.info('saving: ' + OUT_TRAIN_FILE)
df_train.to_csv(OUT_TRAIN_FILE, index=False)
logger.info('saving: ' + OUT_TEST_FILE)
df_test.to_csv(OUT_TEST_FILE, index=False)

logger.info('--- end feature_join.py ---')
