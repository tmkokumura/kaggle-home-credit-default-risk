import pandas as pd
from sklearn.preprocessing import LabelEncoder
from logging import getLogger, StreamHandler, Formatter, FileHandler, DEBUG, INFO

# define constant
IN_FILE_1 = '..\\input\\application_train.csv'
OUT_FILE_1 = '..\\output\\application_train_encode.csv'
COL_FILE_1 = '..\\output\\application_train_columns.csv'

IN_FILE_2 = '..\\input\\application_test.csv'
OUT_FILE_2 = '..\\output\\application_test_encode.csv'
COL_FILE_2 = '..\\output\\application_test_columns.csv'

TGT_COL = 'TARGET'

LOG_DIR = '../log/'
LOG_FILE = 'encode.py.log'

PROHIBITED_CHARS = [' ', '(', ')', ':', ',', '/', '-', '+']

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
df1 = pd.read_csv(IN_FILE_1)
df1_initial_rows = len(df1)
logger.info('df1: {}'.format(df1.shape))
df2 = None
if IN_FILE_2 is not None:
    df2 = pd.read_csv(IN_FILE_2)
    logger.info('df2: {}'.format(df2.shape))

    df1 = pd.concat([df1, df2], axis=0, sort=False)
    logger.info('merge df2 into df1: {}'.format(df1.shape))

logger.info('--- label encoding ---')
le = LabelEncoder()
for col in df1:
    if df1[col].dtype == 'object':
        if len(list(df1[col].unique())) <= 2:
            df1[col] = le.fit_transform(df1[col])
            logger.info('label encode :' + col)
logger.info('df1: {}'.format(df1.shape))

logger.info('--- one-hot encoding ---')
df1 = pd.get_dummies(df1, dummy_na=True)
logger.info('df1: {}'.format(df1.shape))

logger.info('--- replace prohibited character with under score ---')
replace_dict = {}
for col in df1:
    replaced_col = col
    for char in PROHIBITED_CHARS:
        replaced_col = replaced_col.replace(char, '_')
    replace_dict[col] = replaced_col
df1 = df1.rename(index=str, columns=replace_dict)
logger.info('df1: {}'.format(df1.shape))

if IN_FILE_2 is not None:
    logger.info('--- df1-df2 split ---')
    df2 = df1[df1_initial_rows:]
    df2 = df2.drop(columns=[TGT_COL])
    logger.info('df2: {}'.format(df2.shape))
    df1 = df1[:df1_initial_rows]
    logger.info('df1: {}'.format(df1.shape))


logger.info('--- save data ---')
df1.to_csv(OUT_FILE_1, index=False)
logger.info('saved: ' + OUT_FILE_1)
columns = pd.Series(df1.columns)
columns.to_csv(COL_FILE_1, index=False)
logger.info('saved: ' + COL_FILE_1)
if IN_FILE_2 is not None:
    df2.to_csv(OUT_FILE_2, index=False)
    logger.info('saved: ' + OUT_FILE_2)
    columns = pd.Series(df2.columns)
    columns.to_csv(COL_FILE_2, index=False)
    logger.info('saved: ' + COL_FILE_2)

logger.info('end')
