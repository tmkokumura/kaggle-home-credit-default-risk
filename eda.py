import pandas as pd
import numpy as np
from logging import getLogger, Formatter, StreamHandler, FileHandler, INFO, DEBUG

INPUT_DIR = '../input/'
CSV_EXT = '.csv'
LOG_DIR = '../result/'
LOG_FILE = 'eda.py.log'

APP_TRAIN = 'application_train'
APP_TEST = 'application_test'
BUREAU = 'bureau'
BUREAU_BAL = 'bureau_balance'
POS_CASH_BAL = 'POS_CASH_balance'
PREV_APP = 'previous_application'
INST_PAY = 'installments_payments'
CREDIT_CARD_BAL = 'credit_card_balance'

PK_COL = 'SK_ID_CURR'
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

# load data
app_train = pd.read_csv(INPUT_DIR + APP_TRAIN + CSV_EXT)
app_test = pd.read_csv(INPUT_DIR + APP_TEST + CSV_EXT)
app_train_test = pd.concat([app_train, app_test])
bureau = pd.read_csv(INPUT_DIR + BUREAU + CSV_EXT)
# bureau_bal = pd.read_csv(INPUT_DIR + BUREAU_BAL + CSV_EXT)
# pos_cash_bal = pd.read_csv(INPUT_DIR + POS_CASH_BAL + CSV_EXT)
# prev_app = pd.read_csv(INPUT_DIR + PREV_APP + CSV_EXT)
# inst_pay = pd.read_csv(INPUT_DIR + INST_PAY + CSV_EXT)
# credit_card_bal = pd.read_csv(INPUT_DIR + INST_PAY + CSV_EXT)

# check data size
logger.info('--- data size ---')
logger.info(APP_TRAIN + ': {}'.format(app_train.shape))
logger.info(APP_TEST + ': {}'.format(app_test.shape))
logger.info(BUREAU + ': {}'.format(bureau.shape))
# logger.info(BUREAU_BAL + ': {}'.format(bureau_bal.shape))
# logger.info(POS_CASH_BAL + ': {}'.format(pos_cash_bal.shape))
# logger.info(PREV_APP + ': {}'.format(prev_app.shape))
# logger.info(INST_PAY + ': {}'.format(inst_pay.shape))
# logger.info(CREDIT_CARD_BAL + ': {}'.format(credit_card_bal.shape))

# train test
logger.info('--- train & test---')
unq_app_train_id = app_train[PK_COL].unique()
unq_app_test_id = app_test[PK_COL].unique()
unq_app_train_test_id = app_train_test[PK_COL].unique()
logger.info(APP_TRAIN + ': {}'.format(len(unq_app_train_id)))
logger.info(APP_TEST + ': {}'.format(len(unq_app_test_id)))
logger.info('app_train_test: {}'.format(len(unq_app_train_test_id)))

categorical_cols = [col for col in app_train_test.columns if not pd.api.types.is_numeric_dtype(app_train_test[col].dtype)]
logger.info('categorical column: {}'.format(categorical_cols))
categorical_col_num = app_train_test[categorical_cols].nunique(dropna=False)
logger.info('categorical column num: {}'.format(categorical_col_num))

categorical_col_num.rename(columns={0:'col_name', 1:'type_num'})
flag_cols = categorical_col_num[categorical_col_num['type_num']==2]
logger.info('flag column: {}'.format(flag_cols))

# burea  TODO
logger.info('--- bureau ---')
unq_bureau_id = bureau[PK_COL].unique()
logger.info(BUREAU + ': {}'.format(len(unq_bureau_id)))
train_bureau = app_train.join(bureau, on=PK_COL, how='left', lsuffix='_train', rsuffix='_bureau')[[PK_COL + '_train', TGT_COL, 'SK_ID_BUREAU', 'CREDIT_ACTIVE']]
train_bureau_unique = train_bureau['CREDIT_ACTIVE'].unique()
logger.info(train_bureau_unique)
train_bureau_active = train_bureau[train_bureau['CREDIT_ACTIVE'] == 'Active']
train_bureau_closed = train_bureau[train_bureau['CREDIT_ACTIVE'] == 'Closed']
train_bureau_sold = train_bureau[train_bureau['CREDIT_ACTIVE'] == 'Sold']
train_bureau_baddebt= train_bureau[train_bureau['CREDIT_ACTIVE'] == 'Bad debt']
logger.info('train + bureau: {}'.format(train_bureau.shape))
logger.info('train + bureau_active: {}'.format(train_bureau_active.shape))
logger.info('train + bureau_closed: {}'.format(train_bureau_closed.shape))
logger.info('train + bureau_sold: {}'.format(train_bureau_sold.shape))
logger.info('train + bureau_baddebt: {}'.format(train_bureau_baddebt.shape))

train_pos_bureau_active = train_bureau_active[train_bureau_active[TGT_COL] == 1]
train_neg_bureau_active = train_bureau_active[train_bureau_active[TGT_COL] == 0]
logger.info('train_pos + bureau_active: {} ({:.2f}%)'.format(len(train_pos_bureau_active), len(train_pos_bureau_active)/len(train_bureau_active)*100))
logger.info('train_neg + bureau_active: {} ({:.2f}%)'.format(len(train_neg_bureau_active), len(train_neg_bureau_active)/len(train_bureau_active)*100))
train_pos_bureau_closed = train_bureau_closed[train_bureau_closed[TGT_COL] == 1]
train_neg_bureau_closed = train_bureau_closed[train_bureau_closed[TGT_COL] == 0]
logger.info('train_pos + bureau_closed: {} ({:.2f}%)'.format(len(train_pos_bureau_closed), len(train_pos_bureau_closed)/len(train_bureau_closed)*100))
logger.info('train_neg + bureau_closed: {} ({:.2f}%)'.format(len(train_neg_bureau_closed), len(train_neg_bureau_closed)/len(train_bureau_closed)*100))
train_pos_bureau_sold = train_bureau_sold[train_bureau_sold[TGT_COL] == 1]
train_neg_bureau_sold = train_bureau_sold[train_bureau_sold[TGT_COL] == 0]
logger.info('train_pos + bureau_sold: {} ({:.2f}%)'.format(len(train_pos_bureau_sold), len(train_pos_bureau_sold)/len(train_bureau_sold)*100))
logger.info('train_neg + bureau_sold: {} ({:.2f}%)'.format(len(train_neg_bureau_sold), len(train_neg_bureau_sold)/len(train_bureau_sold)*100))
train_pos_bureau_baddebt = train_bureau_baddebt[train_bureau_baddebt[TGT_COL] == 1]
train_neg_bureau_baddebt = train_bureau_baddebt[train_bureau_baddebt[TGT_COL] == 0]
logger.info('train_pos + bureau_baddebt: {} ({:.2f}%)'.format(len(train_pos_bureau_baddebt), len(train_pos_bureau_baddebt)/len(train_bureau_baddebt)*100))
logger.info('train_neg + bureau_baddebt: {} ({:.2f}%)'.format(len(train_neg_bureau_baddebt), len(train_neg_bureau_baddebt)/len(train_bureau_baddebt)*100))

