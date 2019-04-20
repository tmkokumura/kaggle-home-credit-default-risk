import pandas as pd
from tqdm import tqdm
from logging import getLogger, StreamHandler, Formatter, FileHandler, DEBUG, INFO

SUBMISSION_DIR = '../result/'
SUBMISSION_FILES = [
        'submission_1.csv',
        'submission_2.csv',                                 
        'submission_3.csv',
        'submission_4.csv'
        ]

ENSEMBLED_FILE = 'submission_ensembled.csv'

KEY_COL = 'SK_ID_CURR'

LOG_DIR = '../log/'
LOG_FILE = 'ensemble.py.log'

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

logger.info('start ensemble.py')

logger.info('--- load each submission file ---')
file_num = len(SUBMISSION_FILES)
logger.info('loading: ' + SUBMISSION_FILES[0])
df_submission = pd.read_csv(SUBMISSION_DIR + SUBMISSION_FILES[0]).sort_values(by=KEY_COL)
logger.info(SUBMISSION_FILES[0] + ': {}'.format(df_submission.shape))
list_df_submission = [df_submission]
del SUBMISSION_FILES[0]
for submission_file in tqdm(SUBMISSION_FILES):
    logger.info('file: ' + submission_file)
    df_submission = pd.read_csv(SUBMISSION_DIR + submission_file).sort_values(by=KEY_COL)
    logger.info(submission_file + ': {}'.format(df_submission.shape))
    list_df_submission.append(df_submission)
logger.info('{} files loaded'.format(file_num))

rows = len(list_df_submission[0])

logger.info('--- calculate mean score ---')
df_submission_ensemble = list_df_submission[0].copy()
for i in tqdm(range(rows)):
    val = 0.0
    for df_submission in list_df_submission:
        val += df_submission.iat[i, 1]

    df_submission_ensemble.iat[i, 1] = val / file_num

logger.info('--- save ensembled submission file ---')
logger.info('saving: ' + SUBMISSION_DIR + ENSEMBLED_FILE)
df_submission_ensemble.to_csv(SUBMISSION_DIR + ENSEMBLED_FILE, index=False)
logger.info('end  ensemble.py')




