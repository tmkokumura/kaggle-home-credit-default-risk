import pandas as pd
import numpy as np

TRAIN_DATA = '../input/application_train.csv'
TRAIN_DATA_DEBUG = '../input/application_train_mini.csv'
TEST_DATA = '../input/application_test.csv'

def _reduce_mem_usage(data, logger):
    start_mem = data.memory_usage().sum() / 1024**2
    logger.debug('memory usage of dataframe: {:.2f} MB'.format(start_mem))
    
    for col in data.columns:
        col_type = data[col].dtype
        
        if col_type != object:
            c_min = data[col].min()
            c_max = data[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    data[col] = data[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    data[col] = data[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    data[col] = data[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    data[col] = data[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    data[col] = data[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    data[col] = data[col].astype(np.float32)
                else:
                    data[col] = data[col].astype(np.float64)

    end_mem = data.memory_usage().sum() / 1024**2
    logger.debug('Memory usage after optimization: {:.2f} MB'.format(end_mem))
    logger.debug('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))
    
    return data

def _read_csv(path, logger):
    logger.info('enter')
    df = pd.read_csv(path)
    df = _reduce_mem_usage(df, logger)
    logger.info('exit')
    return df

def load_train_data(debug=False, logger):
    logger.info('enter')
    if debug:
        df = _read_csv(TRAIN_DATA_DEBUG, logger)
    else:
        df = _read_csv(TRAIN_DATA, logger)
    logger.info('exit')
    return df

def load_test_data(logger):
    logger.debug('enter')
    df = _read_csv(TEST_DATA, logger)
    logger.debug('exit')
    return df

if __name__ == '__main__':
    print(load_train_data().head())
    print(load_test_data().head())
