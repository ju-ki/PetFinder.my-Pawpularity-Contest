import datetime
import logging
import math
import os
import numpy as np
import pandas as pd
import inspect
import random
import torch
import yaml
import joblib
from time import time
from contextlib import contextmanager


def seed_everything(seed=0):
    random.seed(seed)
    os.environ["PYTHONHASHEDSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


# coding: UTF-8
CONFIG_FILE = '../configs/config.yaml'

with open(CONFIG_FILE) as file:
    yml = yaml.load(file)
RAW_DATA_DIR_NAME = yml['SETTING']['RAW_DATA_DIR_NAME']
SUB_DIR_NAME = yml['SETTING']['SUB_DIR_NAME']


class Util:
    @classmethod
    def dump(cls, value, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(value, path, compress=True)

    @classmethod
    def load(cls, path):
        return joblib.load(path)


class Logger:
    """save log"""

    def __init__(self, path, exp_name=None):
        self.general_logger = logging.getLogger(path)
        stream_handler = logging.StreamHandler()
        file_general_handler = logging.FileHandler(
            os.path.join(path, f'{exp_name}.log'))
        if len(self.general_logger.handlers) == 0:
            self.general_logger.addHandler(stream_handler)
            self.general_logger.addHandler(file_general_handler)
            self.general_logger.setLevel(logging.INFO)

    def info(self, message):
        # display time
        self.general_logger.info(
            '[{}] - {}'.format(self.now_string(), message))

    @staticmethod
    def now_string():
        return str(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))


class Submission:

    @classmethod
    def create_submission(cls, run_name, path, sub_y_column):
        logger = Logger(path)
        logger.info(f'{run_name} - start create submission')

        submission = pd.read_csv(RAW_DATA_DIR_NAME + 'sample_submission.csv')
        pred = Util.load_df_pickle(path + f'{run_name}-pred.pkl')
        submission[sub_y_column] = pred
        submission.to_csv(path + f'{run_name}_submission.csv', index=False)

        logger.info(f'{run_name} - end create submission')


class AbstractBaseBlock:
    def fit(self, input_df, y=None):
        return self.transform(input_df)

    def transform(self, input_df):
        raise NotImplementedError()


class WrapperBlock(AbstractBaseBlock):
    def __init__(self, function):
        self.function = function

    def transform(self, input_df):
        return self.function(input_df)


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def timeSince(since, percent):
    now = time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (remain %s)' % (asMinutes(s), asMinutes(rs))


def reduce_mem_usage(df):
    """ iterate through all the columns of a dataframe and modify the data type
        to reduce memory usage.        
    """
    start_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))

    for col in df.columns:
        col_type = df[col].dtype

        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
        else:
            df[col] = df[col].astype('category')

    end_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
    print('Decreased by {:.1f}%'.format(
        100 * (start_mem - end_mem) / start_mem))

    return df


def import_data(file):
    """create a dataframe and optimize its memory usage"""
    df = pd.read_csv(file, parse_dates=True, keep_date_col=True)
    df = reduce_mem_usage(df)
    return df


def param_to_name(params: dict, key_sep='_', key_value_sep='=') -> str:
    """
    dict を `key=value` で連結した string に変換します.
    Args:
        params:
        key_sep:
            key 同士を連結する際に使う文字列.
        key_value_sep:
            それぞれの key / value を連結するのに使う文字列.
            `"="` が指定されると例えば { 'foo': 10 } は `"foo=10"` に変換されます.
    Returns:
        文字列化した dict
    """
    sorted_params = sorted(params.items())
    return key_sep.join(map(lambda x: key_value_sep.join(map(str, x)), sorted_params))


def cachable(function):
    attr_name = '__cachefile__'

    def wrapper(*args, **kwrgs):
        force = kwrgs.pop('force', False)
        call_args = inspect.getcallargs(function, *args, **kwrgs)

        arg_name = param_to_name(call_args)
        name = attr_name + arg_name

        use_cache = hasattr(function, name) and not force

        if use_cache:
            cache_object = getattr(function, name)
        else:
            print('run')
            cache_object = function(*args, **kwrgs)
            setattr(function, name, cache_object)

        return cache_object

    return wrapper


@cachable
def read_csv(name, INPUT_PATH):

    if '.csv' not in name:
        name = name + '.csv'

    return pd.read_csv(os.path.join(INPUT_PATH, name))


@contextmanager
def timer(logger=None, format_str="{:.3f}[s]", prefix=None, suffix=None):

    if prefix:
        format_str = str(prefix) + format_str
    if suffix:
        format_str = format_str + str(suffix)
    start = time()
    yield
    d = time() - start
    out_str = format_str.format(d)
    if logger:
        logger.info(out_str)
    else:
        print(out_str)


class Timer:
    def __init__(self, logger=None, format_str='{:.3f}[s]', prefix=None, suffix=None, sep=' ', verbose=0):

        if prefix:
            format_str = str(prefix) + sep + format_str
        if suffix:
            format_str = format_str + sep + str(suffix)
        self.format_str = format_str
        self.logger = logger
        self.start = None
        self.end = None
        self.verbose = verbose

    @property
    def duration(self):
        if self.end is None:
            return 0
        return self.end - self.start

    def __enter__(self):
        self.start = time()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end = time()
        if self.verbose is None:
            return
        out_str = self.format_str.format(self.duration)
        if self.logger:
            self.logger.info(out_str)
        else:
            print(out_str)


def create_block(input_df, blocks, y=None, path=None, task="train"):
    """[summary]

    Args:
        input_df (pd.DataFrame): [description]
        blocks (list): [description]
        y ([type], optional): [description]. Defaults to None.
        path ([type], optional): describe path include pkl feature. Defaults to None.
        task (str, optional): if create test dataframe, describe test. Defaults to "train".

    Returns:
        [type]: [description]
    """

    out_df = pd.DataFrame()
    df_lst = []
    print("**" * 20 + f"start create block for {task}" + "**" * 20)
    with Timer(prefix="create test={}".format(task)):
        for block in blocks:
            if "WrapperBlock" in str(block.__class__.__name__).split():
                file_name = os.path.join(
                    path, f"{task}_{str(block.function.__name__)}.pkl")
            elif "LabelEncodingBlock" in str(block.__class__.__name__).split():
                file_name = os.path.join(
                    path, f"{task}_{block.__class__.__name__}_{str(block.cols)}.pkl")
            elif "CountEncodingBlock" in str(block.__class__.__name__).split():
                file_name = os.path.join(
                    path, f"{task}_{block.__class__.__name__}_{str(block.column)}.pkl")
            elif "TargetEncodingBlock" in str(block.__class__.__name__).split():
                file_name = os.path.join(
                    path, f"{task}_{block.__class__.__name__}_{str(block.cols)}.pkl")
            elif "AggregationBlock" in str(block.__class__.__name__).split():
                file_name = os.path.join(
                    path, f"{task}_{block.__class__.__name__}_{str(block.group_key)}.pkl")
            else:
                file_name = os.path.join(
                    path, f"{task}_{block.__class__.__name__}.pkl")
            with Timer(prefix="\t- {}".format(str(block))):
                if os.path.isfile(file_name):
                    print(f"Already is created {block.__class__.__name__}")
                    out_i = Util.load(file_name)
                else:
                    if task == "train":
                        out_i = block.fit(input_df)
                        Util.dump(out_i, file_name)
                    elif task == "test":
                        out_i = block.transform(input_df)
                        Util.dump(out_i, file_name)
            df_lst.append(out_i)
            out_df = pd.concat(df_lst, axis=1)
    return out_df
