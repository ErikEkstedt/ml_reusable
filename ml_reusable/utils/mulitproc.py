from multiprocessing import Pool
from tqdm import tqdm
import time


def some_function(x):
    for i in range(1000):
        x = i % 3
    return x


def run_in_parallel(func, iterable, desc):
    ''' pool.map is faster but wont work with tqdm 
    Arguments:
        func:           function to process iterable with
        iterable:       iterable of data
        desc:           str, description for tqdm
    Returns:
        out:            list of result
    '''
    with Pool() as pool:
        out = list(tqdm(pool.imap(func, iterable),
            total=len(iterable),
            desc=desc,
            dynamic_ncols=True
            ))
    return out
