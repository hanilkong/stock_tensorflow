import pandas as pd
import FinanceDataReader as fdr
from tqdm import tqdm
import time
from multiprocessing import Pool
import multiprocessing
import numpy as np

cpu_count = multiprocessing.cpu_count()
headers = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'Change', 'Code']


def merging_stock_data(code):
    merge_stock_list = []
    stock_list = fdr.DataReader(code, '2017').reset_index().values.tolist()
    for row in stock_list:                                # 불러온 주가 데이터를 1줄씩 불러옴
        row.append(code)                                  # 주가 데이터에 기업 코드를 추가
        merge_stock_list.append(row)                      # 모든 기업의 데이터를 병합

    return merge_stock_list


def make_code(x):
    x = str(x)
    return '0'*(6-len(x)) + x


def get_finance():
    code_data = pd.read_html(
        'http://kind.krx.co.kr/corpgeneral/corpList.do?method=download', header=0)[0]
    code_data['종목코드'] = code_data['종목코드'].apply(make_code)
    code_list = code_data[code_data['상장일'] < '2017-01-01']['종목코드']
    code_data.head()

    start_time = time.time()
    result = []
    #### 멀티 프로세싱 ####
    print('start multi process!!')
    p = Pool(cpu_count)  # 몇개의 코어를 이용할 것인지 설정
    for row in p.map(merging_stock_data, code_list):  # 각 코어에 입력값들을 병렬 처리
        result += row
    p.close()  # 멀티 프로세싱 종료
    p.join()
    result = np.array(result)
    df = pd.DataFrame(result)
    df.to_csv('./stock_data.csv', index=False, header=headers)
    end_time = time.time()
    print('--- 걸린시간: {} ---'.format(end_time - start_time))


start_time = time.time()
