import os
import sys
import logging
import argparse
import json

import torch

import pandas_datareader.data as web


if __name__ == '__main__':

    #입력값 파싱 - mode, 이름, 주식코드, 강화학습법, 시작과 끝 날짜, 학습률, discount factor
    parser = argparse.ArgumentParser()
    parser.add_argument('mode:',choices=['train', 'test', 'update', 'predict'], default='train')
    parser.add_argument('name:',default='test')
    parser.add_argument('stock_code:',nargs='+')
    parser.add_argument('rl_method:', choices=['dqn', 'pg', 'ac', 'a2c', 'a3c', 'monkey'],default='dqn')
    parser.add_argument('start_date:',default=20200101)
    parser.add_argument('end_date:',default=20211231)
    parser.add_argument('lr:',type=float,default=0.0001)
    parser.add_argument('discount_factor:',type=float,default=0.7)
    args = parser.parse_args()


    # 강화학습을 위한 인자 설정
    output_name = f'{args.mode}_{args.name}_{args.rl_method}'
    learning = args.mode in ['train','update']
    reuse_models = args.mode in ['test', 'update', 'predict']
    value_network_name = f'{args.name}_{args.rl_method}_value.mdl'
    policy_network_name = f'{args.name}_{args.rl_method}_policy.mdl'
    start_epsilon = 1 if args.mode in ['train', 'update'] else 0
    num_epoches = 1000 if args.mode in ['train', 'update'] else 1
    num_steps = 5 if args.net in ['lstm', 'cnn'] else 1


    #출력값 경로지정 cwd-current working directory
    output_path = os.path.join(os.getcwd(), output_name)

    # 파라미터 기록
    params = json.dumps(vars(args))
    with open(os.path.join(output_path, 'params.json'), 'w') as f:
        f.write(params)

    #Logger 이용해서 로그 기록하기
    log_path = os.path.join(output_path, f'{output_name}.log')
    if os.path.exists(log_path):
        os.remove(log_path)
    logging.basicConfig(format='%(message)s')
    logger = logging.getLogger("main logger")
    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setLevel(logging.INFO)
    file_handler = logging.FileHandler(filename=log_path, encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    logger.addHandler(stream_handler)
    logger.addHandler(file_handler)
    logger.info(params)

    #강화학습 모듈 임포트. Reinforcementlearner,DQNLearner 등 임포트함


    #차트 데이터 받아오기(data_manager 이용)


    #강화학습 시작









