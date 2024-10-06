import argparse
import os
import logging
import sys
import json

rl_method = 'dqn'
network = 'lstm'
backend = 'pytorch'

if __name__ == '__main__':
    #input Command Line Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['train', 'test', 'update', 'predict'], default='train')
    parser.add_argument('--stock_code')
    parser.add_argument('--start', default='20200101')
    parser.add_argument('--end', default='20201231')
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--discount_factor', type=float, default=0.7)
    parser.add_argument('--balance', type=int, default=100000000)
    args = parser.parse_args()

    #parameter setting
    output_name = f'{args.stock_code}_{args.mode}_{rl_method}_{network}'
    is_learning = args.mode in ['train', 'update']
    reuse_model = args.mode in ['test', 'update', 'predict']

    value_network_name = f'{args.stock_code}_{rl_method}_{network}_value.mdl'
    policy_network_name = f'{args.stock_code}_{rl_method}_{network}_policy.mdl'

    start_epsilon = 1 if args.mode in ['train', 'update'] else 0
    num_epoches = 100 if args.mode in ['train', 'update'] else 1
    num_steps = 5

    #output route
    output_path = './output'
    if not os.path.isdir(output_path):
      os.makedirs(output_path)

    #model route
    value_network_path = './models'
    policy_network_path = './models'

    #save params
    params = json.dumps(vars(args))
    with open(output_path + '/params.json', 'w') as f:
      f.write(params)

    #logging setting
    log_path = os.path.join(output_path, f'{output_name}.log')
    if os.path.exists(log_path):
        os.remove(log_path)
    logging.basicConfig(format='%(message)s')
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setLevel(logging.INFO)
    file_handler = logging.FileHandler(filename=log_path, encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    logger.addHandler(stream_handler)
    logger.addHandler(file_handler)
    logger.info(params)



    common_params = {}
    list_stock_code = []
    list_chart_data = []
    list_training_data = []
    list_min_trading_price = []
    list_max_trading_price = []