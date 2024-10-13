import argparse
import os
import logging
import sys
import json
from model import DQNLearner

from data_mger import load_data  # Assuming you have a data_manager module
from learner import DQNLearner  # Assuming you have a learner module

# Default configurations
RL_METHOD = 'dqn'
NETWORK = 'lstm'
BACKEND = 'pytorch'
BALANCE = 100000000
NUM_STEPS = 5
MIN_TRADING_PRICE = 1000000
MAX_TRADING_PRICE = 100000000


def setup_logger(output_path, output_name):
    """Configures the logging settings."""
    log_path = os.path.join(output_path, f'{output_name}.log')
    if os.path.exists(log_path):
        os.remove(log_path)

    logging.basicConfig(format='%(message)s')
    logger = logging.getLogger('ldtrader')
    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setLevel(logging.INFO)
    file_handler = logging.FileHandler(filename=log_path, encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)

    logger.addHandler(stream_handler)
    logger.addHandler(file_handler)

    return logger


def parse_arguments():
    """Parses command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['train', 'test', 'update', 'predict'], default='train')
    parser.add_argument('--stock_code', required=True)
    parser.add_argument('--start', default='20200101')
    parser.add_argument('--end', default='20201231')
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--discount_factor', type=float, default=0.7)
    return parser.parse_args()


def setup_directories(output_path):
    """Creates necessary directories if they don't exist."""
    if not os.path.isdir(output_path):
        os.makedirs(output_path)


def save_parameters(args, output_path):
    """Saves the parameters to a JSON file."""
    params = json.dumps(vars(args), indent=4)
    with open(os.path.join(output_path, 'params.json'), 'w') as f:
        f.write(params)


def load_and_validate_data():
    """Loads data and ensures it's suitable for training/testing."""
    chart_data, training_data = load_data("coca_cola_stock_with_moving_averages.csv")
    assert len(chart_data) >= NUM_STEPS, "Insufficient data length for training."
    return chart_data, training_data


def main():
    args = parse_arguments()

    # Setting up paths and names
    output_name = f'{args.stock_code}_{args.mode}_{RL_METHOD}_{NETWORK}'
    output_path = './output'
    setup_directories(output_path)

    # Logging configuration
    logger = setup_logger(output_path, output_name)
    logger.info("Starting with parameters:")
    logger.info(json.dumps(vars(args), indent=4))

    # Save parameters
    save_parameters(args, output_path)

    # Load data
    chart_data, training_data = load_and_validate_data()

    # Define learner parameters
    learner_params = {
        'rl_method': RL_METHOD,
        'network': NETWORK,
        'num_steps': NUM_STEPS,
        'learning_rate': args.learning_rate,
        'balance': BALANCE,
        'num_epoches': 100 if args.mode in ['train', 'update'] else 1,
        'discount_factor': args.discount_factor,
        'start_epsilon': 1 if args.mode in ['train', 'update'] else 0,
        'reuse_model': args.mode in ['test', 'update', 'predict'],
        'output_path': output_path,
        'stock_code': args.stock_code,
        'chart_data': chart_data,
        'training_data': training_data,
        'min_trading_price': MIN_TRADING_PRICE,
        'max_trading_price': MAX_TRADING_PRICE,
        'value_network_path': './models',
    }

    # Initialize learner
    learner = DQNLearner(**learner_params)

    # Run learner
    if args.mode in ['train', 'test', 'update']:
        learner.run(learning=args.mode in ['train', 'update'])
        if args.mode in ['train', 'update']:
            learner.save_models()
    elif args.mode == 'predict':
        learner.predict()


if __name__ == '__main__':
    main()