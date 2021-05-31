import logging
import os
import random
from argparse import ArgumentParser
from datetime import datetime

import numpy as np

import torch

import src.datasets.utils as du
import src.datasets.text_classification_dataset as tcds
from src.models.anml.ANML import ANML
from src.models.cls_baseline import Baseline
from src.models.cls_oml import OML

import socket

logging.basicConfig(level='INFO', format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('ContinualLearningLog')

if __name__ == '__main__':

    # Define the ordering of the datasets
    dataset_order_mapping = {
        1: [0, 3],
        2: [3, 0],
        # 3: [2, 0, 3, 1, 4],
        # 4: [3, 4, 0, 1, 2],
        # 5: [2, 4, 1, 3, 0],
        # 6: [0, 2, 1, 4, 3]
    }
    n_classes = 18

    # Parse command line arguments
    parser = ArgumentParser()
    parser.add_argument('--order', type=int, help='Order of datasets', required=True)
    parser.add_argument('--n_epochs', type=int, help='Number of epochs (only for MTL)', default=1)
    parser.add_argument('--lr', type=float, help='Learning rate (only for the baselines)', default=3e-5)
    parser.add_argument('--inner_lr', type=float, help='Inner-loop learning rate', default=0.001)
    parser.add_argument('--meta_lr', type=float, help='Meta learning rate', default=3e-5)
    parser.add_argument('--model', type=str, help='Name of the model', default='bert')
    parser.add_argument('--learner', type=str, help='Learner method', default='oml')
    parser.add_argument('--mini_batch_size', type=int, help='Batch size of data points within an episode', default=16)
    parser.add_argument('--updates', type=int, help='Number of inner-loop updates', default=5)
    parser.add_argument('--write_prob', type=float, help='Write probability for buffer memory', default=1.0)
    parser.add_argument('--max_length', type=int, help='Maximum sequence length for the input', default=448)
    parser.add_argument('--seed', type=int, help='Random seed', default=42)
    parser.add_argument('--replay_rate', type=float, help='Replay rate from memory', default=0.01)
    parser.add_argument('--replay_every', type=int, help='Number of data points between replay', default=9600)
    parser.add_argument('--out_layer', type=str, help='Type of layer used for prediction', default='linear')
    parser.add_argument('--modulation', type=str, help='type of modulation, (input, hebbian or double)',
                        default='input')
    parser.add_argument('--force_cpu', type=bool, help='Force CPU computation (False)', default=False)
    parser.add_argument('--log_file', type=str, help='log file', default=None)
    parser.add_argument('--max_train_size', type=int, help='',
                        default=tcds.MAX_TRAIN_SIZE)
    parser.add_argument('--max_test_size', type=int, help='',
                        default=tcds.MAX_TEST_SIZE)
    parser.add_argument('--model_filename', type=str, help='',
                        default=None)

    args = parser.parse_args()

    tcds.MAX_TRAIN_SIZE = args.max_train_size
    tcds.MAX_TEST_SIZE = args.max_test_size

    if args.model_filename is None:
        raise RuntimeError('Missing model_filename command line argument.')

    if args.log_file is None:
        tag = 'loader-' + args.learner + '-' + str(args.mini_batch_size) + '-' + str(args.order) \
              + '-out_layer_' + args.out_layer \
              + '-modulation_' + args.modulation \
              + '-train_size_' + str(tcds.MAX_TRAIN_SIZE) \
              + '-test_size_' + str(tcds.MAX_TEST_SIZE) \
              + '-' + socket.gethostname() + '-' \
              + '-' + str(datetime.now()).replace(':', '-').replace(' ', '_')
    else:
        tag = args.log_file

    setattr(args, 'tag', tag)
    file_name = 'logs/' + tag + '.log'
    os.makedirs('logs', exist_ok=True)
    fileh = logging.FileHandler(file_name, 'a')
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fileh.setFormatter(formatter)

    vargs = vars(args)

    log = logging.getLogger()  # root logger
    for hdlr in log.handlers[:]:  # remove all old handlers
        log.removeHandler(hdlr)
    log.addHandler(fileh)  # set the new handler

    logger.info('Using configuration: {}'.format(vars(args)))

    # Set base path
    base_path = os.path.dirname(os.path.abspath(__file__))

    # Set random seed
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    # Load the datasets
    logger.info('Loading the datasets')
    train_datasets, test_datasets = [], []
    for dataset_id in dataset_order_mapping[args.order]:
        train_dataset, test_dataset = du.get_dataset(base_path, dataset_id)
        logger.info('Loaded {}'.format(train_dataset.__class__.__name__))
        train_dataset = du.offset_labels(train_dataset)
        test_dataset = du.offset_labels(test_dataset)
        train_datasets.append(train_dataset)
        test_datasets.append(test_dataset)
    logger.info('Finished loading all the datasets')

    # Load the model
    use_cuda = torch.cuda.is_available() and not args.force_cpu
    device = torch.device('cuda' if use_cuda else 'cpu')
    logger.info('Compute using cuda:' + str(use_cuda))
    learner = ANML(device=device, n_classes=n_classes, **vars(args))

    logger.info('Using {} as learner'.format(learner.__class__.__name__))

    # Load trained model
    logger.info('----------Loading starts here----------')
    logger.info('Loading the model from {}'.format(args.model_filename))
    learner.load_model(args.model_filename)

    # Evaluation
    logger.info('----------Testing starts here----------')
    os.makedirs('trace', exist_ok=True)
    setattr(args, 'trace_file', 'trace/trace_' + tag)

    learner.visualise(test_datasets, **vars(args))
