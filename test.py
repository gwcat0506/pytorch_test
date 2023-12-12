import argparse
import torch
from tqdm import tqdm
import data_loader.data_loaders as module_data
import model.loss as module_loss
import model.metric as module_metric
import model.model as module_arch
from parse_config import ConfigParser

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import random 
import os


def main(config):
    logger = config.get_logger('test')
    
    # 데이터 셋 로드
    df = pd.read_csv(config["data_loader"]['args']["data_dir"])
    scaled_df = module_data.apply_scaler(df)

    # train_valid_split
    y_train, y_test = module_data.split_data(scaled_df, config['data_loader']['args']['split_ratio'])
    print("Train | Test :", y_train.shape, y_test.shape)
    
    # valid 로드
    valid_loader = module_data.get_loader(y_test, config['data_loader']['args']['window'], config['data_loader']['args']['batch_size'], False)
    print("Length of dataloader:", len(valid_loader))
    
    # build model architecture
    model = config.init_obj('arch', module_arch)
    logger.info(model)

    # get function handles of loss and metrics
    loss_fn = getattr(module_loss, config['loss'])
    metric_fns = [getattr(module_metric, met) for met in config['metrics']]

    # 형식 맞춰 수정 -> config['resume']
    logger.info('Loading checkpoint: {} ...'.format(config['resume']))
    checkpoint = torch.load(config['resume']) # best model
    state_dict = checkpoint['state_dict']
    if config['n_gpu'] > 1:
        model = torch.nn.DataParallel(model)
    model.load_state_dict(state_dict)

    # prepare model for testing
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()

    total_loss = 0.0
    total_metrics = torch.zeros(len(metric_fns))

    with torch.no_grad():
        for i, (data, target) in enumerate(tqdm(valid_loader)):
            data, target = data.to(device), target.to(device)
            output = model(data)

            #
            # save sample images, or do something with output here
            #

            # computing loss, metrics on test set
            loss = loss_fn(output, target)
            batch_size = data.shape[0]
            total_loss += loss.item() * batch_size
            for i, metric in enumerate(metric_fns):
                # gpu에 할당된 텐서를 numpy로 배열하기 위해 cpu()를 붙여줌.
                total_metrics[i] += metric(output.cpu().detach(), target.cpu().detach()) * batch_size

    
    n_samples = len(valid_loader.sampler) # valid_loader로 수정
    log = {'loss': total_loss / n_samples}
    log.update({
        met.__name__: total_metrics[i].item() / n_samples for i, met in enumerate(metric_fns)
    })
    logger.info(log)


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='PyTorch Template')
    args.add_argument('-c', '--config', default=None, type=str,
                    help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str,
                    help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                    help='indices of GPUs to enable (default: all)')

    config = ConfigParser.from_args(args)
    main(config)
