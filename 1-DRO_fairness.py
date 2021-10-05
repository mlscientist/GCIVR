# coding=utf-8
"""Fairness with noisy protected groups experiments."""
import argparse
import os
import yaml
from pathlib import Path

import numpy as np
from sklearn.cluster import KMeans
import torch
import torch.nn as nn
from copy import deepcopy
from torch.utils.tensorboard import SummaryWriter

from utils import generate_proxy_groups_uniform, generate_proxy_groups_noise_array,\
                  generate_proxy_groups_single_noise, error_rate, group_error_rates, \
                  tpr, group_tprs,increment_dir
from data import load_data_adult
from model import Linear


def violation(
    labels, predictions, epsilon, groups):
  # Returns violations across different group feature thresholds.
  viol_list = []
  overall_tpr = tpr(labels, predictions).cpu()
  for kk in range(groups.shape[1]):
    group_tpr =group_tprs(labels, predictions, groups[:,kk]).cpu()
    viol_list += [overall_tpr - group_tpr - epsilon]
  return np.max(viol_list), viol_list

def lagrangian_loss(labels, predictions, epsilon, 
                    groups, criterion, device, gamma=0.1, alpha=0.1):
  # Returns the lagrangain loss of the constrained problem
  total_loss = torch.zeros(1).to(device)
  main_loss = criterion(predictions, labels)
  m = groups.shape[1]
  for kk in range(m):
    g_inds = groups[:,kk].logical_and((labels >= 1.0).squeeze())
    group_loss = criterion(predictions[g_inds],labels[g_inds])
    const_loss = group_loss - main_loss - epsilon
    total_loss += 1/(m+1) * (1/m + torch.exp(const_loss * alpha / gamma))
  return total_loss


def train_unconstrained(args, device=None):
  """Traqining model"""
  tb_writer = SummaryWriter(log_dir=args.log_dir)
  if device is None:
    device = torch.device('cpu')

  criterion = nn.BCEWithLogitsLoss()
  train_data, test_data = load_data_adult(args.noise_level, 
                                          uniform_groups=args.uniform_groups, 
                                          min_group_frac=args.min_group_frac, 
                                          use_noise_array=args.use_noise_array,
                                          group_features_type=args.group_features_type, 
                                          num_group_clusters=args.num_group_clusters)

  train_data.to(device)
  test_data.to(device)

  model = Linear(train_data.num_features)

  optim =  torch.optim.Adagrad(model.parameters(), lr=args.learning_rate)
 

  for e in range(args.epochs):
    # Check for the beginning of a new epoch.
    if args.resample_proxy_groups:

      # Only resample proxy groups every epochs_per_resample epochs.
      if e % args.epochs_per_resample == 0:
        # Resample the group at the beginning of the epoch.
        # Get groups_train from a ball around init_proxy_groups_train.
        if args.uniform_groups:
          train_data.proxy_groups_tensor = generate_proxy_groups_uniform(
              len(train_data), min_group_frac=args.min_group_frac).long().to(device)
        elif args.use_noise_array:
          train_data.noise_array = train_data.get_noise_array()
          train_data.proxy_groups_tensor = generate_proxy_groups_noise_array(
              train_data.proxy_groups_tensor, noise_array=train_data.noise_array).long().to(device)
        else:
          train_data.proxy_groups_tensor = generate_proxy_groups_single_noise(
              train_data.proxy_groups_tensor, noise_param=args.noise_level).long().to(device)
    model.train()
    optim.zero_grad()
    y_pred = model(train_data.data)
    # loss = criterion(y_pred, (train_data.target-.5)*2)
    main_loss = criterion(y_pred, train_data.targets)
    # model_last.load_state_dict(model.state_dict().copy())
    main_loss.backward()
    optim.step()

    # Snapshot iterate once in 1000 loops.
    if e % args.log == 0:
      model.eval()
      with torch.no_grad():
        y_pred_t = model(train_data.data)
        err = error_rate(train_data.targets, y_pred_t)
        # max_viol, viol_list = violation(
        #     train_data.targets, y_pred_t, args.epsilon, train_data.proxy_groups_tensor)
        max_viol, viol_list = violation(
            train_data.targets, y_pred_t, args.epsilon, train_data.true_groups_tensor.T)

        y_pred_test =  model(test_data.data)
        err_test = error_rate(test_data.targets, y_pred_test)
        # max_viol_test, viol_list_test = violation(
        #     test_data.targets, y_pred_test, args.epsilon, test_data.proxy_groups_tensor)
        max_viol_test, viol_list_test = violation(
            test_data.targets, y_pred_test, args.epsilon, test_data.true_groups_tensor.T)

        tb_writer.add_scalar('Train/Loss',main_loss.item(),e)
        tb_writer.add_scalar('Train/Error', err,e)
        tb_writer.add_scalar('Test/Error',err_test,e)
        tb_writer.add_scalar('Train/Max_violation',max_viol,e)
        tb_writer.add_scalar('Test/Max_violation',max_viol_test,e)
        
        if e % (args.log*10) == 0:
          print("Epoch %d | Error = %.3f | Viol = %.3f | Viol_test = %.3f" %
                (e, err, max_viol, max_viol_test), flush=True)
  return


def train(args, device=None):
  """Traqining model"""
  tb_writer = SummaryWriter(log_dir=args.log_dir)
  if device is None:
    device = torch.device('cpu')

  criterion = nn.BCEWithLogitsLoss()
  train_data, test_data = load_data_adult(args.noise_level, 
                                          uniform_groups=args.uniform_groups, 
                                          min_group_frac=args.min_group_frac, 
                                          use_noise_array=args.use_noise_array,
                                          group_features_type=args.group_features_type, 
                                          num_group_clusters=args.num_group_clusters)

  train_data.to(device)
  test_data.to(device)

  model = Linear(train_data.num_features)
  model_last = deepcopy(model).to(device)

  optimizers = {'main': torch.optim.Adagrad(model.parameters(), lr=args.learning_rate),
                'last': torch.optim.Adagrad(model_last.parameters(), lr=args.learning_rate)
  }
  const_losses = {'main': torch.zeros(1), 'last': torch.zeros(1)}
 

  for e in range(args.epochs):
    # Check for the beginning of a new epoch.
    if args.resample_proxy_groups and args.constrained:

      # Only resample proxy groups every epochs_per_resample epochs.
      if e % args.epochs_per_resample == 0:
        # Resample the group at the beginning of the epoch.
        # Get groups_train from a ball around init_proxy_groups_train.
        if args.uniform_groups:
          train_data.proxy_groups_tensor = generate_proxy_groups_uniform(
              len(train_data), min_group_frac=args.min_group_frac).long().to(device)
        elif args.use_noise_array:
          train_data.noise_array = train_data.get_noise_array()
          train_data.proxy_groups_tensor = generate_proxy_groups_noise_array(
              train_data.proxy_groups_tensor, noise_array=train_data.noise_array).long().to(device)
        else:
          train_data.proxy_groups_tensor = generate_proxy_groups_single_noise(
              train_data.proxy_groups_tensor, noise_param=args.noise_level).long().to(device)
    model.train()
    model_last.train()
    optimizers['main'].zero_grad()
    y_pred = model(train_data.data)
    # loss = criterion(y_pred, (train_data.target-.5)*2)
    main_loss = criterion(y_pred, train_data.targets)
    const_losses['main'] = lagrangian_loss(train_data.targets, y_pred, args.epsilon, train_data.proxy_groups_tensor,
                                  criterion, device, gamma=args.dual_scale) 
    y_pred_last = model_last(train_data.data)
    const_losses['last'] = lagrangian_loss(train_data.targets, y_pred_last, args.epsilon, train_data.proxy_groups_tensor,
                                  criterion, device, gamma=args.dual_scale) 
    model.backward_dro(main_loss, const_losses, optimizers, model_last)
    model_last.load_state_dict(model.state_dict().copy())
    optimizers['main'].step()

    # Snapshot iterate once in 1000 loops.
    if e % args.log == 0:
      model.eval()
      with torch.no_grad():
        y_pred_t = model(train_data.data)
        err = error_rate(train_data.targets, y_pred_t)
        max_viol, viol_list = violation(
            train_data.targets, y_pred_t, args.epsilon, train_data.true_groups_tensor.T)

        y_pred_test =  model(test_data.data)
        err_test = error_rate(test_data.targets, y_pred_test)
        max_viol_test, viol_list_test = violation(
            test_data.targets, y_pred_test, args.epsilon, test_data.true_groups_tensor.T)

        tb_writer.add_scalar('Train/Loss',main_loss.item(),e)
        tb_writer.add_scalar('Train/ConstraintLoss',const_losses['main'].item(),e)
        tb_writer.add_scalar('Train/Error', err,e)
        tb_writer.add_scalar('Test/Error',err_test,e)
        tb_writer.add_scalar('Train/Max_violation',max_viol,e)
        tb_writer.add_scalar('Test/Max_violation',max_viol_test,e)
        
        if e % (args.log*10) == 0:
          print("Epoch %d | Error = %.3f | Viol = %.3f | Viol_test = %.3f" %
                (e, err, max_viol, max_viol_test), flush=True)
  return

    

  


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--constrained', action='store_true')
    parser.add_argument('-e', '--epsilon', default=0.01, type=float) 
    parser.add_argument('-b', '--batch_size', default=50, type=int)
    parser.add_argument('-l', '--learning_rate', default=0.01, type=float)
    parser.add_argument('-i', '--iterations',default=50000, type=int)
    parser.add_argument('-p', '--epochs',default=50000, type=int)
    parser.add_argument('-o', '--log',default=100, type=int)
    parser.add_argument('-g', '--dual_scale', default=0.1, type=float)
    parser.add_argument('-d', '--logdir', default='./runs_noise', type=str)
    parser.add_argument('-n', '--noise_level', default=0.3, type=float)
    parser.add_argument('-un', '--use_noise_array', action='store_true')
    parser.add_argument('-ug', '--uniform_groups', action='store_true')
    parser.add_argument('-gf', '--min_group_frac', default=0.01, type=float)
    parser.add_argument('-ft', '--group_features_type', default='full_group_vec', type=str)
    parser.add_argument('-nc', '--num_group_clusters',default=100, type=int)
    parser.add_argument('-rp', '--resample_proxy_groups', action='store_true')
    parser.add_argument('-er', '--epochs_per_resample', default=1, type=int)
    parser.add_argument('-f', '--full_step',default=100, type=int)


    args = parser.parse_args()

    args.log_dir =  increment_dir(Path(args.logdir) / 'exp')
    os.makedirs(args.log_dir)
    yaml_file = str(Path(args.log_dir) / "args.yaml")
    with open(yaml_file, 'w') as out:
      yaml.dump(args.__dict__, out, default_flow_style=False)
    # Device configuration
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if args.constrained:
      train(args)
    else:
      train_unconstrained(args)