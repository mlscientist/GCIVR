# Lint as: python3
"""Cross-group ranking fairness experiments with per-query constraints."""
import argparse
import yaml
import os
from pathlib import Path
from time import time
from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from utils import increment_dir, evaluate_results_ranking, \
                  error_rate_lambda, get_mask_ranking
from data import MSLTR
from model import Ranking



def group_tensors(predictions,
                  groups,
                  pos_group,
                  neg_group=None,
                  queries=None,
                  query_index=None):
  group_mask = get_mask_ranking(groups, pos_group, neg_group)
  if (queries is not None) and (query_index is not None):
      group_mask = group_mask & (queries == query_index)

  group_labels = torch.ones(group_mask.float().sum().int().item(), dtype=torch.float32)

  group_predictions = predictions[group_mask]

  return group_predictions, group_labels

def lagrangian_loss(labels, predictions, epsilon, 
                    groups, criterion, constraint_groups, 
                    gamma=0.1, alpha=0.1):
  # Returns the lagrangain loss of the constrained problem
  total_loss = torch.zeros(1)
  # main_loss = criterion(predictions, labels)
  m = 2
  for ((pos_group0, neg_group0), (pos_group1, neg_group1)) in constraint_groups:
    group0_predictions, group0_labels = group_tensors(
        predictions, groups, pos_group0, neg_group=neg_group0)
    group1_predictions, group1_labels = group_tensors(
        predictions, groups, pos_group1, neg_group=neg_group1)
    group_loss0 =criterion(group0_predictions,group0_labels)
    group_loss1 =criterion(group1_predictions,group1_labels)
    const_loss0 = group_loss0 - group_loss1 - epsilon
    total_loss += 1/(m+1) * (1/m + torch.exp(const_loss0 * alpha / gamma))
    const_loss1 = group_loss1 - group_loss0 - epsilon
    total_loss += 1/(m+1) * (1/m + torch.exp(const_loss1 * alpha / gamma))
  return total_loss


def train_unconstrained(args):
  # torch.random.manual_seed(121212)
  # np.random.seed(212121)

  if not args.constrained:
    # Unconstrained optimization.
    if args.constraint_type == 'marginal_equal_opportunity':
      valid_groups = [(0, None), (1, None)]
    elif args.constraint_type == 'cross_group_equal_opportunity':
      valid_groups = [(0, 1), (1, 0)]
  else:
    # Constrained optimization.
    if args.constraint_type == 'marginal_equal_opportunity':
      constraint_groups = [((0, None), (1, None))]
      valid_groups = [(0, None), (1, None)]
    elif args.constraint_type == 'cross_group_equal_opportunity':
      constraint_groups = [((0, 1), (1, 0))]
      valid_groups = [(0, 1), (1, 0)]
    elif args.constraint_type == 'custom':
      constraint_groups = args.constraint_groups
    else:
      constraint_groups = []
  
  tb_writer = SummaryWriter(log_dir=args.log_dir)
  # criterion = nn.HingeEmbeddingLoss()
  criterion = nn.BCEWithLogitsLoss()
  train_data = MSLTR(args.data_path, split='train', data_size=args.data_size, seed=args.seed)
  test_data  = MSLTR(args.data_path, split='test',  data_size=args.data_size, seed=args.seed)

  args.metric_fn = error_rate_lambda(train_data)
  args.metric_fn_test = error_rate_lambda(test_data)
  model = Ranking(251)
  if args.optimizer == 'sgd':
    optim = torch.optim.SGD(model.parameters(), lr=args.lr)
  elif args.optimizer == 'adam':
    optim = torch.optim.Adam(model.parameters(), lr=args.lr)
  else:
    optim = torch.optim.Adagrad(model.parameters(), lr=args.lr)

  objectives_train, objectives_test = [], []
  group_violations_train, group_violations_test = [], []
  query_violations_train, query_violations_test = [], []
  query_violations_full_train, query_violations_full_test = [], []
  query_ndcgs_train, query_ndcgs_test = [], []
  batch_index = 0
  for e in range(args.epochs):
    model.train()
    loss_acc = 0.
    for ii in tqdm(range(args.iterations)):
      optim.zero_grad()
      feats, groups = train_data.query(batch_index)
      labels = torch.ones(feats.shape[0])
      preds = model(feats)
      loss = criterion(preds,labels)
      loss.backward()
      loss_acc += loss.item()
      if ii % args.log_step == 0:
        tb_writer.add_scalar('Train/Loss',loss_acc/args.log_step,ii + e*args.iterations)
        loss_acc = 0.
      optim.step()

      batch_index = (batch_index + 1) % train_data.num_queries

    if args.metric_fn is None:
      tt = time()
      error_train, _, group_viol_train, query_viol_train, query_viols_train = evaluate_results_ranking(
          model, train_data, args)
      print("Train evaluation time is: {}".format(time()-tt))
      query_ndcgs_train.append(0)
    else:
      tt = time()
      error_train, group_error_train, query_error_train, query_ndcg_train = args.metric_fn(
          model, valid_groups)
      print("Train evaluation time is: {}".format(time()-tt))
      group_viol_train = [
          group_error_train[0] - group_error_train[1], group_error_train[1] - group_error_train[0]
      ]
      query_viol_train = [np.max(np.abs(query_error_train[:, 0] - query_error_train[:, 1]))]
      query_viols_train = [np.abs(query_error_train[:, 0] - query_error_train[:, 1])]
      query_ndcgs_train.append(np.mean(query_ndcg_train))

    objectives_train.append(error_train)
    group_violations_train.append(
        [x - args.epsilon for x in group_viol_train])
    query_violations_train.append(
        [x - args.epsilon for x in query_viol_train])
    query_violations_full_train.append(
        [x - args.epsilon for x in query_viols_train])
    print(
        '\r Epoch %d: train error = %.3f, train group violation = %.3f, train query violation = %.3f'
        % (e, objectives_train[-1], max(
            group_violations_train[-1]), max(query_violations_train[-1])))

    ## Test set evaluation
    if args.metric_fn_test is None:
      tt = time()
      error_test, _, group_viol_test, query_viol_test, query_viols_test = evaluate_results_ranking(
          model, test_data, args)
      print("Test evaluation time is: {}".format(time()-tt))
      query_ndcgs_test.append(0)
    else:
      tt = time()
      error_test, group_error_test, query_error_test, query_ndcg_test = args.metric_fn_test(
          model, valid_groups)
      print("Test evaluation time is: {}".format(time()-tt))
      group_viol_test = [
          group_error_test[0] - group_error_test[1], group_error_test[1] - group_error_test[0]
      ]
      query_viol_test = [np.max(np.abs(query_error_test[:, 0] - query_error_test[:, 1]))]
      query_viols_test = [np.abs(query_error_test[:, 0] - query_error_test[:, 1])]
      query_ndcgs_test.append(np.mean(query_ndcg_test))

    objectives_test.append(error_test)
    group_violations_test.append(
        [x - args.epsilon for x in group_viol_test])
    query_violations_test.append(
        [x -  args.epsilon for x in query_viol_test])
    query_violations_full_test.append(
        [x -  args.epsilon for x in query_viols_test])
    print(
        '\r Epoch %d: test error = %.3f, test group violation = %.3f, test query violation = %.3f'
        % (e, objectives_test[-1], max(
            group_violations_test[-1]), max(query_violations_test[-1])))

    
    tb_writer.add_scalar('Train/Error',objectives_train[-1],e)
    tb_writer.add_scalar('Test/Error',objectives_test[-1],e)
    tb_writer.add_scalar('Train/Group_violation',max(group_violations_train[-1]),e)
    tb_writer.add_scalar('Test/Group_violation',max(group_violations_test[-1]),e)
    tb_writer.add_scalar('Train/Query_violation',max(query_violations_train[-1]),e)
    tb_writer.add_scalar('Test/Query_violation',max(query_violations_test[-1]),e)

  return




def train_dro(args):
  # torch.random.manual_seed(121212)
  # np.random.seed(212121)

  if not args.constrained:
    # Unconstrained optimization.
    if args.constraint_type == 'marginal_equal_opportunity':
      valid_groups = [(0, None), (1, None)]
    elif args.constraint_type == 'cross_group_equal_opportunity':
      valid_groups = [(0, 1), (1, 0)]
  else:
    # Constrained optimization.
    if args.constraint_type == 'marginal_equal_opportunity':
      constraint_groups = [((0, None), (1, None))]
      valid_groups = [(0, None), (1, None)]
    elif args.constraint_type == 'cross_group_equal_opportunity':
      constraint_groups = [((0, 1), (1, 0))]
      valid_groups = [(0, 1), (1, 0)]
    elif args.constraint_type == 'custom':
      constraint_groups = args.constraint_groups
    else:
      constraint_groups = []
  
  tb_writer = SummaryWriter(log_dir=args.log_dir)
  # criterion = nn.HingeEmbeddingLoss()
  criterion = nn.BCEWithLogitsLoss()
  train_data = MSLTR(args.data_path, split='train', data_size=args.data_size, seed=args.seed)
  test_data  = MSLTR(args.data_path, split='test',  data_size=args.data_size, seed=args.seed)

  args.metric_fn = error_rate_lambda(train_data)
  args.metric_fn_test = error_rate_lambda(test_data)
  model = Ranking(251)
  model_last = deepcopy(model)
  
  const_losses = {'main': torch.zeros(1), 'last': torch.zeros(1)}


  if args.optimizer == 'sgd':
    optimizers = {'main': torch.optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay),
                'last':  torch.optim.SGD(model_last.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    }
  elif args.optimizer == 'adam':
    optimizers = {'main': torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay),
                'last':  torch.optim.Adam(model_last.parameters(), lr=args.lr,weight_decay=args.weight_decay)
    }
  else:
    optimizers = {'main': torch.optim.Adagrad(model.parameters(), lr=args.lr,weight_decay=args.weight_decay),
                'last':  torch.optim.Adagrad(model_last.parameters(), lr=args.lr,weight_decay=args.weight_decay)
    }

  objectives_train, objectives_test = [], []
  group_violations_train, group_violations_test = [], []
  query_violations_train, query_violations_test = [], []
  query_violations_full_train, query_violations_full_test = [], []
  query_ndcgs_train, query_ndcgs_test = [], []
  batch_index = 0
  for e in range(args.epochs):
    model.train()
    model_last.train()
    loss_acc = 0.
    for ii in tqdm(range(args.iterations)):
      optimizers['main'].zero_grad()
      feats, groups = train_data.query(batch_index)
      labels = torch.ones(feats.shape[0])
      preds = model(feats)
      main_loss = criterion(preds, labels)
      const_losses['main'] = lagrangian_loss(labels, preds, args.epsilon, 
                                groups, criterion, constraint_groups, gamma=args.dual_scale)
      pred_last = model_last(feats)
      const_losses['last'] = lagrangian_loss(labels, pred_last, args.epsilon, 
                                groups, criterion, constraint_groups, gamma=args.dual_scale)
      # model_last.load_state_dict(model.state_dict().copy())
      model.backward_dro(main_loss, const_losses, optimizers, model_last)
      model_last.load_state_dict(model.state_dict().copy())
      optimizers['main'].step()
      loss_acc += main_loss.item()
      if ii % args.log_step == 0:
        tb_writer.add_scalar('Train/Loss',loss_acc/args.log_step,ii + e*args.iterations)
        loss_acc = 0.
      batch_index = (batch_index + 1) % train_data.num_queries

    model.eval()
    model_last.eval()
    if args.metric_fn is None:
      tt = time()
      error_train, _, group_viol_train, query_viol_train, query_viols_train = evaluate_results_ranking(
          model, train_data, args)
      print("Train evaluation time is: {}".format(time()-tt))
      query_ndcgs_train.append(0)
    else:
      tt = time()
      error_train, group_error_train, query_error_train, query_ndcg_train = args.metric_fn(
          model, valid_groups)
      print("Train evaluation time is: {}".format(time()-tt))
      group_viol_train = [
          group_error_train[0] - group_error_train[1], group_error_train[1] - group_error_train[0]
      ]
      query_viol_train = [np.max(np.abs(query_error_train[:, 0] - query_error_train[:, 1]))]
      query_viols_train = [np.abs(query_error_train[:, 0] - query_error_train[:, 1])]
      query_ndcgs_train.append(np.mean(query_ndcg_train))

    objectives_train.append(error_train)
    group_violations_train.append(
        [x - args.epsilon for x in group_viol_train])
    query_violations_train.append(
        [x - args.epsilon for x in query_viol_train])
    query_violations_full_train.append(
        [x - args.epsilon for x in query_viols_train])
    print(
        '\r Epoch %d: train error = %.3f, train group violation = %.3f, train query violation = %.3f'
        % (e, objectives_train[-1], max(
            group_violations_train[-1]), max(query_violations_train[-1])))

    ## Test set evaluation

    if args.metric_fn_test is None:
      tt = time()
      error_test, _, group_viol_test, query_viol_test, query_viols_test = evaluate_results_ranking(
          model, test_data, args)
      print("Test evaluation time is: {}".format(time()-tt))
      query_ndcgs_test.append(0)
    else:
      tt = time()
      error_test, group_error_test, query_error_test, query_ndcg_test = args.metric_fn_test(
          model, valid_groups)
      print("Test evaluation time is: {}".format(time()-tt))
      group_viol_test = [
          group_error_test[0] - group_error_test[1], group_error_test[1] - group_error_test[0]
      ]
      query_viol_test = [np.max(np.abs(query_error_test[:, 0] - query_error_test[:, 1]))]
      query_viols_test = [np.abs(query_error_test[:, 0] - query_error_test[:, 1])]
      query_ndcgs_test.append(np.mean(query_ndcg_test))

    objectives_test.append(error_test)
    group_violations_test.append(
        [x - args.epsilon for x in group_viol_test])
    query_violations_test.append(
        [x -  args.epsilon for x in query_viol_test])
    query_violations_full_test.append(
        [x -  args.epsilon for x in query_viols_test])
    print(
        '\r Epoch %d: test error = %.3f, test group violation = %.3f, test query violation = %.3f'
        % (e, objectives_test[-1], max(
            group_violations_test[-1]), max(query_violations_test[-1])))

    
    tb_writer.add_scalar('Train/Error',objectives_train[-1],e)
    tb_writer.add_scalar('Test/Error',objectives_test[-1],e)
    tb_writer.add_scalar('Train/Group_violation',max(group_violations_train[-1]),e)
    tb_writer.add_scalar('Test/Group_violation',max(group_violations_test[-1]),e)
    tb_writer.add_scalar('Train/Query_violation',max(query_violations_train[-1]),e)
    tb_writer.add_scalar('Test/Query_violation',max(query_violations_test[-1]),e)



if __name__=="__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument('-c', '--constrained', action='store_true')
  parser.add_argument('-t', '--constraint_type', default='cross_group_equal_opportunity', 
                      type=str, choices=['marginal_equal_opportunity', 'cross_group_equal_opportunity', 'custom'])
  parser.add_argument('-e', '--epsilon', default=0.25, type=float) 
  parser.add_argument('-b', '--batch_size', default=50, type=int)
  parser.add_argument('-l', '--lr', default=1e-3, type=float)
  parser.add_argument('-i', '--iterations',default=1000, type=int)
  parser.add_argument('-p', '--epochs',default=100, type=int)
  parser.add_argument('-s', '--data_size',default=1000, type=int)
  parser.add_argument('-ls', '--log_step',default=100, type=int)
  parser.add_argument('-g', '--dual_scale', default=500, type=float)
  parser.add_argument('-d', '--logdir', default='./runs_rank', type=str)
  parser.add_argument('-o', '--optimizer', default='adagrad', type=str)
  parser.add_argument('-f', '--full_step',default=100, type=int)
  parser.add_argument('-sd', '--seed',default=42, type=int)
  parser.add_argument('-dp', '--data_path', default='./data/MSLR-WEB10K/Fold1/', type=str)
  parser.add_argument('-w', '--weight_decay', default=0.1, type=float)


  
  args = parser.parse_args()
  args.metric_fn = None
  args.log_dir =  increment_dir(Path(args.logdir) / 'exp')
  os.makedirs(args.log_dir)
  yaml_file = str(Path(args.log_dir) / "args.yaml")
  with open(yaml_file, 'w') as out:
    yaml.dump(args.__dict__, out, default_flow_style=False)
  # Device configuration
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  if not args.constrained:
    train_unconstrained(args)
  else:
    train_dro(args)