# Lint as: python3
"""Intersectional fairness with many constraints."""
import argparse
import os
import yaml
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from copy import deepcopy
from torch.utils.tensorboard import SummaryWriter

from data import load_data
from model import Linear
from utils import increment_dir


def error_rate(labels, predictions, groups=None):
  # Returns the error rate for given labels and predictions.
  if groups is not None:
    if groups.sum() == 0.0:
      return 0.0
    predictions = predictions[groups]
    labels = labels[groups]
  signed_labels = labels - 0.5
  return min((signed_labels * predictions <= 0.0).float().mean(),
                (signed_labels * predictions >= 0.0).float().mean())

def violation(
    labels, predictions, epsilon, group_memberships_list):
  # Returns violations across different group feature thresholds.
  viol_list = []
  overall_error = error_rate(labels, predictions).cpu()
  for kk in range(group_memberships_list.shape[0]):
    group_err = error_rate(
        labels, predictions, group_memberships_list[kk, :].reshape(-1,)).cpu()
    viol_list += [group_err - overall_error - epsilon]
  return np.max(viol_list), viol_list

def lagrangian_loss(labels, predictions, epsilon, 
                    group_memberships_list, criterion,
                    device, gamma=0.1, alpha=0.1):
  # Returns the lagrangain loss of the constrained problem
  total_loss = torch.zeros(1).to(device)
  main_loss = criterion(predictions, labels)
  m = group_memberships_list.shape[0]
  for kk in range(m):
    g_inds = group_memberships_list[kk, :]
    group_loss =criterion(predictions[g_inds],labels[g_inds])
    const_loss = group_loss - main_loss - epsilon
    total_loss += 1/(m+1) * (1/m + torch.exp(const_loss * alpha / gamma))
  return total_loss


def evaluate(
    features, labels, model, epsilon, group_membership_list, path=None):
  # Evaluates and prints stats.
  predictions = model(features).detach().cpu().numpy().reshape(-1,)
  print("Error %.3f" % error_rate(labels.cpu(), predictions))
  _, viol_list = violation(labels.cpu(), predictions, epsilon, group_membership_list.cpu())
  print("99p Violation %.3f" % np.quantile(viol_list, 0.99))
  print("95p Violation %.3f" % np.quantile(viol_list, 0.95))
  print()
  torch.save(torch.stack(viol_list),path)






def train_unconstrained(args):
  # torch.random.manual_seed(121212)
  # np.random.seed(212121)

  tb_writer = SummaryWriter(log_dir=args.log_dir)
  # criterion = nn.HingeEmbeddingLoss()
  criterion = nn.BCEWithLogitsLoss()
  train_data, val_data, test_data = load_data()

  model = Linear(train_data.data.shape[1])
  optim = torch.optim.Adagrad(model.parameters(), lr=0.1)

  objectives_list = []
  objectives_list_test = []
  objectives_list_val = []
  violations_list = []
  violations_list_test = []
  violations_list_val = []
  model_weights = []

  for i in range(args.iterations):
    optim.zero_grad()
    y_pred = model(train_data.data)
    # loss = criterion(y_pred, (train_data.target-.5)*2)
    loss = criterion(y_pred, train_data.target)
    loss.backward()
    optim.step()

    # Snapshot iterate once in 1000 loops.
    if i % args.log == 0:
      err = error_rate(train_data.target, y_pred)
      max_viol, viol_list = violation(
          train_data.target, y_pred, args.epsilon, train_data.group_memberships_list)

      y_pred_test =  model(test_data.data)
      err_test = error_rate(test_data.target, y_pred_test)
      max_viol_test, viol_list_test = violation(
          test_data.target, y_pred_test, args.epsilon, test_data.group_memberships_list)

      y_pred_val = model(val_data.data)
      err_vali =  error_rate(val_data.target, y_pred_val)
      max_viol_val, viol_list_val = violation(
          val_data.target, y_pred_val, args.epsilon, val_data.group_memberships_list)

      objectives_list.append(err)
      objectives_list_test.append(err_test)
      objectives_list_val.append(err_vali)
      violations_list.append(viol_list)
      violations_list_test.append(viol_list_test)
      violations_list_val.append(viol_list_val)
      model_weights.append(model.state_dict().copy())
      
      tb_writer.add_scalar('Train/Error',err,i)
      tb_writer.add_scalar('Val/Error',err_vali,i)
      tb_writer.add_scalar('Test/Error',err_test,i)
      tb_writer.add_scalar('Train/Max_violation',max_viol,i)
      tb_writer.add_scalar('Val/Max_violation',max_viol_val,i)
      tb_writer.add_scalar('Test/Max_violation',max_viol_test,i)

      if i % (args.log*10) == 0:
        print("Epoch %d | Error = %.3f | Viol = %.3f | Viol_vali = %.3f" %
              (i, err, max_viol, max_viol_val), flush=True)

  # Best candidate index.
  best_ind = np.argmin(objectives_list)
  model.load_state_dict(model_weights[best_ind])

  print("Train:")
  evaluate(train_data.data, train_data.target, model,
           args.epsilon, train_data.group_memberships_list,
           path= str(Path(args.log_dir) / "train.pt"))
  print("\nVal:")
  evaluate(val_data.data, val_data.target, model,
           args.epsilon, val_data.group_memberships_list,
           path= str(Path(args.log_dir) / "val.pt"))
  print("\nTest:")
  evaluate(test_data.data, test_data.target, model,
           args.epsilon, test_data.group_memberships_list,
           path= str(Path(args.log_dir) / "test.pt"))
  return

def train_dro(args, device=None):
  # torch.random.manual_seed(121212)
  # np.random.seed(212121)
  if device is None:
    device = torch.device('cpu')

  tb_writer = SummaryWriter(log_dir=args.log_dir)
  # criterion = nn.HingeEmbeddingLoss()
  criterion = nn.BCEWithLogitsLoss()
  train_data, val_data, test_data = load_data()
  train_data.to(device)
  val_data.to(device)
  test_data.to(device)

  # train_loader =  torch.utils.data.DataLoader(train_data, batch_size = args.batch_size, shuffle=True)
  # n = len(train_loader)
  model = Linear(train_data.data.shape[1]).to(device)
  model_last = deepcopy(model).to(device)
  optimizers = {'main': torch.optim.Adagrad(model.parameters(), lr=args.learning_rate),
                'last': torch.optim.Adagrad(model_last.parameters(), lr=args.learning_rate)
  }
  const_losses = {'main': torch.zeros(1), 'last': torch.zeros(1)}


  objectives_list = []
  objectives_list_test = []
  objectives_list_val = []
  violations_list = []
  violations_list_test = []
  violations_list_val = []
  model_weights = []
  n_g = train_data.group_memberships_list.shape[0]
  model_last.train()
  for i in range(args.iterations):
  # for p in range(args.epochs):
  #   for i,(data, target, group, gml) in enumerate(train_loader):
    t = i
    fs = not t % args.full_step
    model.train()
    optimizers['main'].zero_grad()
    y_pred = model(train_data.data)
    # loss = criterion(y_pred, (train_data.target-.5)*2)
    main_loss = criterion(y_pred, train_data.target)
    if fs:
      g_ind = torch.arange(train_data.group_memberships_list.shape[0])
    else:
      g_ind = torch.randint(0,n_g,(args.num_constraint,))
    const_losses['main'] = lagrangian_loss(train_data.target, y_pred, args.epsilon, train_data.group_memberships_list[g_ind,:],
                                  criterion, device, gamma=args.dual_scale) 
    y_pred_last = model_last(train_data.data)
    if not fs:
      const_losses['last'] = lagrangian_loss(train_data.target, y_pred_last, args.epsilon, train_data.group_memberships_list[g_ind,:],
                                    criterion, device, gamma=args.dual_scale) 
    model.backward_dro(main_loss, const_losses, optimizers, model_last,full_step=fs)
    model_last.load_state_dict(model.state_dict().copy())
    optimizers['main'].step()

    # Snapshot iterate once in 1000 loops.
    if t % args.log == 0:
      model.eval()
      with torch.no_grad():
        y_pred_t = model(train_data.data)
        err = error_rate(train_data.target, y_pred_t)
        max_viol, viol_list = violation(
            train_data.target, y_pred_t, args.epsilon, train_data.group_memberships_list)

        y_pred_test =  model(test_data.data)
        err_test = error_rate(test_data.target, y_pred_test)
        max_viol_test, viol_list_test = violation(
            test_data.target, y_pred_test, args.epsilon, test_data.group_memberships_list)

        y_pred_val = model(val_data.data)
        err_vali =  error_rate(val_data.target, y_pred_val)
        max_viol_val, viol_list_val = violation(
            val_data.target, y_pred_val, args.epsilon, val_data.group_memberships_list)

        objectives_list.append(err.item())
        objectives_list_test.append(err_test)
        objectives_list_val.append(err_vali)
        violations_list.append(viol_list)
        violations_list_test.append(viol_list_test)
        violations_list_val.append(viol_list_val)
        model_weights.append(model.state_dict().copy())

        tb_writer.add_scalar('Train/Loss',main_loss.item(),t)
        tb_writer.add_scalar('Train/ConstraintLoss',const_losses['main'].item(),t)
        tb_writer.add_scalar('Train/Error',err,t)
        tb_writer.add_scalar('Val/Error',err_vali,t)
        tb_writer.add_scalar('Test/Error',err_test,t)
        tb_writer.add_scalar('Train/Max_violation',max_viol,t)
        tb_writer.add_scalar('Val/Max_violation',max_viol_val,t)
        tb_writer.add_scalar('Test/Max_violation',max_viol_test,t)
        
        if t % (args.log*10) == 0:
          print("Epoch %d | Error = %.3f | Viol = %.3f | Viol_vali = %.3f" %
                (t, err, max_viol, max_viol_val), flush=True)

  # Best candidate index.
  best_ind = np.argmax(objectives_list)
  model.load_state_dict(model_weights[best_ind])

  print("Train:")
  evaluate(train_data.data, train_data.target, model,
           args.epsilon, train_data.group_memberships_list,
           path= str(Path(args.log_dir) / "train.pt"))
  print("\nVal:")
  evaluate(val_data.data, val_data.target, model,
           args.epsilon, val_data.group_memberships_list,
           path= str(Path(args.log_dir) / "val.pt"))
  print("\nTest:")
  evaluate(test_data.data, test_data.target, model,
           args.epsilon, test_data.group_memberships_list,
           path= str(Path(args.log_dir) / "test.pt"))
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
    parser.add_argument('-d', '--logdir', default='./runs', type=str)
    parser.add_argument('-f', '--full_step',default=500, type=int)
    parser.add_argument('-n', '--num_constraint',default=10, type=int)
    
    args = parser.parse_args()

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
      train_dro(args,device=device)