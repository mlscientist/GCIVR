import re
from pathlib import Path
import glob
import torch
import numpy as np
import random
import math
import pandas as pd

def increment_dir(dir, comment=''):
    # Increments a directory runs/exp1 --> runs/exp2_comment
    n = 0  # number
    dir = str(Path(dir))  # os-agnostic
    dirs = sorted(glob.glob(dir + '*'))  # directories
    if dirs:
        matches = [re.search(r"exp(\d+)", d) for d in dirs]
        idxs = [int(m.groups()[0]) for m in matches if m]
        if idxs:
            n = max(idxs) + 1  # increment
    return dir + str(n) + ('_' + comment if comment else '')


# Helper functions for evaluation.
def error_rate(labels, predictions):
  """Computes error rate."""
  # Recall that the labels are binary (0 or 1).
  signed_labels = (labels * 2) - 1
  return torch.mean((signed_labels * predictions <= 0.0).float())


def group_error_rates(labels, predictions, groups):
  """Returns a list containing error rates for each protected group."""
  errors = []
  for jj in range(groups.shape[1]):
    if groups[:, jj].sum() == 0:  # Group is empty?
      errors.append(0.0)
    else:
      signed_labels_jj = 2 * labels[groups[:, jj] == 1] - 1
      predictions_jj = predictions[groups[:, jj] == 1]
      errors.append(torch.mean((signed_labels_jj * predictions_jj <= 0).float()))
  return errors


def tpr(labels, predictions):
    """Computes true positive rate."""
    # Recall that the labels are binary (0 or 1).
    signed_labels = (labels * 2) - 1
    predictions_pos = predictions[signed_labels > 0]
    return torch.mean((predictions_pos > 0.0).float())


def group_tprs(labels, predictions, group):
    """Returns a list containing tprs for each protected group."""
    if group.sum() == 0:  # Group is empty?
        tprs = 0.0
    else:
        signed_labels_jj = 2 * labels[group == 1] - 1
        predictions_jj = predictions[group == 1]
        predictions_jj_pos = predictions_jj[signed_labels_jj > 0]
        tprs = torch.mean((predictions_jj_pos > 0).float())
    return tprs

#########################

def _print_metric(dataset_name, metric_name, metric_value):
  """Prints metrics."""
  print('[metric] %s.%s=%f' % (dataset_name, metric_name, metric_value))


def compute_quantiles(features,
                      num_keypoints=10,
                      clip_min=None,
                      clip_max=None,
                      missing_value=None):
  """Computes quantiles for feature columns."""
  # Clip min and max if desired.
  if clip_min is not None:
    features = np.maximum(features, clip_min)
    features = np.append(features, clip_min)
  if clip_max is not None:
    features = np.minimum(features, clip_max)
    features = np.append(features, clip_max)
  # Make features unique.
  unique_features = np.unique(features)
  # Remove missing values if specified.
  if missing_value is not None:
    unique_features = np.delete(unique_features,
                                np.where(unique_features == missing_value))
  # Compute and return quantiles over unique non-missing feature values.
  return np.quantile(
      unique_features,
      np.linspace(0., 1., num=num_keypoints),
      interpolation='nearest').astype(float)

def print_metrics_results_dict(results_dict, iterate='best', unconstrained=True):
  """Prints metrics from results_dict."""
  index = -1
  if iterate == 'best':
    if unconstrained:
      index = np.argmin(np.array(results_dict['train.true_error_rates']))
    else:
      index = tfco.find_best_candidate_index(
          np.array(results_dict['train.true_error_rates']),
          np.array(results_dict['train.sampled_violations_max']).reshape(
              (-1, 1)),
          rank_objectives=True)
  for metric_name, values in results_dict.items():
    _print_metric(iterate, metric_name, values[index])


def add_summary_viols_to_results_dict(dataset,
                                      model,
                                      results_dict,
                                      dataset_name,
                                      noise_level,
                                      epsilon=0.03,
                                      n_resamples_per_candidate=10,
                                      use_noise_array=True,
                                      noise_array=None,
                                      uniform_groups=False,
                                      min_group_frac=0.05):
  """Adds metrics to results_dict."""

  predictions = model.predict(dataset.data)
  overall_error = error_rate(dataset.targets, predictions)
  results_dict[dataset_name + '.true_error_rates'].append(overall_error)

  overall_tpr = tpr(dataset.targets, predictions)
  init_proxy_group_tprs = group_tprs(dataset.targets, predictions, dataset.proxy_groups_tensor)
  proxy_group_tpr_violations = [
      overall_tpr - group_tpr - epsilon for group_tpr in init_proxy_group_tprs
  ]
  results_dict[dataset_name + '.proxy_group_violations'].append(
      max(proxy_group_tpr_violations))

  true_group_tprs = group_tprs(dataset.targets, predictions, dataset.true_groups_tensor)
  true_group_tpr_violations = [
      overall_tpr - group_tpr - epsilon for group_tpr in true_group_tprs
  ]
  results_dict[dataset_name + '.true_group_violations'].append(
      max(true_group_tpr_violations))

  sampled_violations = []
  for _ in range(n_resamples_per_candidate):
    # Resample proxy groups.
    if uniform_groups:
      sampled_groups = generate_proxy_groups_uniform(
          len(dataset), min_group_frac=min_group_frac)
    elif use_noise_array:
      sampled_groups = generate_proxy_groups_noise_array(
          dataset.proxy_groups_tensor, noise_array=noise_array)
    else:
      sampled_groups = generate_proxy_groups_single_noise(
          dataset.proxy_groups_tensor, noise_param=noise_level)
    sampled_group_tprs = group_tprs(dataset.targets, predictions, sampled_groups)
    sampled_group_tpr_violations = [
        overall_tpr - group_tpr - epsilon for group_tpr in sampled_group_tprs
    ]
    sampled_violations.append(max(sampled_group_tpr_violations))
  results_dict[dataset_name + '.sampled_violations_max'].append(
      max(sampled_violations))
  results_dict[dataset_name + '.sampled_violations_90p'].append(
      np.percentile(np.array(sampled_violations), 90))
  return results_dict


def generate_proxy_groups_single_noise(input_groups, noise_param=1):
  """Generate proxy groups within noise noise_param."""
  proxy_groups = input_groups.clone()
  num_groups = input_groups.shape[1]
  num_datapoints = input_groups.shape[0]
  noise_idx = random.sample(
      range(num_datapoints), int(noise_param * num_datapoints))
  for j in noise_idx:
    group_index = -1
    for i in range(num_groups):
      if proxy_groups[j][i] == 1:
        proxy_groups[j][i] = 0
        group_index = i
        allowed_new_groups = list(range(num_groups))
        allowed_new_groups.remove(group_index)
        new_group_index = random.choice(allowed_new_groups)
        proxy_groups[j][new_group_index] = 1
        break
    if group_index == -1:
      print('missing group information for datapoint ', j)
  return proxy_groups


def generate_proxy_groups_uniform(num_examples, min_group_frac=0.05):
  """Generate proxy groups within noise noise_param."""

  # Generate a random array of the same shape as input groups. Each column
  # in the array is a a random binary vector where the number of 1's is at least
  # min_group_size.
  group_frac = np.random.uniform(min_group_frac, 1)
  num_in_group = int(num_examples * group_frac)
  group_assignment = np.array([0] * (num_examples - num_in_group) +
                              [1] * num_in_group)
  np.random.shuffle(group_assignment)
  return torch.tensor(group_assignment.reshape((-1, 1)))

def generate_proxy_groups_noise_array(input_groups, noise_array=None):
  """Generate proxy groups within noise noise_param."""

  proxy_groups = input_groups.clone()
  num_groups = input_groups.shape[1]

  for row in proxy_groups:
    new_j = -1
    for k in range(num_groups):
      if row[k] == 1:
        # draw from noise_params to decide which group to switch to.
        new_j = np.random.choice(num_groups, 1, p=noise_array[k])
        row[k] = 0
    assert new_j >= 0
    row[new_j] = 1

  return proxy_groups


############### ranking #####################
def evaluate_results_ranking(model, test_set, args):
  """Returns error rates and violation metrics."""
  # Returns overall, group error rates, group-level constraint violations,
  # query-level constraint violations for model on test set.
  model.eval()
  if args.constraint_type == 'marginal_equal_opportunity':
    g0_error, g0_query_error = group_error_rate_ranking(model, test_set, 0)
    g1_error, g1_query_error = group_error_rate_ranking(model, test_set, 1)
    group_violations = [g0_error - g1_error, g1_error - g0_error]
    query_violations = (g0_query_error - g1_query_error).abs().max().item()
    query_violations_full = (g0_query_error - g1_query_error).abs()
    return (error_rate_ranking(model, test_set), [g0_error, g1_error], group_violations,
            query_violations, query_violations_full)
  else:
    g00_error, g00_query_error = group_error_rate_ranking(model, test_set, 0, 0)
    g01_error, g01_query_error = group_error_rate_ranking(model, test_set, 0, 1)
    g10_error, g10_query_error = group_error_rate_ranking(model, test_set, 1, 1)
    g11_error, g11_query_error = group_error_rate_ranking(model, test_set, 1, 1)
    group_violations_offdiag = [g01_error - g10_error, g10_error - g01_error]
    group_violations_diag = [g00_error - g11_error, g11_error - g00_error]
    query_violations_offdiag = (g01_query_error - g10_query_error).abs().max().item()
    query_violations_diag = (g00_query_error - g11_query_error).abs().max().item()
    query_violations_offdiag_full = (g01_query_error - g10_query_error).abs()
    query_violations_diag_full = (g00_query_error - g11_query_error).abs()

    if args.constraint_type == 'cross_group_equal_opportunity':
      return (error_rate_ranking(model,
                         test_set), [[g00_error, g01_error],
                                     [g10_error, g11_error]], group_violations_offdiag,
              [query_violations_offdiag], query_violations_offdiag_full)
    else:
      return (error_rate_ranking(model, test_set), [[g00_error, g01_error],
                                                    [g10_error, g11_error]],
              group_violations_offdiag + group_violations_diag,
              query_violations_offdiag + query_violations_diag,
                  torch.cat((query_violations_offdiag_full,
                                  query_violations_diag_full)))

def error_rate_ranking(model, dataset):
  """Returns error rate for Keras model on dataset."""

  with torch.no_grad():
    diff = model(dataset.features_pairs)
  return (diff <= 0).float().mean().item()

def group_error_rate_ranking(model, dataset, pos_group, neg_group=None):
  """Returns error rate for Torch model on data set."""
  # Returns error rate for Torch model on data set, considering only document
  # pairs where the protected group for the positive document is pos_group, and
  # the protected group for the negative document (if specified) is neg_group.
  with torch.no_grad():
    diff = model(dataset.features_pairs)
    mask = get_mask_ranking(dataset.group_pairs, pos_group, neg_group)
    diff = diff[mask].reshape((-1))

    unique_qids = dataset.queries_pairs.unique()
    masked_queries = dataset.queries_pairs[mask]
    query_group_errors = torch.zeros(len(unique_qids))
    for qid in unique_qids:
      masked_queries_qid = (masked_queries == qid)
      if (masked_queries_qid).any():
        query_group_errors[qid] = (diff[masked_queries_qid] < 0).float().mean() + \
          0.5 * (diff[masked_queries_qid] == 0).float().mean()
      else:
        query_group_errors[qid] = 0
    return (diff < 0).float().mean().item() + 0.5 * (diff == 0).float().mean().item(), query_group_errors

def get_mask_ranking(groups, pos_group, neg_group=None):
  """Returns a boolean mask selecting positive negative document pairs."""
  # Returns a boolean mask selecting positive-negative document pairs where
  # the protected group for  the positive document is pos_group and
  # the protected group for the negative document (if specified) is neg_group.
  # Repeat group membership positive docs as many times as negative docs.
  mask_pos = groups[:, 0] == pos_group

  if neg_group is None:
    return mask_pos
  else:
    mask_neg = groups[:, 1] == neg_group
    return mask_pos & mask_neg


# nDCG
def dcg(labels, at=None):
  """Compute DCG given labels in order."""
  result = 0.0
  position = 2
  for i in labels:
    if i != 0:
      result += 1 / math.log2(position)
    position += 1
    if at is not None and (position >= at + 2):
      break
  return result


def ndcg(labels, at=None):
  """Compute nDCG given labels in order."""
  return dcg(labels, at=at) / dcg(sorted(labels, reverse=True), at=at)


# A faster 'get_error_rate'
def error_rate_lambda(dataset):
  """Returns error rate for Torch model on dataset."""
  pos_row_id = dataset.labels == 1
  neg_row_id = ~pos_row_id
  row_numbers = np.array(range(len(dataset.labels)))
  pos_data = pd.DataFrame({
      'row_ids': row_numbers[pos_row_id],
      'labels': dataset.labels[pos_row_id].cpu().numpy(),
      'queries': dataset.queries[pos_row_id].cpu().numpy(),
      'groups': dataset.groups[pos_row_id].cpu().numpy()
  })
  neg_data = pd.DataFrame({
      'row_ids': row_numbers[neg_row_id],
      'labels': dataset.labels[neg_row_id].cpu().numpy(),
      'queries':dataset.queries[neg_row_id].cpu().numpy(),
      'groups': dataset.groups[neg_row_id].cpu().numpy()
  })
  pairs = pos_data.merge(
      neg_data, on='queries', how='outer', suffixes=('_pos', '_neg'))

  def error_rate_helper(model, groups=None, at=10):
    model.eval()
    with torch.no_grad():
      preds = model.model(dataset.features[:,:model.dim]).squeeze().detach().cpu().numpy()
    error = (np.array(
        preds[pairs['row_ids_pos']] < preds[pairs['row_ids_neg']])) + 0.5 * (
            np.array(
                preds[pairs['row_ids_pos']] == preds[pairs['row_ids_neg']]))
    # error rate
    error_rate_ = np.nan_to_num(np.mean(error))
    group_error_rate_ = []
    index_ = []
    # group_error_rate
    for g0, g1 in groups:
      if g1 is None:
        index_ = (pairs['groups_pos'] == g0)
      else:
        index_ = (pairs['groups_pos'] == g0) & (pairs['groups_neg'] == g1)
      group_error_rate_.append(np.nan_to_num(np.mean(error[index_])))

    # query_error_rate and query ndcg
    query_error_rate = []
    query_ndcg = []
    for query_id in np.unique(dataset.queries.cpu().numpy()):
      # query error rate
      query_index_ = (pairs['queries'] == query_id)
      query_error_rate_ = []
      for g0, g1 in groups:
        if g1 is None:
          index_ = (pairs['groups_pos'] == g0) & query_index_
        else:
          index_ = ((pairs['groups_pos'] == g0) & (pairs['groups_neg'] == g1)
                    & query_index_)
        query_error_rate_.append(np.nan_to_num(np.mean(error[index_])))
      query_error_rate.append(query_error_rate_)

      # ndcg
      query_index_ = (dataset.queries.cpu().numpy() == query_id)
      query_pred_ = preds[query_index_]
      query_labels = dataset.labels.cpu().numpy()[query_index_]
      query_ndcg.append(ndcg(query_labels[np.argsort(-query_pred_)], at=at))

    return error_rate_, np.array(group_error_rate_), np.array(
        query_error_rate), np.array(query_ndcg)

  return error_rate_helper