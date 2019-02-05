#!/usr/bin/python

"""
Martin Kersner, m.kersner@gmail.com
2015/11/30

Evaluation metrics for image segmentation inspired by
paper Fully Convolutional Networks for Semantic Segmentation.
"""

from __future__ import division
import re
import numpy as np


def pixel_accuracy(eval_segm, gt_segm):
    """
    sum_i(n_ii) / sum_i(t_i)
    """

    check_size(eval_segm, gt_segm)
    cl, n_cl = extract_classes(gt_segm)
    eval_mask, gt_mask = extract_both_masks(eval_segm, gt_segm, cl, n_cl)
    sum_n_ii = 0
    sum_t_i = 0
    for i, c in enumerate(cl):
        curr_eval_mask = eval_mask[i, :, :]
        curr_gt_mask = gt_mask[i, :, :]
        sum_n_ii += np.sum(np.logical_and(curr_eval_mask, curr_gt_mask))
        sum_t_i += np.sum(curr_gt_mask)
    if sum_t_i == 0:
        res_pixel_accuracy = 0
    else:
        res_pixel_accuracy = sum_n_ii / sum_t_i
    return res_pixel_accuracy


def mean_accuracy(eval_segm, gt_segm):
    """
    (1/n_cl) sum_i(n_ii/t_i)
    """

    check_size(eval_segm, gt_segm)
    cl, n_cl = extract_classes(gt_segm)
    eval_mask, gt_mask = extract_both_masks(eval_segm, gt_segm, cl, n_cl)
    accuracy = list([0]) * n_cl
    for i, c in enumerate(cl):
        curr_eval_mask = eval_mask[i, :, :]
        curr_gt_mask = gt_mask[i, :, :]
        n_ii = np.sum(np.logical_and(curr_eval_mask, curr_gt_mask))
        t_i = np.sum(curr_gt_mask)
        if t_i != 0:
            accuracy[i] = n_ii / t_i
    res_mean_accuracy = np.mean(accuracy)
    return res_mean_accuracy


def mean_precision(eval_segm, gt_segm):
    # Calculate Mean Precision
    check_size(eval_segm, gt_segm)
    cl, n_cl = extract_classes(gt_segm)
    eval_mask, gt_mask = extract_both_masks(eval_segm, gt_segm, cl, n_cl)
    precision = list([0]) * n_cl
    for i, c in enumerate(cl):
        curr_eval_mask = eval_mask[i, :, :]
        curr_gt_mask = gt_mask[i, :, :]
        n_ii = np.sum(np.logical_and(curr_eval_mask, curr_gt_mask))
        t_i = np.sum(curr_eval_mask)
        if t_i != 0:
            precision[i] = n_ii / t_i
    res_mean_precision = np.mean(precision)
    return res_mean_precision


def get_num_true_positives(eval_segm, gt_segm):
    # Calculate Mean Precision
    check_size(eval_segm, gt_segm)
    cl, n_cl = extract_classes(gt_segm)
    eval_mask, gt_mask = extract_both_masks(eval_segm, gt_segm, cl, n_cl)
    num_true_positives = 0
    for i, c in enumerate(cl):
        curr_eval_mask = eval_mask[i, :, :]
        curr_gt_mask = gt_mask[i, :, :]
        num_true_positives += np.sum(np.logical_and(curr_eval_mask, curr_gt_mask))

    return num_true_positives


def get_num_false_positives(eval_segm, gt_segm):
    # Calculate Mean Precision
    check_size(eval_segm, gt_segm)
    cl, n_cl = extract_classes(gt_segm)
    eval_mask, gt_mask = extract_both_masks(eval_segm, gt_segm, cl, n_cl)
    num_true_positives = 0
    for i, c in enumerate(cl):
        curr_eval_mask = eval_mask[i, :, :]
        curr_gt_mask = gt_mask[i, :, :]
        num_true_positives += np.sum(np.logical_and(curr_eval_mask, np.logical_not(curr_gt_mask)))

    return num_true_positives


def mean_IU(eval_segm, gt_segm):
    """
    (1/n_cl) * sum_i(n_ii / (t_i + sum_j(n_ji) - n_ii))
    """

    check_size(eval_segm, gt_segm)
    cl, n_cl = union_classes(eval_segm, gt_segm)
    _, n_cl_gt = extract_classes(gt_segm)
    eval_mask, gt_mask = extract_both_masks(eval_segm, gt_segm, cl, n_cl)
    iu = list([0]) * n_cl
    for i, c in enumerate(cl):
        curr_eval_mask = eval_mask[i, :, :]
        curr_gt_mask = gt_mask[i, :, :]
        if (np.sum(curr_eval_mask) == 0) or (np.sum(curr_gt_mask) == 0):
            continue
        n_ii = np.sum(np.logical_and(curr_eval_mask, curr_gt_mask))
        t_i = np.sum(curr_gt_mask)
        n_ij = np.sum(curr_eval_mask)
        iu[i] = n_ii / (t_i + n_ij - n_ii)
    res_mean_iu = np.sum(iu) / n_cl_gt
    return res_mean_iu, iu


def frequency_weighted_IU(eval_segm, gt_segm):
    """
    sum_k(t_k)^(-1) * sum_i((t_i*n_ii)/(t_i + sum_j(n_ji) - n_ii))
    """

    check_size(eval_segm, gt_segm)
    cl, n_cl = union_classes(eval_segm, gt_segm)
    eval_mask, gt_mask = extract_both_masks(eval_segm, gt_segm, cl, n_cl)
    res_frequency_weighted_iu_ = list([0]) * n_cl
    for i, c in enumerate(cl):
        curr_eval_mask = eval_mask[i, :, :]
        curr_gt_mask = gt_mask[i, :, :]
        if (np.sum(curr_eval_mask) == 0) or (np.sum(curr_gt_mask) == 0):
            continue
        n_ii = np.sum(np.logical_and(curr_eval_mask, curr_gt_mask))
        t_i = np.sum(curr_gt_mask)
        n_ij = np.sum(curr_eval_mask)
        res_frequency_weighted_iu_[i] = (t_i * n_ii) / (t_i + n_ij - n_ii)
    sum_k_t_k = get_pixel_area(eval_segm)
    res_frequency_weighted_iu = np.sum(res_frequency_weighted_iu_) / sum_k_t_k
    return res_frequency_weighted_iu


def tryInteger(text):
    # Check String to Integer Parsing
    try:
        return int(text)
    except:
        return text


def alphanumericSort(text):
    # Sort String alphanumerically
    return [tryInteger(c) for c in re.split('([0-9]+)', text)]

"""
Auxiliary functions used during evaluation.
"""
def get_pixel_area(segm):
    return segm.shape[0] * segm.shape[1]


def extract_both_masks(eval_segm, gt_segm, cl, n_cl):
    eval_mask = extract_masks(eval_segm, cl, n_cl)
    gt_mask = extract_masks(gt_segm, cl, n_cl)
    return eval_mask, gt_mask


def extract_classes(segm):
    cl = np.unique(segm)
    n_cl = len(cl)
    return cl, n_cl


def union_classes(eval_segm, gt_segm):
    eval_cl, _ = extract_classes(eval_segm)
    gt_cl, _ = extract_classes(gt_segm)
    cl = np.union1d(eval_cl, gt_cl)
    n_cl = len(cl)
    return cl, n_cl


def extract_masks(segm, cl, n_cl):
    h, w = segm_size(segm)
    masks = np.zeros((n_cl, h, w))
    for i, c in enumerate(cl):
        masks[i, :, :] = segm == c
    return masks


def segm_size(segm):
    try:
        height = segm.shape[0]
        width = segm.shape[1]
    except IndexError:
        raise
    return height, width


def check_size(eval_segm, gt_segm):
    h_e, w_e = segm_size(eval_segm)
    h_g, w_g = segm_size(gt_segm)
    if (h_e != h_g) or (w_e != w_g):
        raise EvalSegErr("DiffDim: Different dimensions of matrices!")


# Class
class EvalSegErr(Exception):
    def __init__(self, value):
        self.value = value

    def __str__(self):
        return repr(self.value)
