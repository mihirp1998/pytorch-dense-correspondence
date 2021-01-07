import sys
import torch
import os
import matplotlib.pyplot as plt
import cv2
import numpy as np
import ipdb
st = ipdb.set_trace
import copy


def compute_gaussian_kernel_heatmap_from_norm_diffs(norm_diffs, variance):
    """
    Computes and RGB heatmap from norm diffs
    :param norm_diffs: distances in descriptor space to a given keypoint
    :type norm_diffs: numpy array of shape [H,W]
    :param variance: the variance of the kernel
    :type variance:
    :return: RGB image [H,W,3]
    :rtype:
    """

    """
    Computes an RGB heatmap from the norm_diffs
    :param norm_diffs:
    :type norm_diffs:
    :return:
    :rtype:
    """

    heatmap = np.copy(norm_diffs)

    heatmap = np.exp(-heatmap / variance)  # these are now in [0,1]
    heatmap *= 255
    heatmap = heatmap.astype(np.uint8)
    heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    return heatmap_color

def compute_heatmap_from_norm_diffs(norm_diffs, variance):
    """
    Computes and RGB heatmap from norm diffs
    :param norm_diffs: distances in descriptor space to a given keypoint
    :type norm_diffs: numpy array of shape [H,W]
    :param variance: the variance of the kernel
    :type variance:
    :return: RGB image [H,W,3]
    :rtype:
    """

    """
    Computes an RGB heatmap from the norm_diffs
    :param norm_diffs:
    :type norm_diffs:
    :return:
    :rtype:
    """

    heatmap = np.copy(norm_diffs)
    scaled_heatmap = (-heatmap)/np.max(heatmap)
    scaled_heatmap = scaled_heatmap + 1.0
    # st()
    # heatmap = np.exp(-heatmap / variance)  # these are now in [0,1]
    heatmap *= 255
    heatmap = heatmap.astype(np.uint8)
    heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    # heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_RGB2BGR)
    return heatmap_color

def grayscale_to_heatmap(name, grayscale, min_val=0.0, max_val=13.0, scheme='jet', normalize_using_data_stats = False, only_return = False):
    # grayscale is BxHxW
    grayscale = torch.from_numpy(grayscale)
    grayscale = grayscale
    # cmap = maplotlib.cm.get_cmap(cmap)
    cmap = plt.get_cmap(scheme)
    # grayscale = (grayscale-grayscale.min())/(grayscale.max()-grayscale.min())
    if normalize_using_data_stats:
        grayscale = (grayscale-grayscale.min())/(grayscale.max()-grayscale.min())
    else:
        grayscale = (torch.clamp(grayscale, min_val, max_val)-min_val)/(max_val-min_val)
    grayscale = grayscale.cpu().numpy()
    rgba_img = cmap(grayscale)
    rgb_img = np.delete(rgba_img, 3, 2)
    rgb_img = torch.from_numpy(rgb_img).permute(2,0,1)*255
    rgb_img = rgb_img.to(torch.int)
    # out = preprocess_color(rgb_img)
    if not only_return:
        self.summ_rgb(name, out)
    out = rgb_img.permute(1,2,0).numpy()
    out = out.astype(np.uint8)
    out = np.ascontiguousarray(out, dtype=np.uint8)
    # st()
    return out

def preprocess_color(x):
    if type(x).__module__ == np.__name__:
        return x.astype(np.float32) * 1./255 - 0.5
    else:
        return x.float() * 1./255 - 0.5


def draw_reticle(img, u, v, label_color):
    """
    Draws a reticle on the image at the given (u,v) position

    :param img:
    :type img:
    :param u:
    :type u:
    :param v:
    :type v:
    :param label_color:
    :type label_color:
    :return:
    :rtype:
    """
    white = (255, 255, 255)
    cv2.circle(img, (u, v), 10, label_color, 1)
    cv2.circle(img, (u, v), 11, white, 1)
    cv2.circle(img, (u, v), 12, label_color, 1)
    cv2.line(img, (u, v + 1), (u, v + 3), white, 1)
    cv2.line(img, (u + 1, v), (u + 3, v), white, 1)
    cv2.line(img, (u, v - 1), (u, v - 3), white, 1)
    cv2.line(img, (u - 1, v), (u - 3, v), white, 1)

