import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import glob
from tqdm import tqdm
import pandas as pd
from shapely.wkt import loads
import os
import numpy as np
import tifffile as tiff
import cv2
import bcolz
from tensorboardX import SummaryWriter
import time
import numbers
import random
import shutil
from collections import defaultdict
from shapely.wkt import dumps
from shapely.geometry import MultiPolygon, Polygon
from shapely.affinity import scale
from PIL import Image
import pdb
import attr
import cv2
from pprint import pprint
from visdom import Visdom
import shutil
import attr
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("white")

class_list = ["Buildings", "Misc. Manmade structures", "Road", "Track", "Trees", "Crops", "Waterway",
              "Standing Water", "Vehicle Large", "Vehicle Small"]

normalize = transforms.Normalize(mean=[414.67273, 463.15457, 326.54767, 496.43326, 280.42242, 326.77518,
                                       463.41257, 454.945, 414.73422, 500.08765, 720.867, 497.47687],
                                 std=[63.99271, 51.419453, 33.771034, 59.144787, 25.715364, 33.685066,
                                      50.723106, 62.862915, 62.952812, 65.08902, 88.63291 , 69.42058])


def save_array(fname, arr):
    arr = bcolz.carray(arr, mode="w", rootdir=fname)
    arr.flush()

def load_array(fname):
    return bcolz.open(fname, mode="r")[:]

def create_dir(path):
    if not os.path.exists(path): os.makedirs(path)


DF = pd.read_csv('data/train_wkt_v4.csv')
GS = pd.read_csv('data/grid_sizes.csv', names=['ImageId', 'Xmax', 'Ymin'], skiprows=1)
SB = pd.read_csv('data/sample_submission.csv')


@attr.s
class HyperParams(object):
    classes = attr.ib(default=list(range(10)))
    net = attr.ib(default="UNet2")
    n_channels = attr.ib(default=12)

    augment_rotations = attr.ib(default=0.)
    augment_flips = attr.ib(default=0)

    bn = attr.ib(default=1)
    filters_base = attr.ib(default=32)

    n_epochs = attr.ib(default=1000)
    oversample = attr.ib(default=0.0)

    opt = attr.ib(default="sgd")
    lr = attr.ib(default=0.01)
    patience = attr.ib(default=2)
    weight_decay = attr.ib(default=0.0)
    batch_size = attr.ib(default=256)
    log_loss_weight = attr.ib(default=0.1)
    dice_loss_weight = attr.ib(default=0.9)

    print_freq = attr.ib(default=100)
    num_gpu = attr.ib(default=1)
    crop_size = attr.ib(default=80)
    samples_per_epoch = attr.ib(default=20000)
    num_workers = attr.ib(default=4)

    @property
    def n_classes(self):
        return len(self.classes)

    def update(self, hps_string):
        if hps_string:
            values = dict(pair.split("=") for pair in hps_string.split(","))
            for field in attr.fields(HyperParams):
                v = values.pop(field.name, None)
                if v is not None:
                    default = field.default
                    assert not isinstance(default, bool)
                    if isinstance(default, (int, float, str)):
                        v = type(default)(v)
                    elif isinstance(default, list):
                        v = [type(default[0])(x) for x in v.split("-")]
                    setattr(self, field.name, v)
            if values:
                raise ValueError("Unknown hyperparams: {}".format(values))


def masktensor2image(tensor):
    # pdb.set_trace()
    image = tensor.cpu().float().numpy()*255
    if image.shape[0] == 1:
        image = np.tile(image, (3,1,1))
    image = image[:, ::10, ::10]
    return image.astype(np.uint8)


class Logger():
    def __init__(self, env):
        self.viz = Visdom(env=env)
        self.losses = {}
        self.loss_windows = {}
        self.image_windows = {}


    def log(self, epoch=None, losses=None, images=None, image_grid=None, env=None):
        # Draw images
        if images:
            for image_name, tensor in images.items():
                # pdb.set_trace()
                if image_name not in self.image_windows:
                    self.image_windows[image_name] = self.viz.image(masktensor2image(tensor.data), opts={'title':image_name})
                else:
                    self.viz.image(masktensor2image(tensor.data), win=self.image_windows[image_name],
                                       opts={'title':image_name})
        if image_grid:
            for image_name, tensor in image_grid.items():
                if image_name not in self.image_windows:
                    self.image_windows[image_name] = self.viz.images(tensor, env=env,
                                                                    opts={'title': image_name})
                else:
                    self.viz.images(tensor, win=self.image_windows[image_name], env=env,
                                   opts={'title': image_name})

        # Plot losses
        if losses:
            for loss_name, loss in losses.items():
                if loss_name not in self.loss_windows:
                    self.loss_windows[loss_name] = self.viz.line(X=np.array([epoch]), Y=np.array([loss]),
                                                                    opts={'xlabel': 'epochs', 'ylabel': loss_name, 'title': loss_name})
                else:
                    self.viz.line(X=np.array([epoch]), Y=np.array([loss]), win=self.loss_windows[loss_name], update='append')


def _convert_coordinates_to_raster(coords, img_size, xymax):
    """
    Resize the polygon coordinates to the specific resolution of an image.
    """
    Xmax, Ymax = xymax
    H, W = img_size
    W1 = 1.0 * W * W / (W + 1)
    H1 = 1.0 * H * H / (H + 1)
    xf = W1 / Xmax
    yf = H1 / Ymax
    coords[:, 1] *= yf
    coords[:, 0] *= xf
    coords_int = np.round(coords).astype(np.int32)
    return coords_int


def _get_xmax_ymin(grid_sizes_panda, imageId):
    """
    To resize the training polygons, we need these parameters for each image.
    """
    xmax, ymin = grid_sizes_panda[grid_sizes_panda.ImageId == imageId].iloc[0, 1:].astype(float)
    return (xmax, ymin)


def _get_polygon_list(wkt_list_pandas, imageId, class_type):
    """
    Load the training polygons with shapely.
    """
#     pdb.set_trace()
    df_image = wkt_list_pandas[wkt_list_pandas.ImageId == imageId]
    multipoly_def = df_image[df_image.ClassType == class_type].MultipolygonWKT
    polygonList = None
    if len(multipoly_def) > 0:
        assert len(multipoly_def) == 1
        polygonList = loads(multipoly_def.values[0])
    return polygonList


def _get_and_convert_contours(polygonList, raster_img_size, xymax):
    """
    Create lists of exterior and interior coordinates of polygons resized to a specific image resolution.
    """
    perim_list = []
    interior_list = []
    if polygonList is None:
        return None
    for k in range(len(polygonList)):
        poly = polygonList[k]
        perim = np.array(list(poly.exterior.coords))
        perim_c = _convert_coordinates_to_raster(perim, raster_img_size, xymax)
        perim_list.append(perim_c)
        for pi in poly.interiors:
            interior = np.array(list(pi.coords))
            interior_c = _convert_coordinates_to_raster(interior, raster_img_size, xymax)
            interior_list.append(interior_c)
    return perim_list, interior_list


def _plot_mask_from_contours(raster_img_size, contours, class_value=1):
    """
    Creates a class mask (0 and 1s) from lists of exterior and interior polygon coordinates.
    """
    img_mask = np.zeros(raster_img_size, np.uint8)
    if contours is None:
        return img_mask
    perim_list, interior_list = contours
    cv2.fillPoly(img_mask, perim_list, class_value)
    cv2.fillPoly(img_mask, interior_list, 0)
    return img_mask


def generate_mask_for_image_and_class(raster_size, imageId, class_type, grid_sizes_panda=GS, wkt_list_pandas=DF):
    """
    Outputs a specific class mask from the training images.
    """
    xymax = _get_xmax_ymin(grid_sizes_panda, imageId)
#     pdb.set_trace()
    polygon_list = _get_polygon_list(wkt_list_pandas, imageId, class_type)
    contours = _get_and_convert_contours(polygon_list, raster_size, xymax)
    mask = _plot_mask_from_contours(raster_size, contours, 1)
    return mask


def get_polygon_list_from_subm(multipoly_def):
    """
    Load the training polygons with shapely.
    """
    polygonList = None
    if len(multipoly_def) > 0:
        assert len(multipoly_def) == 1
        polygonList = loads(multipoly_def.values[0])
    return polygonList


def generate_mask_for_polygon_list(polygon_list, raster_size, imageId, class_type, grid_sizes_panda=GS):
    """
    Outputs a specific class mask from the training images.
    """
    xymax = _get_xmax_ymin(grid_sizes_panda, imageId)
    contours = _get_and_convert_contours(polygon_list, raster_size, xymax)
    mask = _plot_mask_from_contours(raster_size, contours, 1)
    return mask


def M(image_id, dims=12):
    """
    Loads the tiff-files with different number of bands.
    """
    if dims==12:
        img_RGB = np.transpose(tiff.imread("data/three_band/{}.tif".format(image_id)), (1,2,0))
        y, x = img_RGB.shape[:2]
        img_M = np.transpose(tiff.imread("data/sixteen_band/{}_M.tif".format(image_id)), (1,2,0))
        img_M = cv2.resize(img_M, (x, y))
        img_P = tiff.imread("data/sixteen_band/{}_P.tif".format(image_id))
        img_P = cv2.resize(img_P, (x, y))

        img = np.zeros((img_RGB.shape[0], img_RGB.shape[1], dims), "float32")
        img[..., 0:3] = img_RGB
        img[..., 3] = img_P
        img[..., 4:12] = img_M
    return img


def stretch_8bit(bands, lower_percent=2, higher_percent=98):
    """
    Cuts of extreme values of spectral bands to visualize them better for the human eye.
    """
    out = np.zeros_like(bands)
    for i in range(3):
        a = 0
        b = 255
        c = np.percentile(bands[:,:,i], lower_percent)
        d = np.percentile(bands[:,:,i], higher_percent)
        t = a + (bands[:,:,i] - c) * (b - a) / (d - c)
        t[t<a] = a
        t[t>b] = b
        out[:,:,i] =t
    return out.astype(np.uint8)


def mask_to_polygons(mask, epsilon=1, min_area=1.):
    """
    Create a Multipolygon from a mask of 0-1 pixels.
    """
    # find contours of mask of pixels
    image, contours, hierarchy = cv2.findContours(
        ((mask == 1) * 255).astype(np.uint8),
        cv2.RETR_CCOMP, cv2.CHAIN_APPROX_TC89_KCOS)
    # create approximate contours to have reasonable submission size
    approx_contours = [cv2.approxPolyDP(cnt, epsilon, True)
                       for cnt in contours]
    if not contours:
        return MultiPolygon()
    # now messy stuff to associate parent and child contours
    cnt_children = defaultdict(list)
    child_contours = set()
    assert hierarchy.shape[0] == 1
    # http://docs.opencv.org/3.1.0/d9/d8b/tutorial_py_contours_hierarchy.html
    for idx, (_, _, _, parent_idx) in enumerate(hierarchy[0]):
        if parent_idx != -1:
            child_contours.add(idx)
            cnt_children[parent_idx].append(approx_contours[idx])

    # create actual polygons filtering by area (removes artifacts)
    all_polygons = []
    for idx, cnt in enumerate(approx_contours):
        if idx not in child_contours and cv2.contourArea(cnt) >= min_area:
            assert cnt.shape[1] == 1
            poly = Polygon(
                shell=cnt[:, 0, :],
                holes=[c[:, 0, :] for c in cnt_children.get(idx, [])
                       if cv2.contourArea(c) >= min_area])
            all_polygons.append(poly)
    # approximating polygons might have created invalid ones, fix them
    all_polygons = MultiPolygon(all_polygons)
    if not all_polygons.is_valid:
        all_polygons = all_polygons.buffer(0)
        # Sometimes buffer() converts a simple Multipolygon to just a Polygon,
        # need to keep it a Multi throughout
        if all_polygons.type == 'Polygon':
            all_polygons = MultiPolygon([all_polygons])
    return all_polygons

def get_scalers(im_size, x_max, y_min):
    h, w = im_size  # they are flipped so that mask_for_polygons works correctly
    h, w = float(h), float(w)
    w_ = 1.0 * w * (w / (w + 1))
    h_ = 1.0 * h * (h / (h + 1))
    return w_ / x_max, h_ / y_min
        
        