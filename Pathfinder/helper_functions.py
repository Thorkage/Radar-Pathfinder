import sys
import numpy as np
from numpy.core.fromnumeric import shape

import glob
import os
import pickle
from uwibasspp import rdr
import copy
import warnings
from numba import njit, prange
import json
import io
import yaml

from tqdm import tqdm, trange

import xarray as xr
import pandas as pd
import geopandas as gpd

from scipy import ndimage
from skimage.feature import hessian_matrix, hessian_matrix_eigvals
from scipy.ndimage import gaussian_filter
from scipy.ndimage import convolve

from scipy import signal 
from scipy.io import loadmat
import scipy.io
from scipy.stats import mode
from scipy.stats import linregress
import scipy.fftpack
from scipy.spatial.transform import Rotation as R
from scipy.spatial import cKDTree
from scipy.interpolate import griddata, interp1d

from scipy.signal import argrelextrema
from scipy.signal import find_peaks, peak_prominences
from skimage.measure import regionprops, label

from cartopy import crs as ccrs
from pyproj import Proj, Transformer, CRS
import utm
from mpl_toolkits.basemap import Basemap
from shapely.geometry import Polygon
import cartopy.io.shapereader as shpreader
import shapely.geometry as sgeom
from shapely.geometry import Point
from shapely.prepared import prep
from scipy.signal import savgol_filter
from scipy.constants import speed_of_light

from owslib.wmts import WebMapTileService
from owslib.wms import WebMapService
from owslib.wfs import WebFeatureService
url = 'https://geodata.npolar.no/arcgis/rest/services/Basisdata/NP_Ortofoto_Svalbard_WMTS_3857/MapServer/WMTS/1.0.0/WMTSCapabilities.xml?' # Map
wmts = WebMapTileService(url)
layers_list = list(wmts.contents)
layer = layers_list[0]

from datetime import datetime, timezone
import timeit

from pyrpca import rpca_pcp_ialm

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
import cmcrameri
import cmasher as cmr
import cmocean as cmo

from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.patches import ConnectionPatch
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap, Normalize, TwoSlopeNorm

from mpl_toolkits.mplot3d import Axes3D
mpl.rcParams['axes3d.mouserotationstyle'] = 'azel'  
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

import matplotlib.font_manager as font_manager
font_path= '/Users/torka/Library/Fonts/Helvetica.ttf'
font_manager.fontManager.addfont(font_path)
prop = font_manager.FontProperties(fname=font_path)
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = prop.get_name()
# plt.rcParams["mathtext.fontset"] = 'cm'

from matplotlib.widgets import Button, RadioButtons, CheckButtons
from matplotlib.patches import Circle, Rectangle


# class rdr():
#     def __init__(self):
#         self.rx = None
#         self.fasttime = None
#         self.range_air = None
#         self.slowtime = None

#TODO: create a data delivery function (to .nc)
#! currently implemented in a separate notebook (Notebooks/make_export_netCDF.ipynb)
def make_nc_file(RADAR):
    """
    
    """
    return RADAR


def find_nearest_idx(array, value):
    """
    Finds the index of the element in the input array that is closest to the specified value.
    
    Parameters
    ----------
    array : array-like
        The input array in which to search for the nearest value.
    value : float or int
        The value to find the closest match for in the array.
        
    Returns
    -------
    idx : int
        The index of the element in the array that is closest to the specified value.
    """
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx

def flatten(xss):
    """
    Flattens a list of lists into a single list.
    Args:
        xss (Iterable[Iterable[Any]]): A list (or iterable) of lists (or iterables) to be flattened.
    Returns:
        list: A single list containing all elements from the sublists, in order.
    Example:
        >>> flatten([[1, 2], [3, 4], [5]])
        [1, 2, 3, 4, 5]
    """
    return [x for xs in xss for x in xs]


def remove_altitude_outliers(altitude, diff_threshold=10):
    """
    Removes outlier values from an altitude array based on a specified difference threshold.
    This function identifies and replaces altitude values that differ from their neighbors by more than `diff_threshold` with NaN, 
    then interpolates to fill these gaps. The process repeats until no outliers remain according to the threshold.
    Parameters:
        altitude (array-like): Sequence of altitude values to be filtered.
        diff_threshold (float, optional): Maximum allowed difference between consecutive altitude values. 
                                          Values exceeding this threshold are considered outliers. Default is 10.
    Returns:
        numpy.ndarray: The filtered altitude array with outliers removed and interpolated.
    """
    altitude = np.array(altitude)
    
    differences = np.append(np.diff(altitude), 0)
    while np.any(differences > diff_threshold):
        differences = np.append(np.diff(altitude), 0)
        altitude = np.where(abs(differences) > diff_threshold,  np.nan, altitude)
        altitude = pd.Series(altitude).interpolate(limit_direction='both').values
    
    return altitude

def mask_path_in_cost(cost, path, radius=3, strength=1):
    """
    Masks (increases) the cost values along a specified path in a 2D cost array, with a Gaussian-like spread perpendicular to the path.
    Parameters:
        cost (np.ndarray): 2D array representing the cost map to be modified.
        path (Iterable[Tuple[int, int]]): Sequence of (y, x) coordinates representing the path to be masked.
        radius (int, optional): The radius (in pixels) around each path point to apply the masking effect. Default is 3.
        strength (float, optional): The strength of the masking effect. Higher values increase the cost more. Default is 1.
    Returns:
        np.ndarray: A copy of the input cost array with increased values along and near the specified path.
    """
    masked_cost = cost.copy()
    h, w = cost.shape
    for y, x in path:
        y_min = max(0, y - radius)
        y_max = min(h, y + radius + 1)
        for yy in range(y_min, y_max):
            dist2 = (yy - y)**2
            masked_cost[yy, x] += strength * np.exp(-dist2 / radius)
    return masked_cost

def mask_path_in_cost2(cost, path, radius=3, strength=1):
    """
    Masks (increases) the cost values along a given path in a 2D cost array, within a specified vertical radius.
    
    Parameters
    ----------
    
        cost : np.ndarray
            2D array representing the cost map to be modified.
        path : Iterable[Tuple[int, int]]
            Sequence of (y, x) coordinates representing the path to be masked.
        radius : int, optional
            Vertical radius around each path point within which the cost is increased. Default is 3.
        strength : Union[int, np.ndarray], optional
            Strength of the masking. If an array, should be indexed by x. Default is 1.
            
    Returns
    -------
        masked_cost : np.ndarray
            A copy of the input cost array with increased values along and near the specified path.    
    """
    masked_cost = cost.copy()
    h, w = cost.shape
    for y, x in path:
        y_min = max(0, y - radius)
        y_max = min(h, y + radius + 1)
        for yy in range(y_min, y_max):
            dist2 = (yy - y)**2
            masked_cost[yy, x] += strength[x] * np.exp(-dist2 / radius)
    return masked_cost

def mask_cost_below_above_path(cost, path, strength=1, layer='top'):
    """
    Masks (increases) the cost values in a 2D array either above or below a given path.
    This function takes a 2D cost array and a path (as a list of (y, x) coordinates), and increases the cost values either above ('top') or below ('bottom') each point in the path by a specified strength. This can be useful for discouraging pathfinding algorithms from crossing certain regions relative to a given path.
    Args:
        cost (np.ndarray): 2D array representing the cost map.
        path (Iterable[Tuple[int, int]]): Sequence of (y, x) coordinates representing the path.
        strength (float, optional): Value to add to the masked regions. Default is 1.
        layer (str, optional): Determines which side of the path to mask. 
            'top' increases costs above the path, 'bottom' increases costs below. Default is 'top'.
    Returns:
        np.ndarray: A copy of the cost array with the specified regions masked (increased).
    """
    masked_cost = cost.copy()
    for y, x in path:
        if layer == 'top':
            masked_cost[:y, x] += strength
        if layer == 'bottom':
            masked_cost[y:, x] += strength
    return masked_cost



def flatten_to_interface(arr, RADAR, interface='bottom', smoothing_window=5, reduce=False, top_buffer=None):
    """
    Aligns (flattens) a radargram array to a specified interface (either 'top' or 'bottom') using interface indices
    from a RADAR object. Optionally applies smoothing to the interface, reduces output to valid rows, and adds a buffer of NaNs at the top.

    Parameters
    ----------
    arr : np.ndarray
        2D radargram array to be flattened (shape: [depth, trace]).
    RADAR : object
        Object containing interface arrays (PF_top_interface, PF_bottom_interface).
    interface : str, optional
        Which interface to flatten to: 'top' or 'bottom'. Default is 'bottom'.
    smoothing_window : int, optional
        Window size for smoothing the interface indices. Default is 5.
    reduce : bool, optional
        If True, removes rows that are all NaN after flattening. Default is False.
    top_buffer : int or None, optional
        If set, fills the first `top_buffer` rows with NaN after flattening. Default is None.

    Returns
    -------
    flattened_arr : np.ndarray
        The radargram array aligned to the specified interface.
    """
    
    if interface == 'top':
        interface_indices = np.convolve(RADAR.PF_top_interface, np.ones(smoothing_window)/smoothing_window, mode='same').astype(int)
        sign = -1
        
    elif interface == 'bottom':
        interface_indices = np.convolve(RADAR.PF_bottom_interface, np.ones(smoothing_window)/smoothing_window, mode='same').astype(int)
        sign = -1  # Roll down for bottom interface
        
    else:
        raise ValueError("Interface must be 'top' or 'bottom'")
    
    flattened_arr = np.zeros_like(arr)
    
    for i in range(arr.shape[1]):
        flattened_arr[:, i] = np.roll(arr[:, i], sign * interface_indices[i])
    
    if reduce:
        valid_rows = ~np.all(np.isnan(flattened_arr), axis=1)
        flattened_arr = flattened_arr[valid_rows]
    
    if top_buffer:
        flattened_arr[:top_buffer,:] = np.nan
        
    return flattened_arr

def unwrap_radargram(section, altitude, unambiguous_range, dz):
    """
    Unwraps a radargram image by vertically shifting each column according to the corresponding altitude,
    effectively correcting for topographic variations and aligning subsurface features.
    
    Parameters
    ----------
    section : np.ndarray
        2D array (H, W) representing the radargram, where H is the number of depth samples and W is the number of columns (traces).
    altitude : np.ndarray
        1D array of length W containing the altitude (in meters) for each column of the radargram.
    unambiguous_range : float
        The unambiguous range (in meters) of the radar system, used to wrap the altitude values.
    dz : float
        The vertical sampling interval (in meters) per pixel in the radargram.
        
    Returns
    -------
    im_unwrapped : np.ndarray
        The unwrapped radargram image with corrected vertical alignment.
    y_axis : np.ndarray
        1D array representing the depth axis (in meters) for the unwrapped image.
    shifts : list of int
        List of vertical shift values (in pixels) applied to each column.
    """
        
    H, W = section.shape
    R = unambiguous_range
    max_depth = np.max(altitude)
    # H_new = int(np.ceil((max_depth + R) / dz)) + H  # buffer for overflow
    H_new = int(((max_depth) / dz)) + H

    im_unwrapped = np.empty((H_new, W), dtype=section.dtype)
    shifts = []
    for x in range(W):
        true_depth = altitude[x]
        within_range_depth = true_depth % R
        roll_pixels = int(round(-within_range_depth / dz))

        col = np.roll(section[:, x], roll_pixels)

        y_shift = int(round(true_depth / dz))
        y_end = y_shift + H
        if y_end <= H_new:
            im_unwrapped[y_shift:y_end, x] = col
        else:
            im_unwrapped[y_shift:H_new, x] = col[:H_new - y_shift]
            
        shifts.append(y_shift)

    y_axis = np.arange(H_new) * dz
    
    return im_unwrapped, y_axis, shifts




# def flatten_to_altitude(RADAR, section):
#     """
    
#     """
#     # Compute vertical spacing (delta_z) in section (in same units as uwibass.alt)
#     delta_z = RADAR.range_air[1]  # assumes range_snow is in same units as alt

#     # Reference altitude (e.g., median or first value)
#     ref_alt = np.nanmedian(RADAR.GPS_Alt)

#     # Compute shift (in pixels) for each column to align to ref_alt
#     shifts = np.round((RADAR.GPS_Alt - ref_alt) / delta_z).astype(int)

#     # Create a copy of section to flatten
#     flattened_section = np.zeros_like(section)
#     for i in range(section.shape[1]):
#         flattened_section[:, i] = np.roll(section[:, i], -shifts[i])

#     return flattened_section
    
    
    
def mag2db(magnitude):
        magnitude = np.maximum(magnitude, 1e-10)  # Prevent log(0) or negative values
        return 20 * np.log10(magnitude)

def NormalizeData(data):
    return (data - np.nanmin(data)) / (np.nanmax(data) - np.nanmin(data))

def movingAvg(my_vector,KERNEL_SIZE):
    kernel = np.ones(KERNEL_SIZE) / KERNEL_SIZE
    output = np.convolve(my_vector, kernel, mode='same')
    return output

def scale(input, min, max):
    input += -(np.min(input))
    input /= np.max(input) / (max - min)
    input += min
    return input

def Quickboost(im, degree = 1):
    minval = np.nanpercentile(im, degree)
    maxval = np.nanpercentile(im, 100-degree)
    im = np.clip(im, minval, maxval)
    im = ((im - minval) / (maxval - minval)) 
    return im

def interpolate_zeros(array):
    out = array
    nonzeros = [(k, x , 0-k) for k, x in enumerate(out) if x != 0]

    for i in np.arange(len(array)):
        if out[i] == 0:
            # sorted_nonzeros = 
            out[i] = sorted(nonzeros, key=lambda x: abs(x[0] - i))[0][1]
        
    return out

def interpolateImx(x,im):
# %Interpolate the x axis of an image based on a non-linear vector x
# % x is the non-linear vector. Now. also works iwth 1d arrays!

    imout = np.copy(im)*0
    xout = np.linspace(np.min(x), np.max(x), len(x))
    if im.ndim ==1:
        # print('1D array')
        imout = np.interp(xout, x, im)

    if im.ndim ==2:
        # print('2D array')
        for i in np.arange(np.size(im,0)):
            imout[i,:] = np.interp(xout, x, im[i,:])

    return imout,xout

def mscatter(x,y,ax=None, m=None, **kw):
    
    import matplotlib.markers as mmarkers
    if not ax: ax=plt.gca()
    sc = ax.scatter(x,y,**kw)
    if (m is not None) and (len(m)==len(x)):
        paths = []
        for marker in m:
            marker =str(marker)
            if isinstance(marker, mmarkers.MarkerStyle):
                marker_obj = marker
            else:
                marker_obj = mmarkers.MarkerStyle(marker)
            path = marker_obj.get_path().transformed(
                        marker_obj.get_transform())
            paths.append(path)
        sc.set_paths(paths)
    return sc
# def get_gridvalues_at_points(
#     x, y,
#     grid_file: str= '/Users/torka/Downloads/DTU21MSS_1min_TP.nc'
# ) -> np.ndarray:
    
#     grid = xr.open_dataset(grid_file)

#     # trying to build a generalized function, but screw that
#     bbox = np.array([np.min(x), np.min(y), np.max(x), np.max(y)])

#     min_x = np.min(bbox[[0,2]]) - 100
#     max_x = np.max(bbox[[0,2]]) + 100
#     min_y = np.min(bbox[[1,3]]) - 100
#     max_y = np.max(bbox[[1,3]]) + 100

#     # Open MSS dataset
#     # Subset MSS grid to lat > 60 for efficiency
#     grid = grid.sel(x=slice(int(np.floor(min_x)), int(np.ceil(max_x))),
#                 y=slice(int(max_y ), int(min_y ))
#                 )

#     # Convert MSS lat/lon grid to x/y using transformer
#     x_grid = grid['x'].values
#     y_grid = grid['y'].values
#     X_grid, Y_grid = np.meshgrid(x_grid, y_grid)
#     grid_grid = grid['band_data'].values

#     # Prepare points to query
#     x_pts = x
#     y_pts = y

#     # Use griddata for fast linear interpolation
#     points = np.column_stack([X_grid.ravel(), Y_grid.ravel()])
#     values = grid_grid.ravel()
#     pts_query = np.column_stack([x_pts, y_pts])

#     grid_interp = griddata(points, values, pts_query, method='linear')

#     return grid_interp#, x_grid, y_grid, grid_grid
