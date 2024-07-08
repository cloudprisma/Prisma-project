#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pyvsnr import VSNR
import numpy as np

from skimage import io, img_as_float
from skimage.restoration import denoise_nl_means, estimate_sigma
import scipy as sp
import scipy.linalg as linalg

import h5py
import os
import shutil

# =============================================================================
# 1. Read data
# =============================================================================

def read_data(filename,spectral_region,sensor):
    """
    Goals: open-read and scale the data cube
    Parameters
    ----------
        filename: path and name of the dataset
        sensor: choose between 'HCO','HRC'
        spectral_region: VNIR - SWIR - PAN
        
    Returns-> np array with the scaled data cube
    """
     
    spectral_region = spectral_region.upper()
    sensor = sensor.upper()
     
    #print(layer,sensor) 
    with h5py.File(filename,  mode='r') as hf:
        
        sds = f'HDFEOS/SWATHS/PRS_L1_{sensor}/Data Fields/{spectral_region}_Cube'
        #print(f'ok1 sds {sds}')
        ds = hf[sds][:].astype(np.single) # single or Float32
        #print('ok2')       
        # Read attributes
        scale = hf.attrs[f'ScaleFactor_{spectral_region.capitalize()}'] 
        #print('ok3')
        offset = hf.attrs[f'Offset_{spectral_region.capitalize()}']
        #print('ok4')
    
    ds = ds/scale - offset
    #print('ok5') #
    ds = np.transpose(ds, axes=[2,0,1]) # data rearranged as (lines,samples,bands) ######
    #print('ok6')
    return ds

# =============================================================================
# 2. Destriping data
# =============================================================================

def destripe(band, alpha=1e-2, name='gabor', sigma=(1, 30), theta=20, maxit=100, thr=1e-4):
    """
    Goals: apply VSNR algorithm originaly coded in MATLAB - see the Pierre Weiss
    https://www.math.univ-toulouse.fr/~weiss/PageCodes.html
    
    Parameters
    ----------
    alpha: float
        vsnr parameter associated to the filter
    name: str, optional
        Filter type to choose among Gabor filter and horizontal or vertical
        Dirac filter : 'gabor', 'dirac_h', 'dirac_v' respectively
    sigma: tuple of 2 floats
        Standard deviations of the Gaussian envelope in the x and y directions
        respectively
    theta: float, optional
        Rotation angle (in clockwise) to apply [in degrees]
    maxit: int, optional
        Maximum iterations of the iterative processing
    thr - (cvg_threshold): float, optional
        Convergence criteria value used to stop the iterative processing
    gabor: Gabor filter in physical space (real part) defined on a :math:`3-\sigma`
        support
        
    Returns-> corrected info an convergence criterias...
    """
       
    # vsnr object creation
    vsnr = VSNR(band.shape)

    # add filter (at least one!)
    vsnr.add_filter(alpha=alpha, name=name, sigma=sigma, theta=theta)
    
    # vsnr initialization
    vsnr.initialize()

    # image processing
    img_corr = vsnr.eval(band, maxit=maxit, cvg_threshold=thr)
    
    # Note: If I want some additional information (i.e. lambda) 
    # I would ask in vsnr.__dict__ / vsnr = VSNR((10,10)) 
    # dic_vsnr = vsnr.__dict__   or vsnr.__dict__['var'] or vsnr.varname
    # return img_corr, vsnr.cvg_criterias, dic_vsnr 

    return img_corr, vsnr.cvg_criterias 
    
# =============================================================================
# 3. Estimating Noise Level
# =============================================================================

def est_additive_noise(data):  
    
    '''
    est_additive_noise: hyperspectral noise estimation. This function infers the noise in a 
    hyperspectral data set, by assuming that the reflectance at a given band is well modelled 
    by a linear regression on the remaining bands.
    parameters. Code Base on Nascimento & Bioucas-Dias 
    available at http://www.lx.it.pt/~bioucas/code.htm under a non-specified Copyright (c)       
    Parameters
    ----------

    data: is LxN numpy matrix with the hyperspectral data set 
            where L is the number of bands 
            and N is the number of pixels
    noise_type: additive or poisson
    verbose: [optional] on or off

    Returns-> 
    w is the noise estimates for every pixel (LxN)
    Rw is the noise correlation matrix estimates (LxL)

    '''   
    
    if data.shape[0] < 2: print('Too few bands to estimate the noise')
       
    N,L = data.shape
    small = 1e-6
    RR = np.dot(np.transpose(data), data)  # equation (11)
    RRi = np.linalg.inv(RR+small*np.eye(L))  # equation (11)  
    w = np.zeros((N,L)) 
    print('Process: computing the sample correlation matrix and its inverse\n')  
    for i in range(L):
        
        # equation (14)  
        XX = RRi - np.dot(RRi[:,i].reshape(-1,1),RRi[i,:].reshape(1,-1))/RRi[i,i]
        #print(f'XX: {XX}')
        
        RRa = RR[:,i].reshape(-1,1)
        RRa[i] = 0  # this remove the effects of XX[:,i]
        #print(f'RRa: {RRa}')
        
        # equation (9)
        beta = np.dot(XX, RRa)
        beta[i] = 0 # this remove the effects of XX(i,:) 
        
        #print(f'Beta: {beta}')
        
        # equation (10) w(i,:) = r(i,:) - beta'*r - reverse: np.transpose(data)
        w[:,i] = data[:,i] - np.dot(np.transpose(beta),np.transpose(data))  # note that beta(i)=0 => beta(i)*r(i,:)=0 
        #print(f'w: {w}')
        
    print('... Computing noise correlation matrix\n')
    Rw = np.diag(np.diag(np.dot(np.transpose(w),w)/N))  
    Rw2 = np.mean(Rw[Rw > 0])
    
      
    print('Process: Ok!')
        
    return w, Rw, Rw2

def est_poisson_noise(data):
    '''
    est_poisson_noise: hyperspectral noise estimation. This function infers the noise in a 
    hyperspectral data set, by assuming that the reflectance at a given band is well modelled 
    by a linear regression on the remaining bands.
    parameters. Code Base on Nascimento & Bioucas-Dias 
    available at http://www.lx.it.pt/~bioucas/code.htm under a non-specified Copyright (c)       
    Parameters
    ----------
    data: is LxN numpy matrix with the hyperspectral data set 
    where L is the number of bands 
    and N is the number of pixels
    Returns-> 
    w is the noise estimates for every pixel (LxN)
    Rw is the noise correlation matrix estimates (LxL)
    '''
    
    if data.shape[0] < 2: print('Too few bands to estimate the noise')
    
    N,L = data.shape
    w = np.zeros((N,L)) 
    sq_data = np.sqrt(data*(data>0))  # prevent negative values
    #print(f'sq_data: {sq_data}')
    
    u, _, _ = est_additive_noise(sq_data)  # noise estimates 
    #print(f'u: {u}')
    
    x = (sq_data - u)**2  # signal estimates   
    #print(f'x: {x}')
    
    w = np.sqrt(x)*u*2 
    #print(f'w: {w}')
    
    Rw = np.dot(np.transpose(w),w)/N
    #print(f'Rw: {Rw}')
    
    Rw2 = np.mean(Rw[Rw > 0]) 
    #print(f'Rw2: {Rw2}')
    
    return w, Rw,Rw2

# =============================================================================
# 4. Denoise by non local means
# =============================================================================
# Perform non-local means denoising on 2D-4D grayscale or RGB images
# https://scikit-image.org/docs/stable/api/skimage.restoration.html#skimage.restoration.denoise_nl_means
# see main.py


# =============================================================================
# 5. Saving data 
# =============================================================================


def save_copy_he5(ref_file, array, method, sensor, spectral_region,output_he5, fast_mode, suffix):
    dest_file = os.path.join(output_he5,f'{os.path.basename(ref_file)[:-4]}_{suffix}.he5')
    print(f'\n save_copy_he5 - > name and output_name:   \n {dest_file}')
    product_out= f'HDFEOS/SWATHS/PRS_L1_{sensor}/Data Fields/{spectral_region}_{method}_Cube'
    
    if method  == 'normal' and fast_mode == True:
        product_out= f'HDFEOS/SWATHS/PRS_L1_{sensor}/Data Fields/{spectral_region}_{method}_fast_mode_Cube'
     
    if not os.path.exists(dest_file):
        shutil.copyfile(ref_file, dest_file)
     
    with h5py.File(dest_file, 'r+') as f:
        if f.get(product_out):
            del f[product_out]
            
        f.create_dataset(product_out, dtype=array.dtype, shape=array.shape, data=array)


