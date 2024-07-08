#!/usr/bin/env python3
# -*- coding: utf-8 -*-



import numpy as np
import h5py
import os
import sys

from skimage import data, img_as_float
from skimage.restoration import denoise_nl_means, estimate_sigma
from skimage.metrics import peak_signal_noise_ratio
from skimage.util import random_noise
import argparse

import functions_he5 as f_he5


"""
# Estimate non local means
# https://scikit-image.org/docs/stable/api/skimage.restoration.html#skimage.restoration.denoise_nl_means
# https://scikit-image.org/docs/stable/auto_examples/filters/plot_nonlocal_means.html
"""  

def createParser():
    '''
    Command line parser.
    '''
     
    parser = argparse.ArgumentParser(description='This function contains the Prisma basic processing, based denoising code of the VSNR algorithm by Fehrenbach & Weiss and HySime (hyperspectral signal subspace identiﬁcation by minimum error, algorithm originally Nascimento & Bioucas-Dias') 
     
    parser.add_argument('-if ','--input filename', dest='input_he5', type=str, required=True, help='Input the path and filename in .he5 format')
    parser.add_argument('-s','--sensor', dest='sensor', type=str, required=False, help='Input options: HCO or HRC', default='HRC')
    parser.add_argument('-sr','--spectral_region', dest='spectral_region', type=str, required=False, help='Input options: VNIR or SWIR', default='VNIR')
    parser.add_argument('-nt', '--noise_type', dest='noise_type', type=str, required=True, help='Input options: additive or poisson or normal') 
    parser.add_argument('-op ','--output path', dest='output_he5', type=str, required=False, help='Output path for denoised images in .he5 format')
    parser.add_argument('-su ','--suffix', dest='suffix', type=str, required=False, help='Suffix for filename output. Default = denoised',default='denoised')

    
    # Non_local means parameters 
    parser.add_argument('-ps', '--patch_size', dest='patch_size', type=int, required=False, help='Input patch_size... for instance: 5 \nsee: https://scikit-image.org/docs/stable/auto_examples/filters/plot_nonlocal_means.html', default=5)     
    parser.add_argument('-pd', '--patch_distance', dest='patch_distance', type=int, required=False, help='Input patch_size... for instance: 6  \nsee: https://scikit-image.org/docs/stable/auto_examples/filters/plot_nonlocal_means.html', default=6)   
    parser.add_argument('-hd', '--h_decay', dest='h', type=float, required=False, help='Input h ... for instance: 0.6  \nsee: https://scikit-image.org/docs/stable/auto_examples/filters/plot_nonlocal_means.html', default=0.6)   
    parser.add_argument('-f_m', '--fast_mode', dest='fast_mode', action='store_true', help='Write -f_m, if you want to use fast_mode is True -> see: https://scikit-image.org/docs/stable/auto_examples/filters/plot_nonlocal_means.html')   
   
    return parser

def cmdLineParse(iargs=None):
    parser = createParser()
    return parser.parse_args(args=iargs)
    
def main(args=None):
     
    inps = cmdLineParse(args)
     
    input_he5 = inps.input_he5
    sensor = inps.sensor
    spectral_region = inps.spectral_region
    noise_type = inps.noise_type.lower()
    patch_size = int(inps.patch_size)
    patch_distance = int(inps.patch_distance)
    h = float(inps.h)
    fast_mode = inps.fast_mode
    output_he5 = inps.output_he5
    suffix= inps.suffix
    
    if not os.path.exists(input_he5): 
        print(f'File {input_he5}, not found...\n')
        print(f'Check your path or file')
        sys.exit(1)
    
    if output_he5 == None:
        output_he5 = os.path.dirname(input_he5)

    else:
        if not os.path.exists(output_he5):
            os.makedirs(output_he5)          
        
    # Executing functions
    data = f_he5.read_data(input_he5,spectral_region,sensor)
     
    data = data[:,:,3:8]  # Test Comment later   
    lines, samples, bands = data.shape      
        
    print('1. Process Read Data')
    print('-------- '*8)
    print(f'Filename: {input_he5}')
    print(f'Spectral_region, Sensor selected: {spectral_region, sensor}')
    print(f'Data \nLines, Samples, Bands: {lines, samples, bands}')
       
    # =============================================================================
    # Destriping
    # =============================================================================
    print('\n2. Process Destriping Information')
    print('-------- '*8)
        
    # Creating matrix to store data corrected 
    data_corr = np.empty_like(data) 
    cvg_crit = []
    # Default: f_he5.destripe(band, alpha=1e-2, name='gabor', sigma=(1, 30), theta=20, maxit=100, thr=1e-4)
    # bands = data_corr.shape[-1]
    for i in range(bands):       
        print(f'Band: {i}')
        data_corr[:, :, i], cvg_c = f_he5.destripe(data[:, :, i])
        cvg_crit.append(cvg_c)
        
    print('shape: ', data_corr.shape) 
    
    # =============================================================================
    # Estimation Noise Nascimento and Bioucas-Dias Method or 
    # =============================================================================
    rows, cols, L  = data_corr.shape
    N = rows*cols
    data2 = data_corr.copy().reshape(N, L)
    
    print(f'\n3. Process Estimating the noise - {noise_type} from the image')     
    print('-------- '*8)
    
    if noise_type == 'additive':
        w, Rw, Rw2 = f_he5.est_additive_noise(data2)
        print(f'Estimated noise = {Rw2}')         
        
    elif noise_type == 'poisson':
        print(f'Estimating Noise - {noise_type}')
        w, Rw, Rw2 = f_he5.est_poisson_noise(data2)
        print(f'Estimated noise = {Rw2}')       
                
    elif noise_type == 'normal':
        sigma_est = np.nanmean(estimate_sigma(data_corr, channel_axis=-1)) 
        print(f'Estimated noise standard deviation = {sigma_est}')
        patch_kw = dict(patch_size=inps.patch_size,patch_distance=inps.patch_distance,channel_axis=-1)   

    else:
        'This method is not available! Enter: normal, additive or poisson'
        
        
      
    # =============================================================================
    # Denoise
    # =============================================================================
    print('\n4. Process Denoise')
    print('-------- '*8)

    
    if noise_type == 'additive' or noise_type == 'poisson':
        print('Base on Nascimento & Bioucas-Dias')
        w = w.reshape(rows, cols, L)
        denoise_w = data_corr - w
        denoise_w = np.transpose(denoise_w, axes=[1,2,0]) # data rearranged as original (samples,bands,lines) 
        print(f'\nProcess Saving data ... {noise_type}: \nShape{denoise_w.shape}')
        f_he5.save_copy_he5(input_he5, denoise_w, noise_type, sensor, spectral_region, output_he5, fast_mode, suffix)
        matrix = denoise_w
        
    elif noise_type == 'normal':
        print('By non local means:')
        # slow algorithm
        if fast_mode == False:
            denoise_nlm = denoise_nl_means(data_corr, h=h * sigma_est, fast_mode=fast_mode,**patch_kw)  
            denoise_nlm = np.transpose(denoise_nlm, axes=[1,2,0])  # data rearranged as original (samples,bands,lines) 
            print(f'\nProcess Saving data ... {noise_type}: \nShape{denoise_nlm.shape}') 
            f_he5.save_copy_he5(input_he5, denoise_nlm, noise_type, sensor, spectral_region, output_he5, fast_mode, suffix)
            matrix = denoise_nlm
            
        else:
            print('fast_mode True')
            denoise_nlm_f = denoise_nl_means(data_corr, h=h * sigma_est, sigma=sigma_est,fast_mode=fast_mode, **patch_kw)  
            denoise_nlm_f = np.transpose(denoise_nlm_f, axes=[1,2,0])  # data rearranged as original (samples,bands,lines) 
            print(f'\nProcess Saving data ... {noise_type}: \nShape{denoise_nlm_f.shape}')
            f_he5.save_copy_he5(input_he5, denoise_nlm_f, noise_type, sensor, spectral_region, output_he5, fast_mode, suffix)
            matrix = denoise_nlm_f
       
     
    print('-------- -------- -------- -  END  - -------- -------- -------- --------')
     
    #return inps,data_corr,matrx          
      
    return inps,data_corr,matrix          
    
# ~ ax[0].imshow(destripe[400:600,0,600:800].T,cmap='gray')    
if __name__ == '__main__':
    '''
    Main driver.
    '''
        
    res = main()
