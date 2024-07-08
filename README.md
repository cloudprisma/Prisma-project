# 🛰️ SAPP4VU: Sviluppo di Algoritmi prototipali Prisma per la Stima del Danno Ambientale e della Vulnerabilità alla Land Degradation

## 🚀 "Implementazione di algoritmi numerico-statistici per la caratterizzazione e rimozione del rumore e per la cloud detection in immagini iperspettrali.” 

## Description
This repository contains the Python sources of the Prisma basic processing, based denoising code of the VSNR algorithm originally coded in MATLAB - see the [Pierre Weiss website](https://www.math.univ-toulouse.fr/~weiss/PageCodes.html) & [pyvsnr](https://github.com/patquem/pyvsnr/tree/main) and HySime (hyperspectral signal subspace identiﬁcation by minimum error, algorithm originally coded in MATLAB) - see the [José Bioucas-Dias website](http://www.lx.it.pt/~bioucas/code.htm)

## How it works?
  
![image](https://github.com/argennof/Prisma-proyect/assets/11649711/05c072a4-b8d7-4be7-9000-21372e2bf280)


## **Requirements**
   - [x] matplotlib==3.7.1
   - [x] h5py==3.8.0
   - [x] scipy
   - [x] numpy
   - [x] scikit-image==0.20.0
   - [x] pyvsnr==1.0.0
   - [x] argparse
 

----
# 📡 How to run? 
----
  1. Create an environment, for instance:
  ```
    $ pip install virtualenv
    $ python3.1  -m venv <virtual-environment-name>
  ```
  
  2. Activate your virtual environment:
  ```
      $ source env/bin/activate
  ```
  3.  Install the requirements in the Virtual Environment, you can easily just pip install the libraries. For example:
  ```
      $ pip install pyvsnr
  ```
  or  If you use the requirements.txt file:
  ```
      $ pip install -r requirements.txt
  ```

  4. Download the scripts available here ( _*main.py_ and _*functions_he5.py_ ) and save them into the same directory.
  5. Then, once at the directory, execute the next command in a terminal e.g.
 ```
      $ python3 main.py -if ./PRS_L1_STD_OFFL_20210922101425_20210922101429_0001.he5 -s HRC -sr VNIR -nt additive 
  ```
  ----
 ## :bulb: Usage:
     
      python3 main.py [-if <filename>] [-s {HRC (default), HCO}] [-sr {VNIR (default), SWIR}] 
      [-nt {additive, poisson, normal}] [-ps] [-pd] [-h_d] [-f_m]

    
    Commands and Options:
    
      --h, --help:
       Output usage information
      
      -if, --input filename <filename>, string, required.                 
      The data file to process. It must be loaded in .he5 format. 
   
      -op, --output path <directory>, string, optional.
      Output path for denoised images. The output format will be in .he5 
      
      -s, --sensor {HRC (default), HCO}, string, optional.
      Select a specific sensor to process the information. It may be one of 
      the following: HCO or HRC. Otherwise, the default option is ("HRC").
      
      -sr, --spectral_region {VNIR (default), SWIR}, string, optional.               
      Select a spectral region to process the information. It may be one of 
      the following: VNIR or SWIR. Otherwise, the default option is ("VNIR"), string, optional.

      -su,--suffix  <suffix> (default),string, optional.
      Suffix for filename output.
      
      -nt, --noise_type {additive, poisson, normal}, string, required.
      Select a noise type to process the information. It may be one of the following: additive, poisson or normal.
      
      -ps, --patch_size  integer, optional -> Enabled only for normal option.
      Size of patches used for denoising. For more information see skimage.restoration.denoise_nl_means.

      -pd, --patch_distance, integer, optional -> Enabled only for normal option.
      Maximal distance in pixels where to search patches used for denoising. For more information see 
      skimage.restoration.denoise_nl_means.

      -hd, --h_decay, float, optional -> Enabled only for normal option.
      Cut-off distance (in gray levels). The higher h, the more permissive one is in accepting patches. 
      A higher h results in a smoother image, at the expense of blurring features. For a Gaussian noise 
      of standard deviation sigma, a rule of thumb is to choose the value of h to be sigma of slightly 
      less. For more information see skimage.restoration.denoise_nl_means.
 
      -f_m, --fast_mode,string, optional -> Enabled only for normal option.
      Write -f_m, if you want to make fast_mode=True. 
      A fast version of the non-local means algorithm is used. For more information see skimage.restoration.denoise_nl_means.

      

> <picture>
>   <source media="(prefers-color-scheme: light)" srcset="https://raw.githubusercontent.com/Mqxx/GitHub-Markdown/main/blockquotes/badge/light-theme/info.svg">
>   <img alt="Info" src="https://raw.githubusercontent.com/Mqxx/GitHub-Markdown/main/blockquotes/badge/dark-theme/info.svg">
> </picture><br>
>
> [skimage.restoration.denoise_nl_means](https://scikit-image.org/docs/stable/api/skimage.restoration.html#skimage.restoration.denoise_nl_means)

## :heavy_check_mark: **More examples**: 
1. If you choose HCO - VNIR - Additive and Output path: 
```
$ python3 main.py -if ./PRS_L1_STD_OFFL_20210922101425_20210922101429_0001.he5 -s HCO -sr VNIR -nt additive -op ./Results/
```
### Additive noise removal example:
![image](https://github.com/argennof/Prisma-proyect/assets/11649711/c0e57428-ca05-4da7-9b31-5a8507016270)

2. If you choose HCO - VNIR - Poisson: 
```
$ python3 main.py -if ./PRS_L1_STD_OFFL_20210922101425_20210922101429_0001.he5 -s HCO -sr VNIR -nt poisson 
```
### Poisson noise removal result:
![image](https://github.com/argennof/Prisma-proyect/assets/11649711/dfc7823c-4781-45ef-8f9b-d1ba111301dd)

3. If you choose HCO - VNIR - Normal - Fast mode: 
```
$ python3 main.py -if ./PRS_L1_STD_OFFL_20210922101425_20210922101429_0001.he5 -s HCO -sr SWIR -nt normal -hd 0.1 -f_m 
```

Get some help:
```
$ python3 main.py  --help
```
  ----
# 📝 Authors information
This is adapted to Python from the original Matlab codes developed by:
 - [x] Jérôme Fehrenbach and Pierre Weiss.
 - [x] José Bioucas-Dias and José Nascimento

All credit goes to the original author.

In case you use the results of this code with your article, please don't forget to cite:

- [x] Fehrenbach, Jérôme, Pierre Weiss, and Corinne Lorenzo. "Variational algorithms to remove stationary noise: applications to microscopy imaging." IEEE Transactions on Image Processing 21.10 (2012): 4420-4430.
- [x] Fehrenbach, Jérôme, and Pierre Weiss. "Processing stationary noise: model and parameter selection in variational methods." SIAM Journal on Imaging Sciences 7.2 (2014): 613-640.
- [x] Escande, Paul, Pierre Weiss, and Wenxing Zhang. "A variational model for multiplicative structured noise removal." Journal of Mathematical Imaging and Vision 57.1 (2017): 43-55.
- [x] Bioucas-Dias, J. and  Nascimento, J.  "Hyperspectral subspace identification", IEEE Transactions on Geoscience and Remote Sensing, vol. 46, no. 8, pp. 2435-2445, 2008
