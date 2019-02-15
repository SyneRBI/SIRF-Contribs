'''User implemented De Pierro MAPEM reconstruction
Real data implementation of De Pierro MAPEM, using a Bowsher weighted quadratic
penalty. The guidance image (here a T1-weighted MR image) must be pre-aligned 
to the PET image and sampled on the same image grid.
Implemented by Sam Ellis (13th Feb 2019)

Usage:
  dePierroMap_eg [--help | options]

Options:
  -f <file>, --file=<file>    raw data file
                              [default: my_forward_projection.hs]
  -p <path>, --path=<path>    path to data files, defaults to data/examples/PET
                              subfolder of SIRF root folder
  -s <subs>, --subs=<subs>    number of subsets [default: 12]
  -i <siter>, --subiter=<siter>    number of sub-iterations [default: 24]
  -e <engn>, --engine=<engn>  reconstruction engine [default: STIR]
'''

## CCP PETMR Synergistic Image Reconstruction Framework (SIRF)
## Copyright 2015 - 2017 Rutherford Appleton Laboratory STFC
## Copyright 2015 - 2017 University College London.
##
## This is software developed for the Collaborative Computational
## Project in Positron Emission Tomography and Magnetic Resonance imaging
## (http://www.ccppetmr.ac.uk/).
##
## Licensed under the Apache License, Version 2.0 (the "License");
##   you may not use this file except in compliance with the License.
##   You may obtain a copy of the License at
##       http://www.apache.org/licenses/LICENSE-2.0
##   Unless required by applicable law or agreed to in writing, software
##   distributed under the License is distributed on an "AS IS" BASIS,
##   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
##   See the License for the specific language governing permissions and
##   limitations under the License.

__version__ = '0.1.0'

def my_dePierroMap(image, obj_fun, beta, filter, weights, sensitivity_image):
    
    # Check that weights are normalised
    if (np.abs(np.sum(weights,axis=1)-1)>1.0e-6).any():
        raise ValueError("Weights should sum to 1 for each voxel")
        
    # Create OSEM reconstructor
    print('Setting up reconstruction object')
    OSEM_reconstructor = OSMAPOSLReconstructor()
    OSEM_reconstructor.set_objective_function(obj_fun)                             
    OSEM_reconstructor.set_num_subsets(21)
    OSEM_reconstructor.set_num_subiterations(21*10)
    OSEM_reconstructor.set_up(image)
    num_subiterations = OSEM_reconstructor.get_num_subiterations()
    
    current_image = image.clone()

    for iter in range(1,num_subiterations + 1):
        print('\n------------- Subiteration %d' % iter) 

        # clear the temp files from the current working directory (vital when
        # reconstructing real datasets with many iterations)
        if np.mod(iter,5) == 0:
            os.system('rm *.hv *.hs *.v *.s *.ahv')
        
        # Calculate imageReg and return as an array
        imageReg_array = dePierroReg(current_image.as_array(),weights)
        
        # OSEM image update
        OSEM_reconstructor.update(current_image)
        imageEM_array = current_image.as_array()
        
        # Final image update
        imageUpdated_array = dePierroUpdate \
            (imageEM_array, imageReg_array, beta, sensitivity_image.as_array())
        
        # Fill image and truncate to cylindrical field of view        
        current_image.fill(imageUpdated_array)
        filter.apply(current_image)
        
    image_out = current_image.clone()
    return image_out


def dePierroUpdate(imageEM, imageReg, beta, sensImg):
    
    delta = 1e-6*abs(sensImg).max()
    sensImg[sensImg < delta] = delta # avoid division by zero
    beta_j = beta/sensImg
    
    b_j = 1 - beta_j*imageReg
    
    numer = (2*imageEM)
    denom = ((b_j**2 + 4*beta_j*imageEM)**0.5 + b_j)
    
    delta = 1e-6*abs(denom).max()
    denom[denom < delta] = delta # avoid division by zero
    
    imageUpdated = numer/denom
    
    return imageUpdated

def dePierroReg(image,weights):
    
    # get size and vectorise image for indexing 
    imSize = image.shape
    imageVec = image.reshape(-1,1,order='F').flatten('F')
    
    # get the neigbourhoods of each voxel
    weightsSize = weights.shape
    w = int(round(weightsSize[1]**(1.0/3))) # side length of neighbourhood
    nhoodInd    = neighbourExtract(imSize,w)
    nhoodIndVec = nhoodInd.reshape(-1,1,order='F').flatten('F')
    
    # retrieve voxel intensities for neighbourhoods 
    resultVec = np.float32(imageVec[nhoodIndVec])
    result = resultVec.reshape(nhoodInd.shape,order='F')
    
    # compute xreg
    try:
        imageReg = 0.5*np.sum(weights*(result + np.float32(image).reshape(-1,1,order='F')),axis=1)
    except:
        tmpVar = 1;    
    imageReg = imageReg.reshape(imSize,order='F')
    
    return imageReg

def neighbourExtract(imageSize,w):
    # Adapted from Prior class        
    n = imageSize[0]
    m = imageSize[1]
    h = imageSize[2]
    wlen = 2*np.floor(w/2)
    widx = xidx = yidx = np.arange(-wlen/2,wlen/2+1)

    if h==1:
        zidx = [0]
        nN = w*w
    else:
        zidx = widx
        nN = w*w*w
        
    Y,X,Z = np.meshgrid(np.arange(0,m), np.arange(0,n), np.arange(0,h))                
    N = np.zeros([n*m*h, nN],dtype='int32')
    l = 0
    for x in xidx:
        Xnew = setBoundary(X + x,n)
        for y in yidx:
            Ynew = setBoundary(Y + y,m)
            for z in zidx:
                Znew = setBoundary(Z + z,h)
                N[:,l] = ((Xnew + (Ynew)*n + (Znew)*n*m)).reshape(-1,1).flatten('F')
                l += 1
    return N
    
def setBoundary(X,n):
    # Boundary conditions for neighbourExtract
    # Adapted from Prior class
    idx = X<0
    X[idx] = X[idx] + n
    idx = X>n-1
    X[idx] = X[idx] - n
    return X.flatten('F')

# %%
import os
import sys
import matplotlib.pyplot as plt
from pUtilities import show_2D_array
from pSTIR import *
import numpy as np
import Prior as pr

data_path = '/media/sf_SIRF_data/sino_rawdata_100/'
#data_path='/home/sirfuser/data/NEMA'
print('Finding files in %s' % data_path)

num_subsets = 12
    
# set filenames 
# input files
sino_file = 'my_data_sino.hs'
norm_file = 'my_data_norm.hs'
attn_file = 'my_data_mumap.hv'
rand_file = 'my_data_rand.hs'
mr_file = 'my_data_MR_SIRF.hv'

# output goes to files
msg_red = MessageRedirector('info.txt', 'warn.txt', 'error.txt')

acq_data = AcquisitionData(data_path + sino_file)

#%%

# copy the acquisition data into a Python array
acq_array = acq_data.as_array()
print('acquisition data dimensions: %dx%dx%d' % acq_array.shape)
# use a slice number for display that is appropriate for the NEMA phantom
z = 71
show_2D_array('Acquisition data', acq_array[z,:,:])

# create acquisition model
acq_model = AcquisitionModelUsingRayTracingMatrix()
acq_model.set_num_tangential_LORs(10);

#%% Correction sinograms
norm_file = 'data-norm.n.hdr'
asm_norm = AcquisitionSensitivityModel(data_path + norm_file)
acq_model.set_acquisition_sensitivity(asm_norm)

# ---------------- taken from the example-----------------------------------
attn_image = ImageData(data_path + attn_file)
attn_acq_model = AcquisitionModelUsingRayTracingMatrix()
attn_acq_model.set_num_tangential_LORs(10)
asm_attn = AcquisitionSensitivityModel(attn_image, attn_acq_model)

# temporary fix pending attenuation offset fix in STIR:
# converting attenuation into 'bin efficiency'
asm_attn.set_up(acq_data)
attn_factors = AcquisitionData(acq_data)
attn_factors.fill(1.0)
print('applying attenuation (please wait, may take a while)...')
asm_attn.unnormalise(attn_factors)
asm_attn = AcquisitionSensitivityModel(attn_factors)
asm = AcquisitionSensitivityModel(asm_norm, asm_attn)
acq_model.set_acquisition_sensitivity(asm)
# --------------------------------------------------------------------------

# randoms
randoms = AcquisitionData(data_path + rand_file)
randoms_array=randoms.as_array()
show_2D_array('randoms',randoms_array[z,:,:])
acq_model.set_background_term(randoms)

# MR guidance
mr_image = ImageData(data_path + mr_file)
mr_array = mr_image.as_array()
show_2D_array('MR image',mr_array[45,110:220,115:225])


#%%
# define objective function to be maximized as
# Poisson logarithmic likelihood (with linear model for mean)
obj_fun = make_Poisson_loglikelihood(acq_data)
obj_fun.set_acquisition_model(acq_model)

#%%

# create initial image estimate from one iteration of MLEM
recon_init = OSMAPOSLReconstructor()
recon_init.set_objective_function(obj_fun)
                             
recon_init.set_num_subsets(1)
recon_init.set_num_subiterations(1)
nxny = (344, 344, 127)
initial_image = acq_data.create_uniform_image(1.0, nxny)

image=initial_image

recon_init.set_up(image)

recon_init.set_current_estimate(image)

recon_init.process()

image = recon_init.get_current_estimate()


# %% bit more prep

# create filter that zeroes the image outside a cylinder of the same
# diameter as the image xy-section size
filter = TruncateToCylinderProcessor()

# filter image estimate to FOV
filter.apply(image)

# get the full sensitivity image
obj_fun2 = make_Poisson_loglikelihood(acq_data)
obj_fun2.set_acquisition_model(acq_model)
obj_fun2.set_num_subsets(1)
obj_fun2.set_up(image)
sensitivity_image = obj_fun2.get_subset_sensitivity(0)


# %% guided reconstruction

# create a Prior for computing Bowsher weights
myPrior = pr.Prior(sensitivity_image.as_array().shape)
weights = myPrior.BowshserWeights(mr_array,7)
weights = np.float32(weights/7.0)

image_guided = my_dePierroMap(image, obj_fun, 50000, filter, weights, sensitivity_image)
image_array_guided = image_guided.as_array()
show_2D_array('Reconstructed guided', image_array_guided[45,110:220,115:225])

image_guided.write('output_images/image_guided.v')

## %% OSEM reconstruction (beta = 0)
#
#image_OSEM = my_dePierroMap(image, obj_fun, 0, filter, weights, sensitivity_image)
#image_array_OSEM = image_OSEM.as_array()
#show_2D_array('Reconstructed OSEM image', image_array_OSEM[45,110:220,115:225])
#
#image_OSEM.write('output_images/image_OSEM.v')
#
## %% unguided reconstruction
#
## uniform weights
#weights = np.ones([image.as_array().size,27],dtype='float')
#weights = np.float32(weights/27.0)
#
#image_unguided = my_dePierroMap(image, obj_fun, 50000, filter, weights, sensitivity_image)
#image_array_unguided = image_unguided.as_array()
#show_2D_array('Reconstructed unguided', image_array_unguided[45,110:220,115:225])
#
#image_unguided.write('output_images/image_unguided.v')

