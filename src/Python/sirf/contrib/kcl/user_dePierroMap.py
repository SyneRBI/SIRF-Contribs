'''User implemented De Pierro MAPEM reconstruction
Example implementation of De Pierro's modified MAPEM algorithm for weighted
quadratically penalised PET image reconstruction. We use the formulation of 
Wang and Qi (2015), with the simplifying assumption of weights that sum to 1
for each neighbourhood. Examples include uniform weights and Bowsher weights
calculated from a noise-free image.
Implemented by Sam Ellis and Camila Munoz at the SIRF Hackathon 2 (Dec 2018)

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
from docopt import docopt
args = docopt(__doc__, version=__version__)

from pUtilities import show_2D_array

import numpy as np

import Prior as pr  # Import Python Prior class by Abi
import pSTIR as pet

# import engine module
exec('from p' + args['--engine'] + ' import *')

# process command-line options
num_subsets = int(args['--subs'])
num_subiterations = int(args['--subiter'])
data_file = args['--file']
data_path = args['--path']
if data_path is None:
    data_path = petmr_data_path('pet')
raw_data_file = existing_filepath(data_path, data_file)

def my_dePierroMap(image, obj_fun, beta, filter, num_subsets, num_subiterations, weights, sensitivity_image):
    
    # Check that weights are normalised
    if (np.abs(np.sum(weights,axis=1)-1)>1.0e-6).any():
        raise ValueError("Weights should sum to 1 for each voxel")
        
    # Create OSEM reconstructor
    OSEM_reconstructor = pet.OSMAPOSLReconstructor()
    OSEM_reconstructor.set_output_filename_prefix('subiter')
    OSEM_reconstructor.set_objective_function(obj_fun)
    OSEM_reconstructor.set_num_subsets(num_subsets)
    OSEM_reconstructor.set_num_subiterations(num_subiterations)    
    OSEM_reconstructor.set_up(image)
    
    current_image = image.clone()

    for iter in range(1,num_subiterations + 1):
        print('\n------------- Subiteration %d' % iter) 
        
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
    resultVec = imageVec[nhoodIndVec]
    result = resultVec.reshape(nhoodInd.shape,order='F')
    
    # compute xreg
    imageReg = 0.5*np.sum(weights*(result + image.reshape(-1,1,order='F')),axis=1)
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
    

def main():

    # output goes to files
    msg_red =pet.MessageRedirector('info.txt', 'warn.txt', 'errr.txt')

    # create acquisition model
    acq_model = pet.AcquisitionModelUsingRayTracingMatrix()

    # PET acquisition data to be read from the file specified by --file option
    print('raw data: %s' % raw_data_file)
    acq_data = pet.AcquisitionData(raw_data_file)
    
    # Noisy data for testing
    noisy_data = acq_data.clone()
    noisy_data_array = np.random.poisson(acq_data.as_array()/10)
    noisy_data.fill(noisy_data_array)

    # create filter that zeroes the image outside a cylinder of the same
    # diameter as the image xy-section size
    filter = pet.TruncateToCylinderProcessor()

    # create initial image estimate
    image_size = (111, 111, 31)
    voxel_size = (3, 3, 3.375) # voxel sizes are in mm
    image = pet.ImageData()
    image.initialise(image_size, voxel_size)
    image.fill(1.0)
    filter.apply(image)

    # create objective function
    obj_fun = pet.make_Poisson_loglikelihood(noisy_data)
    obj_fun.set_acquisition_model(acq_model)
    obj_fun.set_num_subsets(num_subsets)
    obj_fun.set_up(image)
    
    # create new obj_fun to get sensitivity image for one subset for dePierro MAPEM
    obj_fun2 = pet.make_Poisson_loglikelihood(noisy_data)
    obj_fun2.set_acquisition_model(acq_model)
    obj_fun2.set_num_subsets(1)
    obj_fun2.set_up(image)
    sensitivity_image = obj_fun2.get_subset_sensitivity(0)

    # create new noise-free obj_fun to use for guidance in dePierro MAPEM
    obj_fun3 = pet.make_Poisson_loglikelihood(acq_data)
    obj_fun3.set_acquisition_model(acq_model)
    obj_fun3.set_num_subsets(num_subsets)
    obj_fun3.set_up(image)    

    # uniform weights
    weightsUniform = np.ones([sensitivity_image.as_array().size,27],dtype='float')
    weightsUniform = weightsUniform/27.0
    
    # noise free recon with beta = 0 for guidance   
    image_noiseFree = my_dePierroMap \
        (image, obj_fun3, 0, filter, num_subsets, num_subiterations, weightsUniform, sensitivity_image)
    
    # create a Prior for computing Bowsher weights
    myPrior = pr.Prior(sensitivity_image.as_array().shape)
    weightsBowsher = myPrior.BowshserWeights(image_noiseFree.as_array(),10)
    weightsBowsher = np.float64(weightsBowsher/10.0)
    
    # dePierro MAPEM with uniform and Bowsher weights
    beta = 5000.0
    image_dp_b = my_dePierroMap \
        (image, obj_fun, beta, filter, num_subsets, num_subiterations, weightsBowsher, sensitivity_image)
    
    image_dp_u = my_dePierroMap \
        (image, obj_fun, beta, filter, num_subsets, num_subiterations, weightsUniform, sensitivity_image)

    # show reconstructed images at z = 20    
    image_dp_b_array = image_dp_b.as_array()
    image_dp_u_array = image_dp_u.as_array()
    show_2D_array('Noise free image', image_noiseFree.as_array()[20,:,:])
    show_2D_array('DP Bowsher', image_dp_b_array[20,:,:])
    show_2D_array('DP Uniform', image_dp_u_array[20,:,:])
    show_2D_array('Difference DP', image_dp_b_array[20,:,:] - image_dp_u_array[20,:,:])

# if anything goes wrong, an exception will be thrown 
# (cf. Error Handling section in the spec)
try:
    main()
    print('done')
except pet.error as err:
    # display error information
    print('%s' % err.value)
