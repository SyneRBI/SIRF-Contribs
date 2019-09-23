#%% Initial imports etc
import numpy as np
import matplotlib.pyplot as plt
import os
import shutil
import pSTIR as pet

from sirf.Utilities import examples_data_path
from ccpi.optimisation.algorithms import CGLS, PDHG
from ccpi.optimisation.operators import GradientSIRF, BlockOperator
from ccpi.optimisation.functions import KullbackLeibler, IndicatorBox, \
                                         BlockFunction, MixedL21Norm, L2NormSquared
                                         
from ccpi.plugins.regularisers import FGP_TV

#%% go to directory with input files

# adapt this path to your situation (or start everything in the relevant directory)
os.chdir(examples_data_path('PET'))

#%% copy files to working folder and change directory to where the output files are
shutil.rmtree('working_folder/thorax_single_slice',True)
shutil.copytree('thorax_single_slice','working_folder/thorax_single_slice')
os.chdir('working_folder/thorax_single_slice')

#%% Read in images
image = pet.ImageData('emission.hv');
image_array=image.as_array()*.05
image.fill(image_array);
mu_map = pet.ImageData('attenuation.hv');
mu_map_array=mu_map.as_array();

#%% Show Emission image

print('Size of emission: {}'.format(image.shape))

plt.imshow(image.as_array()[0])
plt.title('Emission')
plt.show()

plt.imshow(mu_map.as_array()[0])
plt.title('Attenuation')
plt.show()

#%% save max for future displays
cmax = image_array.max()*.6

#%% create acquisition model

am = pet.AcquisitionModelUsingRayTracingMatrix()
# we will increate the number of rays used for every Line-of-Response (LOR) as an example
# (it is not required for the exercise of course)
am.set_num_tangential_LORs(5)
templ = pet.AcquisitionData('template_sinogram.hs')
am.set_up(templ,image); 

# Compute operator norm
from ccpi.optimisation.operators import LinearOperator


def norm(self):
    return LinearOperator.PowerMethod(self, 10)[0]
    
setattr(pet.AcquisitionModelUsingRayTracingMatrix, 'norm', norm)
normK = am.norm()


#%% simulate some data using forward projection

acquired_data=am.forward(image)
acquisition_array = acquired_data.as_array()

#%% Display bitmaps of a middle sinogram
plt.imshow(acquisition_array[0,0,:,:])
plt.title('Acquisition Data')
plt.show()

#%% Generate a noisy realisation of the data

noisy_array=np.random.poisson(acquisition_array).astype('float64')
print(' Maximum counts in the data: %d' % noisy_array.max())
# stuff into a new AcquisitionData object
noisy_data = acquired_data.clone()
noisy_data.fill(noisy_array)

plt.imshow(noisy_data.as_array()[0,0,:,:])
plt.title('Noisy Acquisition Data')
plt.show()

#%% TV reconstruction using PDHG algorithm 

from ccpi.framework import ImageData

alpha = 0.8
 
gtmp = FGP_TV(alpha, 50, 1e-7, 0, 0, 0, 'cpu' )

def SIRF_proximal(x, tau, out = None):

    x_array = x.as_array()
    x_cil = ImageData(x_array)    
    
    if out is not None:
    
        # x is a SIRF ImageData
        prox_tmp = gtmp.proximal(x_cil, tau)
        out.fill(prox_tmp)
            
sigma = 10
tau = 1/(sigma*normK**2)     
           
g = FGP_TV(alpha, 50, 1e-7, 0, 0, 1, 'cpu' ) 
g.proximal = SIRF_proximal
f = KullbackLeibler(noisy_data)
#f = BlockFunction(f1, f2)   

def show_data(it, obj, x):
    plt.imshow(x.as_array()[0])
    plt.show()

# Setup and run the PDHG algorithm
pdhg = PDHG(f = f, g = g, operator = am, tau = tau, sigma = sigma)
pdhg.max_iteration = 1000
pdhg.update_objective_interval = 10
pdhg.run(1000, callback = show_data)
