#%% Initial imports etc
import numpy as np
import matplotlib.pyplot as plt
import os
import shutil
import pSTIR as pet

from sirf.Utilities import examples_data_path
from ccpi.optimisation.algorithms import CGLS, PDHG, FISTA
from ccpi.optimisation.operators import GradientSIRF, BlockOperator, LinearOperator
from ccpi.optimisation.functions import KullbackLeibler, IndicatorBox, BlockFunction, MixedL21Norm    
from ccpi.framework import ImageData
from ccpi.plugins.regularisers import FGP_TV

#%% go to directory with input files

# adapt this path to your situation (or start everything in the relevant directory)
os.chdir('/home/sirfuser/Documents/Hackathon4/')
#
##%% copy files to working folder and change directory to where the output files are
shutil.rmtree('exhale-output',True)
shutil.copytree('Exhale','exhale-output')
os.chdir('exhale-output')

#%% Read in images
image = pet.ImageData('pet_dyn_4D_resp_simul_dynamic_0_state_0_attenuation_map.hv');
image.fill(1)
image_array=image.as_array()
mu_map = pet.ImageData('pet_dyn_4D_resp_simul_dynamic_0_state_0_attenuation_map.hv');
mu_map_array=mu_map.as_array();

#%% Show Emission image

print('Size of emission: {}'.format(image.shape))

plt.imshow(image.as_array()[0])
plt.title('Emission')
plt.show()

plt.imshow(mu_map.as_array()[0])
plt.title('Attenuation')
plt.show()

#%% create acquisition model

am = pet.AcquisitionModelUsingRayTracingMatrix()
# we will increate the number of rays used for every Line-of-Response (LOR) as an example
# (it is not required for the exercise of course)
am.set_num_tangential_LORs(12)
templ = pet.AcquisitionData('pet_dyn_4D_resp_simul_dynamic_0_state_0.hs')
pet.AcquisitionData.set_storage_scheme('memory')
am.set_up(templ,image)

#%% simulate some data using forward projection

acquired_data=templ
acquisition_array = acquired_data.as_array()

plt.imshow(acquisition_array[0,100,:,:], vmin = 0, vmax = 7)
plt.title('Acquisition Data')
plt.show()

#%% Generate a noisy realisation of the data

#noisy_array=np.random.poisson(acquisition_array).astype('float64')
#print(' Maximum counts in the data: %d' % noisy_array.max())
## stuff into a new AcquisitionData object
noisy_data = acquired_data.clone()
#noisy_data.fill(noisy_array)


#noisy_dat
plt.imshow(noisy_data.as_array()[0,100,:,:])
plt.title('Noisy Acquisition Data')
plt.show()

#%%


def show_data(it, obj, x):
    plt.imshow(x.as_array()[63])
    plt.colorbar()
    plt.show()


init_image=image.clone()

    #%%
from ccpi.optimisation.functions import ZeroFunction

def KL_call(self, x):
    return self.get_value(x)
    
setattr(pet.ObjectiveFunction, '__call__', KL_call)
fidelity = pet.PoissonLogLikelihoodWithLinearModelForMeanAndProjData()

fidelity.set_acquisition_model(am)
fidelity.set_acquisition_data(noisy_data)
fidelity.set_num_subsets(21)
fidelity.set_up(image)

fidelity.get_num_subsets = 21

#%%
alpha = 1e-12
fidelity.L = 1000
regularizer = ZeroFunction()

x_init = init_image
fista = FISTA()
fista.set_up(x_init = x_init , f = fidelity, g = regularizer)
fista.max_iteration = 500
fista.update_objective_interval = 50
fista.run(500, verbose = False, callback = show_data)

#%%


cgls = CGLS(x_init = init_image, operator = am, data = noisy_data)
cgls.max_iteration = 20
cgls.update_objective_interval = 5
cgls.run(20, verbose = True)

#%%
sol_cgls = cgls.get_output().as_array()
plt.imshow(sol_cgls[63], vmin = 0)
plt.title('CGLS reconstruction')
plt.colorbar()
plt.show()
