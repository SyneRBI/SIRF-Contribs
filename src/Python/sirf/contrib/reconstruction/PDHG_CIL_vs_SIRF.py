#%% Initial imports etc
import numpy as np
import matplotlib.pyplot as plt
import os
import shutil
import pSTIR as pet

from sirf.Utilities import examples_data_path
from ccpi.optimisation.algorithms import CGLS, PDHG, FISTA
from ccpi.optimisation.operators import GradientSIRF, BlockOperator, LinearOperator
from ccpi.optimisation.functions import KullbackLeibler, IndicatorBox, FunctionOperatorComposition, BlockFunction, MixedL21Norm    
from ccpi.framework import ImageData
from ccpi.plugins.regularisers import FGP_TV

from utils_SIRF_CIL_reconstruction import run_CIL_SIRF_utils

run_CIL_SIRF_utils()
#%%

EXAMPLE = 'SMALL'

if EXAMPLE == 'SIMULATION':
    
    # adapt this path to your situation (or start everything in the relevant directory)
    os.chdir('/home/sirfuser/Documents/Hackathon4/')    
    ##%% copy files to working folder and change directory to where the output files are
    shutil.rmtree('exhale-output',True)
    shutil.copytree('Exhale','exhale-output')
    os.chdir('exhale-output')
    
    attenuation_header = 'pet_dyn_4D_resp_simul_dynamic_0_state_0_attenuation_map.hv'
    image_header = attenuation_header
    sinogram_header = 'pet_dyn_4D_resp_simul_dynamic_0_state_0.hs'

elif EXAMPLE == 'SMALL':
    # adapt this path to your situation (or start everything in the relevant directory)
    os.chdir(examples_data_path('PET'))
#    #
#    ##%% copy files to working folder and change directory to where the output files are
    shutil.rmtree('working_folder/thorax_single_slice',True)
    shutil.copytree('thorax_single_slice','working_folder/thorax_single_slice')
    os.chdir('working_folder/thorax_single_slice')
        
elif EXAMPLE == 'BRAIN':
    # adapt this path to your situation (or start everything in the relevant directory)
    os.chdir(examples_data_path('PET'))
#    #
#    ##%% copy files to working folder and change directory to where the output files are
    shutil.rmtree('working_folder/brain',True)
    shutil.copytree('brain','working_folder/brain')
    os.chdir('working_folder/brain')
    
image_header = 'emission.hv'
attenuation_header = 'attenuation.hv'
sinogram_header = 'template_sinogram.hs'

# Read in images
image = pet.ImageData(image_header);
image_array=image.as_array()
mu_map = pet.ImageData(attenuation_header);
mu_map_array=mu_map.as_array();

# Show Emission image
print('Size of emission: {}'.format(image.shape))

plt.imshow(image.as_array()[0])
plt.colorbar()
plt.title('Emission')
plt.show()

plt.imshow(mu_map.as_array()[0])
plt.colorbar()
plt.title('Attenuation')
plt.show()

#%% AcquisitionModel

am = pet.AcquisitionModelUsingRayTracingMatrix()
am.set_num_tangential_LORs(12)
templ = pet.AcquisitionData(sinogram_header)
pet.AcquisitionData.set_storage_scheme('memory')
am.set_up(templ,image)

#% simulate some data using forward projection
if EXAMPLE == 'SIMULATION':
    
    acquired_data = templ
    image.fill(1)
    noisy_data = acquired_data.clone()

elif EXAMPLE == 'SMALL' or EXAMPLE == 'BRAIN':
    
    acquired_data=am.forward(image)
    
    acquisition_array = acquired_data.as_array()

    np.random.seed(10)
    noisy_data = acquired_data.clone()
    scale = 100
    noisy_array = scale * np.random.poisson(acquisition_array/scale).astype('float64')
    print(' Maximum counts in the data: %d' % noisy_array.max())
    noisy_data.fill(noisy_array)

if EXAMPLE == 'SIMULATION':
    slice_show = 64
elif EXAMPLE == 'SMALL':
    slice_show = 0    
elif EXAMPLE == 'BRAIN':
    slice_show = 10
    
# Show util per iteration
def show_data(it, obj, x):
    plt.imshow(x.as_array()[slice_show])
    plt.colorbar()
    plt.show()

#%% TV reconstruction

alpha = 2.5

print('#############################')
print('Running PDHG_SIRF')
print('#############################')
          
operator = am      
g = FGP_TV(alpha, 50, 1e-7, 0, 1, 0, 'cpu' ) 

fidelity = pet.PoissonLogLikelihoodWithLinearModelForMeanAndProjData()
fidelity.set_acquisition_model(am)
fidelity.set_acquisition_data(noisy_data)
fidelity.set_num_subsets(1)
fidelity.set_up(image)
fidelity.b = noisy_data
fidelity.bnoise = noisy_data * 0.0

#%%
normK = operator.norm()
     
sigma = 0.01    
tau = 1/(sigma*normK**2)     
    
def SIRF_update_objective(self):
    
    if isinstance(self.f, pet.PoissonLogLikelihoodWithLinearModelForMeanAndProjData):
        p1 = self.f(self.x) + self.g(self.x)
        d1 = -(self.f.convex_conjugate(self.y) + self.g.convex_conjugate(-1*self.operator.adjoint(self.y)))
    else:
        p1 = self.f(operator.direct(self.x)) + self.g(self.x)
        d1 = -(self.f.convex_conjugate(self.y) + self.g.convex_conjugate(-1*self.operator.adjoint(self.y)))

    self.loss.append([p1, d1, p1-d1])    
    
setattr(PDHG, 'update_objective', SIRF_update_objective)
        
# Setup and run the PDHG algorithm
pdhg_SIRF = PDHG(f = fidelity, g = g, operator = operator, tau = tau, sigma = sigma)
pdhg_SIRF.max_iteration = 500
pdhg_SIRF.update_objective_interval = 100
pdhg_SIRF.run(1000, callback = show_data)

#%%
print('#############################')
print('Running PDHG_CIL')
print('#############################')

f = KullbackLeibler(noisy_data)
        
# Setup and run the PDHG algorithm
pdhg_CIL = PDHG(f = f, g = g, operator = operator, tau = tau, sigma = sigma)
pdhg_CIL.max_iteration = 500
pdhg_CIL.update_objective_interval = 100
pdhg_CIL.run(1000, callback = show_data)
   
#%%

sol_pdhg_cil = pdhg_CIL.get_output().as_array()
sol_pdhg_sirf = pdhg_SIRF.get_output().as_array()
diff = np.abs(sol_pdhg_cil - sol_pdhg_sirf)

plt.figure(figsize = (10,10))

plt.subplot(3,1,1)
plt.imshow(sol_pdhg_cil[0])
plt.colorbar()

plt.subplot(3,1,2)
plt.imshow(sol_pdhg_sirf[0])
plt.colorbar()

plt.subplot(3,1,3)
plt.imshow(diff[0])
plt.colorbar()        
