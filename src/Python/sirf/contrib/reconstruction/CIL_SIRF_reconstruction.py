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

#% go to directory with input files

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
    #
    ##%% copy files to working folder and change directory to where the output files are
    shutil.rmtree('working_folder/thorax_single_slice',True)
    shutil.copytree('thorax_single_slice','working_folder/thorax_single_slice')
    os.chdir('working_folder/thorax_single_slice')
    
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

# Define norm for the acquisition model
def norm(self):
    return LinearOperator.PowerMethod(self, 10)[0]
    
setattr(pet.AcquisitionModelUsingRayTracingMatrix, 'norm', norm)

#%%

am = pet.AcquisitionModelUsingRayTracingMatrix()
# we will increate the number of rays used for every Line-of-Response (LOR) as an example
# (it is not required for the exercise of course)
am.set_num_tangential_LORs(5)
templ = pet.AcquisitionData(sinogram_header)
pet.AcquisitionData.set_storage_scheme('memory')
am.set_up(templ,image)

#% simulate some data using forward projection
if EXAMPLE == 'SIMULATION':
    
    acquired_data = templ
    image.fill(1)
    noisy_data = acquired_data.clone()

elif EXAMPLE == 'SMALL':
    
    acquired_data=am.forward(image)
    
    acquisition_array = acquired_data.as_array()

    np.random.seed(10)
    noisy_data = acquired_data.clone()
    scale = 100
    noisy_array = scale * np.random.poisson(acquisition_array/scale).astype('float64')
    print(' Maximum counts in the data: %d' % noisy_array.max())
    noisy_data.fill(noisy_array)



#%% Display bitmaps of a middle sinogram
    
plt.imshow(noisy_array[0,0,:,:])
plt.title('Acquisition Data')
plt.show()

# Show util per iteration
def show_data(it, obj, x):
    plt.imshow(x.as_array()[0])
    plt.colorbar()
    plt.show()

#%% TV reconstruction using algorithm below

alpha = 2.5

ALGORITHM = 'FISTA_CIL' # or PDHG_CIL, PDHG_SIRF, FISTA_CIL, FISTA_SIRF, OSMAPOSL

if  ALGORITHM == 'PDHG_CIL':
    
    method = 'implicit'
    
    if method == 'explicit':
        
        # Create operators
        op1 = GradientSIRF(image) 
        op2 = am
    
        # Create BlockOperator
        operator = BlockOperator(op1, op2, shape=(2,1) ) 
        
        f2 = KullbackLeibler(noisy_data)  
        g =  IndicatorBox(lower=0)    
                
        f1 = alpha * MixedL21Norm() 
        f = BlockFunction(f1, f2)  
        normK = operator.norm()
        
    elif method == 'implicit':
        
        operator = am      
        g = FGP_TV(alpha, 50, 1e-7, 0, 1, 0, 'cpu' ) 
        f = KullbackLeibler(noisy_data)
        normK = operator.norm()
         
    sigma = 0.001
    tau = 1/(sigma*normK**2)      
        
    # Setup and run the PDHG algorithm
    pdhg = PDHG(f = f, g = g, operator = operator, tau = tau, sigma = sigma)
    pdhg.max_iteration = 500
    pdhg.update_objective_interval = 50
    pdhg.run(1000, callback = show_data)
        
elif ALGORITHM == 'FISTA_CIL':
    
    tmp_fun = KullbackLeibler(noisy_data)
    tmp_fun.L = 1
    f = FunctionOperatorComposition(tmp_fun, am)
    f.L = 100
    g = FGP_TV(alpha, 50, 1e-7, 0, 1, 0, 'cpu' )
    
    # initial image
    
#    init_image=image.clone()
#    cmax = image_array.max()
#    init_image.fill(cmax/32)
#    init_image = image.clone()
#    
#    def make_cylindrical_FOV(image):
#        """truncate to cylindrical FOV"""
#        filter = pet.TruncateToCylinderProcessor()
#        filter.apply(image)
#    
#    make_cylindrical_FOV(init_image)
#    
#    plt.imshow(init_image.as_array()[0])
#    plt.colorbar()
#    plt.show()
    
#    cmax = image.as_array().max()
    x_init = image.allocate(1)
    fista = FISTA(x_init=x_init, f = f, g = g)
    fista.max_iteration = 500
    fista.update_objective_interval = 50
    fista.run(500, verbose=True, callback = show_data) 
    
elif ALGORITHM == 'FISTA_SIRF':
    
    from sirf.Utilities import check_status, assert_validity
    import sirf.pystir as pystir

    def KL_call(self, x):
        return -self.get_value(x)
    
    def gradient(self, image, subset = -1, out = None):
        
        assert_validity(image, pet.ImageData)
        grad = pet.ImageData()
        grad.handle = pystir.cSTIR_objectiveFunctionGradient\
            (self.handle, image.handle, subset)
        check_status(grad.handle)
        
        if out is None:
            return -1*grad  
        else:
            out.fill(-1*grad)

    setattr(pet.ObjectiveFunction, '__call__', KL_call)
    setattr(pet.PoissonLogLikelihoodWithLinearModelForMeanAndProjData, 'gradient', gradient)

    fidelity = pet.PoissonLogLikelihoodWithLinearModelForMeanAndProjData()
    fidelity.set_acquisition_model(am)
    fidelity.set_acquisition_data(noisy_data)
    fidelity.set_num_subsets(4)
    fidelity.set_up(image)
    fidelity.L = 100
        
    g = FGP_TV(alpha, 50, 1e-7, 0, 1, 0, 'cpu' )

    x_init = image.allocate(1) #init_image.clone()
#    x_init = image.clone()
    fista = FISTA(x_init = x_init, f = fidelity, g = g)
    fista.max_iteration = 2000
    fista.update_objective_interval = 200
    fista.run(2000, verbose=True, callback = show_data) 
    
elif ALGORITHM == 'OSMAPOSL':
    
    fidelity = pet.PoissonLogLikelihoodWithLinearModelForMeanAndProjData()
    fidelity.set_acquisition_model(am)
    fidelity.set_acquisition_data(noisy_data)
    fidelity.set_num_subsets(4)
    fidelity.set_up(image)
    
    recon = pet.OSMAPOSLReconstructor()
    recon.set_objective_function(fidelity)
    recon.set_num_subsets(4)
    num_iters=10;
    recon.set_num_subiterations(num_iters)
    
    reconstructed_image = image.allocate(1)
    recon.set_up(reconstructed_image)
    recon.reconstruct(reconstructed_image)

    plt.imshow(reconstructed_image.as_array()[0])
    plt.colorbar()
    plt.show()    

    
    
