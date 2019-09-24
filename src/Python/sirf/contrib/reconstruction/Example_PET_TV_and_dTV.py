#%% Initial imports etc
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
import matplotlib.pyplot as plt
import os
import shutil
import pSTIR as pet

from ccpi.optimisation.algorithms import Algorithm
from ccpi.optimisation.functions import ZeroFunction
import numpy


from sirf.Utilities import examples_data_path
from ccpi.optimisation.algorithms import CGLS, PDHG, FISTA
from ccpi.optimisation.operators import BlockOperator, LinearOperator
from ccpi.optimisation.functions import KullbackLeibler, IndicatorBox, BlockFunction, MixedL21Norm, ZeroFunction
from ccpi.framework import ImageData
from ccpi.plugins.regularisers import FGP_TV, FGP_dTV


class FISTA_OS(Algorithm):
    
    r'''Fast Iterative Shrinkage-Thresholding Algorithm 
    
    Problem : 
    
    .. math::
    
      \min_{x} f(x) + g(x)
    
    |
    
    Parameters :
        
      :parameter x_init : Initial guess ( Default x_init = 0)
      :parameter f : Differentiable function
      :parameter g : Convex function with " simple " proximal operator


    Reference:
      
        Beck, A. and Teboulle, M., 2009. A fast iterative shrinkage-thresholding 
        algorithm for linear inverse problems. 
        SIAM journal on imaging sciences,2(1), pp.183-202.
    '''
    
    
    def __init__(self, **kwargs):
        
        '''creator 
        
        initialisation can be done at creation time if all 
        proper variables are passed or later with set_up'''
        
        super(FISTA_OS, self).__init__()
        f = kwargs.get('f', None)
        g = kwargs.get('g', ZeroFunction())
        x_init = kwargs.get('x_init', None)

        if x_init is not None and f is not None:
            print(self.__class__.__name__ , "set_up called from creator")
            self.set_up(x_init=x_init, f=f, g=g)

    def set_up(self, x_init, f, g=ZeroFunction()):

        self.y = x_init.copy()
        self.x_old = x_init.copy()
        self.x = x_init.copy()
        self.u = x_init.copy()

        self.f = f
        self.g = g
        if f.L is None:
            raise ValueError('Error: Fidelity Function\'s Lipschitz constant is set to None')
        self.invL = 1/f.L
        self.t = 1
        self.update_objective()
        self.configured = True
            
    def update(self):
        #self.t_old = self.t
        self.t_old = 1
        self.t = 1
        #self.f.gradient(self.y, out=self.u)
        i = numpy.random.randint(0, self.f.num_subsets)
        self.u = self.f.gradient(self.y, i)
        #self.u.__imul__( -self.invL )
        self.u.__imul__( self.invL )
        self.u.__iadd__( self.y )

        self.g.proximal(self.u, self.invL, out=self.x)
        
        self.t = 0.5*(1 + numpy.sqrt(1 + 4*(self.t_old**2)))
        
        self.y = self.x - self.x_old
        self.y.__imul__ ((self.t_old-1)/self.t)
        self.y.__iadd__( self.x )
        
        self.x_old.fill(self.x)

        
    def update_objective(self):
        #self.loss.append(0)  
        self.loss.append( -self.f(self.x) + self.g(self.x) )     
    
    
    
#% go to directory with input files

EXAMPLE = 'SIMULATION'

if EXAMPLE == 'SIMULATION':
    # adapt this path to your situation (or start everything in the relevant directory)
    #os.chdir('/home/sirfuser/Documents/Hackathon4/')
    os.chdir('/Users/me549/Desktop/hackathon4/PET/SimulationData')
    #
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

#%

#% Read in images
image = pet.ImageData(image_header);
image_array=image.as_array()
mu_map = pet.ImageData(attenuation_header);
mu_map_array=mu_map.as_array();

#% Show Emission image
#print('Size of emission: {}'.format(image.shape))

#plt.imshow(image.as_array()[0])
#plt.title('Emission')
#plt.show()

#plt.imshow(mu_map.as_array()[0])
#plt.title('Attenuation')
#plt.show()

#% create acquisition model

#%
am = pet.AcquisitionModelUsingRayTracingMatrix()
# we will increate the number of rays used for every Line-of-Response (LOR) as an example
# (it is not required for the exercise of course)
am.set_num_tangential_LORs(12)
am.set_num_tangential_LORs(5)
templ = pet.AcquisitionData(sinogram_header)
pet.AcquisitionData.set_storage_scheme('memory')
am.set_up(templ,image)

#% simulate some data using forward projection

if EXAMPLE == 'SIMULATION':
    acquired_data=templ
    image.fill(1)
    noisy_data = acquired_data.clone()

elif EXAMPLE == 'SMALL':
    image /= 100
    acquired_data=am.forward(image)
    
    acquisition_array = acquired_data.as_array()

    noisy_data = acquired_data.clone()
    noisy_array=np.random.poisson(acquisition_array).astype('float64')
    print(' Maximum counts in the data: %d' % noisy_array.max())
    noisy_data.fill(noisy_array)

#%
#% Generate a noisy realisation of the data

#noisy_array=np.random.poisson(acquisition_array).astype('float64')
#print(' Maximum counts in the data: %d' % noisy_array.max())
## stuff into a new AcquisitionData object


#noisy_dat
#plt.imshow(noisy_data.as_array()[0,100,:,:])
#plt.title('Noisy Acquisition Data')
#plt.show()

init_image=image.clone()
init_image.fill(.1)

def show_image(it, obj, x):
    plt.clf()
    plt.imshow(x.as_array()[63])
    plt.colorbar()
    plt.show()
    
#%%

def KL_call(self, x):
    return self.get_value(x)
    
setattr(pet.ObjectiveFunction, '__call__', KL_call)
fidelity = pet.PoissonLogLikelihoodWithLinearModelForMeanAndProjData()

fidelity.set_acquisition_model(am)
fidelity.set_acquisition_data(noisy_data)
fidelity.set_num_subsets(1)
fidelity.num_subsets = 1
fidelity.set_up(image)

def show_iterate(it, obj, x):
    plt.imshow(x.as_array()[0])
    plt.colorbar()
    plt.show()
    
    
#%%
fidelity.L = 1000
#regularizer = ZeroFunction()
#regularizer = IndicatorBox(lower=0)

lambdaReg = .005 / fidelity.num_subsets
iterationsTV = 50
tolerance = 1e-5
methodTV = 0
nonnegativity = True
printing = False
device = 'cpu'
#regularizer = FGP_TV(lambdaReg,iterationsTV,tolerance,methodTV,nonnegativity,printing,device)
eta_const = 1e-2
ref_data = mu_map.clone()
regularizer = FGP_dTV(ref_data, lambdaReg,iterationsTV,tolerance,eta_const, methodTV,
                      nonnegativity,device)
                 
x_init = init_image.clone()
fista = FISTA_OS()
fista.set_up(x_init=x_init, f=fidelity, g=regularizer)
fista.max_iteration = 500

#%%
fista.run(50, verbose=True)
    
#%%    
plt.clf()
plt.imshow(fista.x.as_array()[0])
plt.title('iter={}'.format(iteration))
plt.colorbar()
plt.show()

#%%
    recon = pet.OSMAPOSLReconstructor()
    recon.set_objective_function(fidelity)
    recon.set_num_subsets(2)
    recon.set_num_subiterations(20)
    recon.set_input(noisy_data)

    # set up the reconstructor based on a sample image
    # (checks the validity of parameters, sets up objective function
    # and other objects involved in the reconstruction, which involves
    # computing/reading sensitivity image etc etc.)
    print('setting up, please wait...')
    recon.set_up(init_image)
    
    recon.set_current_estimate(init_image)
    
    
    recon.process()
    
    x1 = recon.get_current_estimate()
    