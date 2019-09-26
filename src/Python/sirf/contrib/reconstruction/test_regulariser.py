#%% Initial imports etc
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
import os
import shutil
#import pSTIR as pet
from sirf import STIR as pet

from ccpi.optimisation.algorithms import Algorithm
import numpy


from sirf.Utilities import examples_data_path
from ccpi.optimisation.algorithms import CGLS, PDHG, FISTA
from ccpi.optimisation.operators import BlockOperator, LinearOperator
from ccpi.optimisation.functions import KullbackLeibler, IndicatorBox, \
          BlockFunction, MixedL21Norm, ZeroFunction
from ccpi.framework import ImageData
from ccpi.plugins.regularisers import FGP_TV, FGP_dTV


EXAMPLE = 'SIMULATION'

if EXAMPLE == 'SIMULATION':
    # adapt this path to your situation (or start everything in the relevant directory)
    #os.chdir('/home/sirfuser/Documents/Hackathon4/')
    #os.chdir('/Users/me549/Desktop/hackathon4/PET/SimulationData')
    os.chdir('/mnt/data/CCPPETMR/201909_hackathon/Simulations/PET/SimulationData')
    #
    ##%% copy files to working folder and change directory to where the output files are
    shutil.rmtree('exhale-output',True)
    shutil.copytree('Exhale','exhale-output')
    os.chdir('exhale-output')
    
    attenuation_header = 'pet_dyn_4D_resp_simul_dynamic_0_state_0_attenuation_map.hv'
    attenuation_header = attenuation_header.encode('ascii','replace')
    image_header = attenuation_header
    sinogram_header = 'pet_dyn_4D_resp_simul_dynamic_0_state_0.hs'
    sinogram_header = sinogram_header.encode('ascii', 'replace')

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

am = pet.AcquisitionModelUsingRayTracingMatrix()
# we will increate the number of rays used for every Line-of-Response (LOR) as an example
# (it is not required for the exercise of course)
am.set_num_tangential_LORs(12)
am.set_num_tangential_LORs(5)
templ = pet.AcquisitionData(sinogram_header)
#pet.AcquisitionData.set_storage_scheme('memory')
am.set_up(templ,image)

projected_image = am.adjoint(templ)
result = projected_image.as_array()
    
#%%
L = 0.01
#regularizer = ZeroFunction()
#regularizer = IndicatorBox(lower=0)

lambdaReg = 5. 
iterationsTV = 50
tolerance = 1e-5
methodTV = 0
nonnegativity = True
printing = False
device = 'gpu'
regularizer0 = FGP_TV(lambdaReg,iterationsTV,tolerance,methodTV,nonnegativity,printing,device)
reg0 = regularizer0.proximal(projected_image, 1/L)

eta_const = 1e-2
#ref_data = mu_map.clone()
regularizer1 = FGP_dTV(mu_map, lambdaReg,iterationsTV,tolerance,eta_const,
                      methodTV, nonnegativity, device)

reg1 = regularizer1.proximal(projected_image, 1/L)



fig = plt.figure(figsize=(10,3))
gs = gridspec.GridSpec(2, 4, figure=fig, width_ratios=(1,1,1,1))
figno = 0
sliceno = 65
# first graph
ax = fig.add_subplot(gs[0, figno])
aximg = ax.imshow(result[sliceno])
ax.set_title('original, slice {}'.format(sliceno))



figno += 1
# first graph
ax = fig.add_subplot(gs[0, figno])
aximg = ax.imshow(reg0.as_array()[sliceno])
ax.set_title('TV, tau {}, slice {}'.format(1/L, sliceno))

figno += 1
# first graph
ax = fig.add_subplot(gs[0, figno])
aximg = ax.imshow(mu_map.as_array()[sliceno])
ax.set_title('mu_map, slice {}'.format(sliceno))

figno += 1
# first graph
ax = fig.add_subplot(gs[0, figno])
aximg = ax.imshow(reg1.as_array()[sliceno])
ax.set_title('dTV, tau {}, slice {}'.format(1/L, sliceno))

####
figno = 1
# first graph
ax = fig.add_subplot(gs[1, figno])
aximg = ax.imshow((projected_image-reg0).as_array()[sliceno])
ax.set_title('orig - TV'.format(1/L, sliceno))

figno = 3
# first graph
ax = fig.add_subplot(gs[1, figno])
aximg = ax.imshow((projected_image-reg1).as_array()[sliceno])
ax.set_title('orig - dTV'.format(sliceno))

figno = 2
# first graph
ax = fig.add_subplot(gs[1, figno])
aximg = ax.imshow((reg1-reg0).as_array()[sliceno])
ax.set_title('regularisers difference'.format(1/L, sliceno))

# adjust spacing between plots
fig.tight_layout() 
#plt.subplots_adjust(wspace=0.4)
plt.show()
