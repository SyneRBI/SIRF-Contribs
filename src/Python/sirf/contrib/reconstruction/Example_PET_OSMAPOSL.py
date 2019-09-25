#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 25 14:37:32 2019

@author: edo
"""
#%% Initial imports etc
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
import matplotlib.pyplot as plt
import os
import shutil
#import pSTIR as pet
from sirf import STIR as pet

from ccpi.optimisation.algorithms import Algorithm
from ccpi.optimisation.functions import ZeroFunction
import numpy


from sirf.Utilities import examples_data_path
from ccpi.optimisation.algorithms import CGLS, PDHG, FISTA
from ccpi.optimisation.operators import BlockOperator, LinearOperator
from ccpi.optimisation.functions import KullbackLeibler, IndicatorBox, \
          BlockFunction, MixedL21Norm, ZeroFunction
from ccpi.framework import ImageData
#from ccpi.plugins.regularisers import FGP_TV, FGP_dTV
from ccpi.plugins.regularisers import FGP_TV
    
    
#% go to directory with input files

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

image = pet.ImageData(image_header);
image_array=image.as_array()
mu_map = pet.ImageData(attenuation_header);
mu_map_array=mu_map.as_array();


#% create acquisition model

am = pet.AcquisitionModelUsingRayTracingMatrix()
# we will increate the number of rays used for every Line-of-Response (LOR) as an example
# (it is not required for the exercise of course)
am.set_num_tangential_LORs(12)
am.set_num_tangential_LORs(5)
templ = pet.AcquisitionData(sinogram_header)
am.set_up(templ,image)

noisy_data = templ.clone()
fidelity = pet.PoissonLogLikelihoodWithLinearModelForMeanAndProjData()
fidelity.set_acquisition_model(am)
fidelity.set_acquisition_data(noisy_data)
fidelity.set_num_subsets(1)
fidelity.num_subsets = 1
fidelity.set_up(image)


if True:
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
    
    init_image=image.clone()
    init_image.fill(.1)
    recon.set_up(init_image)
    

    recon.set_current_estimate(init_image)
    
    
    recon.process()
    
    x1 = recon.get_current_estimate()
    fname = 'OSMAPOSL_rec.s'.encode('ascii', 'replace') 
    saveto = os.path.join(os.getcwd(), fname)
    print("saving to {}".format(saveto))
    x1.write(saveto)

