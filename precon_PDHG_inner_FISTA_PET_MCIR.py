"""MCIR for PET

Usage:
  PET_MCIR [--help | options]

Options:
  -T <pattern>, --trans=<pattern>     transformation pattern, * or % wildcard (e.g., tm_ms*.txt). Enclose in quotations.
  -t <str>, --trans_type=<str>        transformation type (tm, disp, def) [default: tm]
  -S <pattern>, --sino=<pattern>      sinogram pattern, * or % wildcard (e.g., sino_ms*.hs). Enclose in quotations.
  -a <pattern>, --attn=<pattern>      attenuation pattern, * or % wildcard (e.g., attn_ms*.hv). Enclose in quotations.
  -R <pattern>, --rand=<pattern>      randoms pattern, * or % wildcard (e.g., rand_ms*.hs). Enclose in quotations.
  -n <norm>, --norm=<norm>            ECAT8 bin normalization file
  -i <int>, --iter=<int>              num iterations [default: 10]
  -r <string>, --reg=<string>         regularisation ("none","FGP_TV", ...) [default: none]
  -o <outp>, --outp=<outp>            output file prefix [default: recon]
  -d <nxny>, --nxny=<nxny>            image x and y dimensions as string '(nx,ny)'
                                      (no space after comma) [default: (127,127)]
  -I <str>, --initial=<str>           Initial estimate
  --visualisations                    show visualisations
  --nifti                             save output as nifti
  --gpu                               use GPU projector
  -v <int>, --verbosity=<int>         STIR verbosity [default: 0]
  -s <int>, --save_interval=<int>     save every x iterations [default: 10]
"""

## CCP PETMR Synergistic Image Reconstruction Framework (SIRF)
## Copyright 2020 University College London.
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

import os
from ast import literal_eval
from glob import glob
from docopt import docopt
from sirf.Utilities import error, show_2D_array
import pylab
import sirf.Reg as reg
import sirf.STIR as pet
from ccpi.optimisation.algorithms import PDHG
from ccpi.optimisation.functions import KullbackLeibler, BlockFunction, IndicatorBox
from ccpi.optimisation.operators import CompositionOperator, BlockOperator, LinearOperator
from ccpi.plugins.regularisers import FGP_TV
from ccpi.filters import regularisers
import numpy as np


pet.AcquisitionData.set_storage_scheme('memory')

__version__ = '0.1.0'
args = docopt(__doc__, version=__version__)


def file_exists(filename):
    """Check if file exists, optionally throw error if not"""
    return os.path.isfile(filename)


def check_file_exists(filename):
    """Check file exists, else throw error"""
    if not file_exists:
        raise error('File not found: %s' % filename)


# Multiple files
trans_pattern = str(args['--trans']).replace('%','*')
sino_pattern = str(args['--sino']).replace('%','*')
attn_pattern = str(args['--attn']).replace('%','*')
rand_pattern = str(args['--rand']).replace('%','*')
num_iters = int(args['--iter'])
regularisation = str(args['--reg'])
trans_type = str(args['--trans_type'])

if attn_pattern is None:
    attn_pattern = ""
if rand_pattern is None:
    rand_pattern = ""

# Norm
norm_file = None
if args['--norm']:
    norm_file = str(args['--norm'])
    if not os.path.isfile(norm_file):
        raise error("Norm file not found: " + norm_file)

# Number of voxels
nxny = literal_eval(args['--nxny'])

# Output file
outp_prefix = str(args['--outp'])

# Initial estimate
initial_estimate = None
if args['--initial']:
    initial_estimate = str(args['--initial'])

visualisations = True if args['--visualisations'] else False
nifti = True if args['--nifti'] else False
use_gpu = True if args['--gpu'] else False

# Verbosity
pet.set_verbosity(int(args['--verbosity']))

# Verbosity
save_interval = int(args['--save_interval'])


def get_resampler(image, ref=None, trans=None):
    """returns a NiftyResample object for the specified transform and image"""
    if ref is None:
        ref = image
    resampler = reg.NiftyResample()
    resampler.set_reference_image(ref)
    resampler.set_floating_image(image)
    resampler.set_padding_value(0)
    resampler.set_interpolation_type_to_linear()
    if trans is not None:
        resampler.add_transformation(trans)
    return resampler


def get_asm_attn(sino, attn, acq_model):
    """Get attn ASM from sino, attn image and acq model"""
    asm_attn = pet.AcquisitionSensitivityModel(attn, acq_model)
    # temporary fix pending attenuation offset fix in STIR:
    # converting attenuation into 'bin efficiency'
    asm_attn.set_up(sino)
    bin_eff = pet.AcquisitionData(sino)
    bin_eff.fill(1.0)
    asm_attn.unnormalise(bin_eff)
    asm_attn = pet.AcquisitionSensitivityModel(bin_eff)
    return asm_attn


def main():

    ############################################################################################
    # Parse input files
    ############################################################################################


    if sino_pattern is None:
        raise AssertionError("--sino missing")
    sino_files = sorted(glob(sino_pattern))
    attn_files = sorted(glob(attn_pattern))
    rand_files = sorted(glob(rand_pattern))

    num_ms = len(sino_files)
    # Check some sinograms found
    if num_ms == 0:
        raise AssertionError("No sinograms found!")
    # If any rand, check num == num_ms
    if len(rand_files) > 0 and len(rand_files) != num_ms:
        raise AssertionError("#rand should match #sinos. "
                             "#sinos = " + str(num_ms) + ", #rand = " + str(len(rand_files)))

    # For attn, there should be 0, 1 or num_ms images
    if len(attn_files) != 0 and len(attn_files) != 1 and len(attn_files) != num_ms:
        raise AssertionError("#attn should be 0, 1 or #sinos")
    
    ############################################################################################
    # Read input
    ############################################################################################

    sinos_raw = [pet.AcquisitionData(file) for file in sino_files]
    attns = [pet.ImageData(file) for file in attn_files]
    rands = [pet.AcquisitionData(file) for file in rand_files]

    # If any sinograms contain negative values (shouldn't be the case), set them to 0
    sinos = [0]*num_ms
    for ind in range(num_ms):
        if (sinos_raw[ind].as_array() < 0).any():
            print("Input sinogram " + str(ind) + " contains -ve elements. Setting to 0...")
            sinos[ind] = sinos_raw[ind].clone()
            sino_arr = sinos[ind].as_array()
            sino_arr[sino_arr < 0] = 0
            sinos[ind].fill(sino_arr)
        else:
            sinos[ind] = sinos_raw[ind]

    ############################################################################################
    # Initialise recon image
    ############################################################################################

    if initial_estimate:
        image = pet.ImageData(initial_estimate)
    else:
        image = sinos[0].create_uniform_image(0.0, nxny)
        # If using GPU, need to make sure that image is right size.
        if use_gpu:
            image.initialise(dim=(127,320,320), vsize=(2.03125,2.08626,2.08626))
            image.fill(0.0)

    ############################################################################################
    # Set up resamplers
    ############################################################################################

    if trans_pattern == "None":
        print("No transformations given")
        resamplers = None
    else:
        trans_files = sorted(glob(trans_pattern))
        # Should have as many trans as sinos
        if num_ms != len(trans_files):
            raise AssertionError("#trans should match #sinos. "
                                 "#sinos = " + str(num_ms) + ", #trans = " + str(len(trans_files)))

        if trans_type == "tm":
            trans = [reg.AffineTransformation(file) for file in trans_files]
        elif trans_type == "disp":
            trans = [reg.NiftiImageData3DDisplacement(file) for file in trans_files]
        elif trans_type == "def":
            trans = [reg.NiftiImageData3DDeformation(file) for file in trans_files]
        else:
            raise error("Unknown transformation type")

        # Set up the resamplers
        resamplers = [get_resampler(image, trans=tran) for tran in trans]

    ############################################################################################
    # Resample attenuation images (if necessary)
    ############################################################################################

    if use_gpu:
        for i in range(len(attns)):
            resam = get_resampler(attns[i], ref=image)
            attns[i] = resam.forward(attns[i])

    ############################################################################################
    # Set up acquisition models
    ############################################################################################

    print("Setting up acquisition models...")
    if not use_gpu:
        acq_models = num_ms * [pet.AcquisitionModelUsingRayTracingMatrix()]
    else:
        acq_models = num_ms * [pet.AcquisitionModelUsingNiftyPET()]

    # If present, create ASM from ECAT8 normalisation data
    asm_norm = None
    if norm_file:
        asm_norm = pet.AcquisitionSensitivityModel(norm_file)

    # Loop over each motion state
    for ind in range(num_ms):
        # Create attn ASM if necessary
        asm_attn = None
        if len(attns) == num_ms:
            asm_attn = get_asm_attn(sinos[ind], attns[ind], acq_models[ind])
        elif len(attns) == 1:
            print("Resampling attn im " + str(ind) + " into required motion state...")
            resampler = get_resampler(attns[0], trans=trans[ind])
            resampled_attn = resampler.forward(attns[0])
            asm_attn = get_asm_attn(sinos[ind], resampled_attn, acq_models[ind])

        # Get ASM dependent on attn and/or norm
        asm = None
        if asm_norm and asm_attn:
            print("AcquisitionSensitivityModel contains norm and attenuation...")
            asm = pet.AcquisitionSensitivityModel(asm_norm, asm_attn)
        elif asm_norm:
            print("AcquisitionSensitivityModel contains norm...")
            asm = asm_norm
        elif asm_attn:
            print("AcquisitionSensitivityModel contains attenuation...")
            asm = asm_attn
        if asm:
            print("Setting AcquisitionSensitivityModel...")
            acq_models[ind].set_acquisition_sensitivity(asm)

        if len(rands) > 0:
            acq_models[ind].set_background_term(rands[ind])
          
        # Set up
        acq_models[ind].set_up(sinos[ind], image)

    ############################################################################################
    # Set up reconstructor
    ############################################################################################

    
    print("Setting up reconstructor...")

    # If there is motion, create composition operators containing acquisition models and resamplers
    if resamplers != None:
        C = [ CompositionOperator(am, res, preallocate=True) for am, res in zip (*(acq_models, resamplers)) ]
    else:
        C = acq_models
                            
    # Configure the PDHG algorithm
    kl = [ KullbackLeibler(b = sino*(1./3874.37388490063), eta=(sino * 0 + 1e-5)) for sino in sinos ] 
    f = BlockFunction(*kl)
    K = BlockOperator(*C)*(1./3874.37388490063)
    precond = True  

    if precond:

        tmp_tau = K.adjoint(K.range_geometry().allocate(1)) 
        tau = 0.*tmp_tau
        tmp_tau_np = 1./tmp_tau.as_array()
        tmp_tau_np[tmp_tau_np==np.inf]=1    
        tau.fill(tmp_tau_np)

        tmp_sigma = K.direct(K.domain_geometry().allocate(1)) 
        print(tmp_sigma.shape)
        print(tmp_sigma[0])
        sigma = 0.*tmp_sigma
        tmp_sigma_np = 1./tmp_sigma[0].as_array()       
        tmp_sigma_np[tmp_sigma_np==np.inf]=1
        sigma[0].fill(tmp_sigma_np)

        def precond_proximal(self, x, tau, out=None):
            pars = {'algorithm' : FGP_TV, \
                    'input' : np.asarray(x.as_array()/tau.as_array(), dtype=np.float32),\
                    'regularization_parameter':self.lambdaReg, \
                    'number_of_iterations' :self.iterationsTV ,\
                    'tolerance_constant':self.tolerance,\
                    'methodTV': self.methodTV ,\
                    'nonneg': self.nonnegativity ,\
                    'printingOut': self.printing}

            res , info = regularisers.FGP_TV(pars['input'], 
                      pars['regularization_parameter'],
                      pars['number_of_iterations'],
                      pars['tolerance_constant'], 
                      pars['methodTV'],
                      pars['nonneg'],
                      self.device)
            if out is not None:
                out.fill(res)
            else:
                out = x.copy()
                out.fill(res)
            out*=tau
            print("run with precond")        
            return out 

        setattr(FGP_TV, 'proximal', precond_proximal) 

    else:
        # precomputed, for 10 iterations
        normK = 3874.37388490063
        # normK = LinearOperator.PowerMethod(K, iterations=10)[0]
#         sigma = 1/normK
#         tau = 1/normK 
        sigma = 1.
        tau = 1.
#         sigma = 1e-5
#         tau = 1/(sigma*normK**2)   
              
    if regularisation == 'none':
        G = IndicatorBox(lower=0)
    elif regularisation == 'FGP_TV':
        r_alpha = 0.0001
        r_iterations = 200
        r_tolerance = 1e-7          
        r_iso = 0
        r_nonneg = 1
        r_printing = 0
        G = FGP_TV(r_alpha, r_iterations, r_tolerance, r_iso, r_nonneg, r_printing, 'gpu')
    else:
        raise error("Unknown regularisation")
         
    def PDHG_new_update(self):
        
        # save previous iteration
        self.x_old.fill(self.x)
        self.y_old.fill(self.y)

        # Gradient ascent for the dual variable
        self.operator.direct(self.xbar, out=self.y_tmp)     
        self.y_tmp *= self.sigma
        self.y_tmp += self.y_old                

        self.f.proximal_conjugate(self.y_tmp, self.sigma, out=self.y)
        
        # Gradient descent for the primal variable
        self.operator.adjoint(self.y, out=self.x_tmp)
        self.x_tmp *= -1*self.tau
        self.x_tmp += self.x_old
        
        self.g.proximal(self.x_tmp, self.tau, out=self.x)

        # Update
        self.x.subtract(self.x_old, out=self.xbar)
        self.xbar *= self.theta
        self.xbar += self.x     
    
    if precond:
        setattr(PDHG, 'update', PDHG_new_update)  
      
    
    pdhg = PDHG(f=f, g=G, operator=K, sigma=sigma, tau=tau,
                max_iteration = 250,
                update_objective_interval=50,
                x_init = image)

    # Get filename
    outp_file = outp_prefix
    if len(attn_files) > 0:
        outp_file += "_wAC"
    if norm_file:
        outp_file += "_wNorm"
    outp_file += "_Reg-" + regularisation
    if regularisation == 'FGP_TV':
        outp_file += "-alpha" + str(r_alpha)
        outp_file += "-sigma" + str(sigma)
        outp_file += "-tau" + str(tau)
    outp_file += "_nGates" + str(len(sino_files))
    outp_file +="normalise_am"
    if resamplers == None:
        outp_file += "_noMotion"

    for i in range(1, num_iters+1, save_interval):
        pdhg.run(save_interval, very_verbose=True)
        out = pdhg.get_output()    
        if not nifti:
            out.write(outp_file + "_iters" + str(i))
        else:
            reg.NiftiImageData(out).write(outp_file + "_iters" + str(i))

    if visualisations:
        # show reconstructed image
        out_arr = out.as_array()
        z = out_arr.shape[0]//2
        show_2D_array('Reconstructed image', out.as_array()[z, :, :])
        pylab.show()


# if anything goes wrong, an exception will be thrown 
# (cf. Error Handling section in the spec)
try:
    main()
    print('done')
except error as err:
    # display error information
    print('%s' % err.value)
