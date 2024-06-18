import os
from sirf import STIR as pet
from sirf.contrib.partitioner import partitioner

# engine's messages go to files, except error messages, which go to stdout
_ = pet.MessageRedirector('info.txt', 'warn.txt')
# Needed for get_subsets()
pet.AcquisitionData.set_storage_scheme('memory')
# fewer message from STIR and SIRF
pet.set_verbosity(0)

def initial_OSEM(acquired_data, additive_term, mult_factors, initial_image):
    num_subsets = 1
    data, acq_models, obj_funs = partitioner.data_partition(acquired_data, additive_term, mult_factors, num_subsets)

    obj_fun = pet.make_Poisson_loglikelihood(data[0])
    obj_fun.set_acquisition_model(acq_models[0])
    recon = pet.OSMAPOSLReconstructor()
    recon.set_objective_function(obj_fun)
    recon.set_current_estimate(initial_image)
    # some arbitrary numbers here
    recon.set_num_subsets(2)
    num_subiters = 14
    recon.set_num_subiterations(num_subiters)
    recon.set_up(initial_image)
    recon.process()
    return recon.get_output()

def construct_RDP(penalty_strength, initial_image, kappa, max_scaling=1e-3):
    '''
    Construct the Relative Difference Prior (RDP)
    
    WARNING: return prior with beta/num_subsets (as currently needed for BSREM implementations)
    '''
    prior = pet.RelativeDifferencePrior()
    # need to make it differentiable
    epsilon = initial_image.max() * max_scaling
    prior.set_epsilon(epsilon)
    prior.set_penalisation_factor(penalty_strength)
    prior.set_kappa(kappa)
    prior.set_up(initial_image)
    return prior
    
def add_prior(prior, objective_functions):
    '''Add prior evenly to every objective function.
    
    WARNING: it modifies the objective functions'''
    for f in objective_functions:
        f.set_prior(prior)


# DATA LOADING 
os.chdir('/home/jovyan/work/Challenge24/data')

acquired_data = pet.AcquisitionData('prompts.hs')

additive_term = pet.AcquisitionData('additive.hs')

mult_factors = pet.AcquisitionData('multfactors.hs')

initial_image = pet.ImageData('OSEM_image.hv')
osem_sol = initial_image
# This should be an image to give voxel-dependent weights 
# (here predetermined as the row-sum of the Hessian of the log-likelihood at an initial OSEM reconstruction, see eq. 25 in [7])
kappa = initial_image.allocate(1.)


# DATA SPLITTING
num_subsets = 7
data, acq_models, obj_funs = partitioner.data_partition(acquired_data, additive_term, mult_factors, num_subsets, mode='staggered', initial_image=initial_image)

# OPTIMISATION PROBLEM CREATION

# add RDP prior to the objective functions
step_size = 1e-7
add_regulariser = True
if add_regulariser:
    alpha = 500
    prior = construct_RDP(alpha, initial_image, kappa)
    add_prior(prior, obj_funs)
    step_size = 1e-10

# EXAMPLE ALGORITHM
#set up and run the gradient descent algorithm
from cil.optimisation.functions import SGFunction
from cil.optimisation.algorithms import GD
from cil.optimisation.utilities import Sampler
from cil.optimisation.utilities.callbacks import TextProgressCallback, ProgressCallback
from cil.utilities.display import show2D 


sampler = Sampler.random_without_replacement(len(obj_funs))
# requires a minus sign for CIL's algorithm as they are minimisers
F = - SGFunction(obj_funs, sampler=sampler)
alg = GD(initial=initial_image, objective_function=F, step_size=step_size)
alg.update_objective_interval = num_subsets
alg.run(num_subsets * 1, callbacks=[TextProgressCallback()])

# DISPLAy THE RESULTS
cmax = .15
im_slice = 70
osem_sol = initial_image
show2D([osem_sol.as_array()[im_slice,:,:], 
        alg.solution.as_array()[im_slice,:,:]], 
       title=['OSEM',f"SGD epoch {alg.iteration/num_subsets}"], cmap="PuRd", fix_range=[(0, 0.2),(0,0.2)])