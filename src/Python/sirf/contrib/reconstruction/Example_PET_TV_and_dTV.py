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
#from ccpi.plugins.regularisers import FGP_TV


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
        self.t_old = self.t
        self.x_old.fill(self.x)
        #self.t_old = 1
        #self.t = 1
        #self.f.gradient(self.y, out=self.u)
        #i = numpy.random.randint(0, self.f.num_subsets)
        i = 0
        self.u = self.f.gradient(self.y, i)
        # negative gradient is returned from STIR
        #self.u.__imul__( -self.invL )
        self.u.__imul__( self.invL )
        self.u.__iadd__( self.y )

        #self.g.proximal(self.u, self.invL, out=self.x)
        self.x = self.g.proximal(self.u, self.invL )
        
        self.t = 0.5*(1 + numpy.sqrt(1 + 4*(self.t_old**2)))
        
        #self.y = self.x - self.x_old
        #self.y.__imul__ ((self.t_old-1)/self.t)
        #self.y.__iadd__( self.x )
        self.x.subtract(self.x_old, out=self.y)
        self.y.multiply((self.t_old-1)/self.t, out=self.y)
        self.y.add(self.x, out=self.y)
        #self.y = self.y * ((self.t_old-1)/self.t) + self.x
        

        
    def update_objective(self):
        #self.loss.append(0)  
        # negative objective is returned from STIR
        self.loss.append( -self.f(self.x) + self.g(self.x) )     
    
    
    
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
#pet.AcquisitionData.set_storage_scheme('memory')
am.set_up(templ,image)

# see test operator passes dot_test
assert LinearOperator.dot_test(am) == True

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
fidelity.L = 1e4
#regularizer = ZeroFunction()
#regularizer = IndicatorBox(lower=0)

# l = lambdaReg / fidelity.L
# lambdaReg = l * fidelity.L = 1e-4 * 1e4 = 1e0
l = 1e2
lambdaReg = l * fidelity.L / fidelity.num_subsets
#lambdaReg = 1e0 / fidelity.num_subsets
iterationsTV = 100
tolerance = 1e-5
methodTV = 0
nonnegativity = True
printing = False
device = 'gpu'
regularizer = FGP_TV(lambdaReg,iterationsTV,tolerance,methodTV,nonnegativity,printing,device)
eta_const = 1e-2
#regularizer = ZeroFunction()

#regularizer = ZeroFunction()
if False:
    ref_data = mu_map.clone()
    regularizer = FGP_dTV(ref_data, lambdaReg,iterationsTV,tolerance,eta_const,
                      methodTV, nonnegativity, device)
                 
# regularizer = ZeroFunction()
x_init = init_image * 0.
fista = FISTA_OS()
fista.set_up(x_init=x_init, f=fidelity, g=regularizer)
fista.max_iteration = 2

#%%
last_result = x_init.as_array()

def show_slices(iteration, objective, solution):
    result = solution.as_array()
    fig = plt.figure(figsize=(10,3))
    gs = gridspec.GridSpec(1, 4, figure=fig, width_ratios=(1,1,1,0.2))

    figno = 0
    sliceno = 45
    # first graph
    ax = fig.add_subplot(gs[0, figno])
    aximg = ax.imshow(result[sliceno])
    ax.set_title('iter={}, slice {}'.format(iteration, sliceno))

    figno += 1
    sliceno += 10
    # first graph
    ax = fig.add_subplot(gs[0,figno])
    aximg = ax.imshow(result[sliceno])
    ax.set_title('iter={}, slice {}'.format(iteration, sliceno))

    figno += 1
    sliceno += 10
    # first graph
    ax = fig.add_subplot(gs[0,figno])
    aximg = ax.imshow(result[sliceno])
    ax.set_title('iter={}, slice {}'.format(iteration, sliceno))



    # colorbar
    axes = fig.add_subplot(gs[0,figno+1])
    plt.colorbar(aximg, cax=axes)

    # adjust spacing between plots
    fig.tight_layout() 
    #plt.subplots_adjust(wspace=0.4)
    last_result = result
    plt.show()

interactive_plot = False
import multiprocessing
class ProcessLinePlotter(object):
    '''from https://matplotlib.org/3.1.0/gallery/misc/multiprocess_sgskip.html'''
    
    def __init__(self):
        self.x = []
        self.y = []
    def __call__(self, pipe):
        '''configure on call'''
        print ("Starting LinePlotter")
        self.pipe = pipe
        self.fig , self.ax = plt.subplots(1,2)
        timer = self.fig.canvas.new_timer(interval=1000)
        timer.add_callback(self.call_back)
        timer.start()
        print ('Done')
        plt.show()

    def terminate(self):
        '''terminate the process'''
        plt.close('all')
    def call_back(self):
        '''callback to plot to the canvas'''
        while self.pipe.poll():
            command = self.pipe.recv()
            if command is None:
                self.terminate()
                return False
            else:
                # save the data for plotting
                self.x.append(command[0])
                self.y.append(command[1])
                img = command[2]
                #y = [el/self.y[0] for el in self.y]
                self.ax[0].imshow(img)
                #plt.colorbar(img, cax=self.ax[1])
                self.ax[1].plot(self.x, self.y, 'r-')
                self.ax[0].set_title('min={:.3e}, max={:.3e}'.format(img.min(), img.max()))

                self.ax[1].set_title('iter={}, Obj {}'.format(len(self.x), self.y[-1]))

                #self.fig.colorbar()
        self.fig.canvas.draw()
        return True


if interactive_plot:
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    #plt.ion()
    #im = fig.add_subplot(122)
    #im.imshow(abs(x_init.as_array()[0]))
    fig.show()
    fig.canvas.draw()
    plt.show()

    for _ in fista:
        if fista.iteration >= 5:
            break
        ax.clear()
        #pixval.append( gd.get_output().as_array()[0][46][160])
        #print ("\rIteration {} Loss: {} pix {}".format(fista.iteration, 
        #       gd.get_last_loss(), pixval[-1]/gadgval))
        ax.semilogy([val/fista.loss[0] for val in fista.loss])
        #im.imshow(abs(gd.get_output().as_array()[0]))
        fig.canvas.draw()
else:
    #fista.run(5, verbose=False, callback=show_slices)
    plotter = ProcessLinePlotter()
    # parent, child
    plot_pipe, plotter_pipe = multiprocessing.Pipe()
    # attach the child pipe to the process 
    plot_process = multiprocessing.Process(target=plotter, 
              args=(plotter_pipe,))
    # start the process
    plot_process.start()
    iterations = 1
    fista.max_iteration -= 1
    while iterations > 0:
        fista.max_iteration += iterations
        for _ in fista:
            #show_slices(fista.iteration, 0, fista.get_output()-fista.x_old)   
            plot_pipe.send((fista.iteration, fista.get_last_objective(),fista.get_output().as_array()[64]))
        iterations = input("Run more iterations? Specify how many. Zero to stop: ")
    # close the plot
    plot_pipe.send(None)
    



#%%

fname = "FISTA_L_{:.2e}_l_{:.2e}_it_{}.h".format(fidelity.L, 0, fista.iteration).encode('ascii', 'replace')
fista.get_output().write(fname)

result = fista.get_output().as_array()

fig = plt.figure(figsize=(10,3))
gs = gridspec.GridSpec(1, 4, figure=fig, width_ratios=(1,1,1,0.2))

figno = 0
sliceno = 45
# first graph
ax = fig.add_subplot(gs[0, figno])
aximg = ax.imshow(result[sliceno])
ax.set_title('iter={}, slice {}'.format(fista.iteration, sliceno))

figno += 1
sliceno += 10
# first graph
ax = fig.add_subplot(gs[0,figno])
aximg = ax.imshow(result[sliceno])
ax.set_title('iter={}, slice {}'.format(fista.iteration, sliceno))

figno += 1
sliceno += 10
# first graph
ax = fig.add_subplot(gs[0,figno])
aximg = ax.imshow(result[sliceno])
ax.set_title('iter={}, slice {}'.format(fista.iteration, sliceno))



# colorbar
axes = fig.add_subplot(gs[0,figno+1])
plt.colorbar(aximg, cax=axes)

# adjust spacing between plots
fig.tight_layout() 
#plt.subplots_adjust(wspace=0.4)
plt.show()

fname = "FISTA_reg_L{}_it{}".format(fidelity.L, fista.iteration)
saveto = os.path.join(os.getcwd(), fname)
#plt.savefig(saveto)

#%%
#%matplotlib inline    
#plt.clf()

#fig = plt.figure(figsize=(10,30))
#figno = 1
#sliceno = 45
#ax = fig.add_subplot(1,4,figno)
#aximg = ax.imshow(result[sliceno])
#ax.set_title('iter={}, slice {}'.format(fista.iteration, sliceno))
#aximg.set_clim(clim.min(), clim.max())
##plt.colorbar()
#
#sliceno += 10 
#figno += 1
#ax = fig.add_subplot(1,4,figno)
#aximg = ax.imshow(result[sliceno])
#ax.set_title('iter={}, slice {}'.format(fista.iteration, sliceno ))
#aximg.set_clim(clim.min(), clim.max())
##plt.colorbar()
#
#
#sliceno += 10 
#figno += 1
#ax = fig.add_subplot(1,4,figno)
#aximg = ax.imshow(result[sliceno])
#ax.set_title('iter={}, slice {}'.format(fista.iteration, sliceno ))
#aximg.set_clim(clim.min(), clim.max())
#
#
#figno+=1
#ax = fig.add_subplot(1,4,figno)
##ax.subplots_adjust(bottom=0.2, right=0.8, top=0.9)
##cax = plt.axes([0.85, 0.1, 0.075, 0.8])
#plt.colorbar(aximg, cax=ax)
#
#plt.show()

#%%
if False:
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
    
