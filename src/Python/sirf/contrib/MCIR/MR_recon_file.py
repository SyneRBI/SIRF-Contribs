
__version__ = '0.1.0'

# import engine module
import sirf.Gadgetron as pMR
import sirf.Reg as pReg

# import further modules
import numpy as np
import matplotlib.pyplot as plt

import sys

from ccpi.optimisation.functions import LeastSquares, L2NormSquared, ZeroFunction, \
                                    IndicatorBox, FunctionOperatorComposition, BlockFunction
from ccpi.optimisation.algorithms import FISTA, CGLS, GradientDescent, PDHG
from ccpi.optimisation.operators import LinearOperator, CompositionOperator, BlockOperator
from ccpi.framework import BlockDataContainer

sys.path.append('/home/sirfuser/devel/buildVM/sources/CCPi-FrameworkPlugins/Wrappers/Python/ccpi/plugins/')
from regularisers import FGP_TV, TGV
from ccpi.framework import DataContainer as cilDataContainer

'''
Define wrapper to allow for TV and TGV for complex data (i.e. real and imaginary part get regularised separately
'''
class cilPluginToSIRFFactory(object):
    '''Factory to create SIRF wrappers for CCPi CIL plugins'''

    @staticmethod
    def getInstance(thetype, **kwargs):
        '''Returns an instance of a CCPi CIL plugin wrapped to work on SIRF DataContainers'''
        obj = thetype(**kwargs)
        orig_prox = obj.proximal
        obj.proximal = cilPluginToSIRFFactory.prox(orig_prox,
                                               obj.__class__.__name__)
        return obj

    @staticmethod
    def prox(method, classname):
        def wrapped(x, sigma, out=None):
            '''Wrapped method'''
            if isinstance(x, pMR.ImageData):
                print("complex implementation")
                # if the data is MR => complex we operate the regulariser
                # only on the real part
                y = x.as_array()
                out_y = method(cilDataContainer(y.real), sigma)
                y.real[:] = out_y.as_array()
                out_y = method(cilDataContainer(y.imag), sigma)
                y.imag[:] = out_y.as_array()
            else:
                y = method(x, sigma)
                y = y.as_array()

            if out is not None:
                out.fill(y)
            else:
                out = x.copy()
                out.fill(y)
            return out

        return wrapped

'''
Set parameters
'''
fpath_output = '/media/sf_SIRF_data/Output/'
fpath_input = '/media/sf_SIRF_data/16_07_21_patient7/raw/'
fpath_par = '/media/sf_SIRF_data/16_07_21_patient7/mr_rec/'
fname_input = 'meas_MID00241_FID69145_Tho_T1_fast_ismrmrd.h5'

# Reconstruction options
# Regularisation for reconstruction of different motion gates: None, tv
reg_ms_fista = 'tv'

# Regularisation for MCIR: None, tv, tgv
reg_mcir_fista = 'tgv'


'''
Load in data and calculate coil sensitivity maps
'''
# %% GO TO MR FOLDER
pMR.AcquisitionData.set_storage_scheme('memory')

filename_full_file = fpath_input + fname_input
acq_data = pMR.AcquisitionData(filename_full_file)
acq_data = pMR.preprocess_acquisition_data(acq_data)
acq_data = pMR.set_grpe_trajectory(acq_data)

# Select first echo
acq_data_echo0 = acq_data.new_acquisition_data(empty=True)
for jnd in range(0, acq_data.number() - 1, 3):
    acq_data_echo0.append_acquisition(acq_data.acquisition(jnd))

acq_data_echo0.sort_by_time()

# Add dcf
kdcf = np.load(fpath_par + 'dcf_all.npy')
acq_data_echo0 = pMR.set_densitycompensation_as_userfloat(acq_data_echo0, kdcf.astype(np.float32))

csm = pMR.CoilSensitivityData()
csm.smoothness = 50
csm.calculate(acq_data_echo0)

# Load indices for motion states
ms_idx = np.load(fpath_par + 'resp_idx.npy')
num_ms = len(ms_idx)

# Load dcf for motion states
kdcf = np.load(fpath_par + 'dcf_resp.npy')

'''
Reconstruct different motion gates
'''
# List of motion state images
im_ms_rec = []

# Index of coronal slice to be visualised
sl_idx = 65

fig, ax = plt.subplots(1, num_ms//2)
plt.setp(ax, xticks=[], yticks=[])
for ms_rec_idx in range(num_ms):
    acq_ms = acq_data_echo0.new_acquisition_data(empty=True)

    # Add motion resolved data
    for jnd in range(len(ms_idx[ms_rec_idx])):
        cacq = acq_data_echo0.acquisition(ms_idx[ms_rec_idx][jnd])
        acq_ms.append_acquisition(cacq)

    # Add dcf
    acq_ms = pMR.set_densitycompensation_as_userfloat(acq_ms, kdcf[:, ms_rec_idx].astype(np.float32))
    acq_ms.sort_by_time()

    # Create acquisition model
    E = pMR.AcquisitionModel(acqs=acq_ms, imgs=csm)
    E.set_coil_sensitivity_maps(csm)

    # Pseudo-inverse
    rec_im = E.adjoint(acq_ms)

    E = pMR.AcquisitionModel(acqs=acq_ms, imgs=rec_im)
    E.set_coil_sensitivity_maps(csm)

    num_it_fista = 10
    x_init = rec_im.clone()
    f = LeastSquares(E, acq_ms, c=1)

    if reg_ms_fista == 'tv':
        G = cilPluginToSIRFFactory.getInstance(FGP_TV, lambdaReg=1e-7, iterationsTV=10,
                                               tolerance=1e-7, methodTV=0, nonnegativity=0,
                                               printing=1, device='cpu')
    elif reg_ms_fista == None:
        G = ZeroFunction()
    else:
        assert 0, 'reg_ms_fista should be None or tv'

    # Run FISTA for least squares
    fista = FISTA(x_init=x_init, f=f, g=G)
    fista.max_iteration = num_it_fista
    fista.update_objective_interval = 2
    fista.run(100, verbose=True)

    im = fista.get_output()

    im_ms_rec.append(im.abs())
    ax[ms_rec_idx // 2].imshow(np.fliplr(np.rot90(np.abs(im.as_array()[sl_idx, :, :]),-1)))
    ax[ms_rec_idx // 2].plot([50, 150], [85, 85], '--w')

fig.savefig(fpath_output + 'fig_fista_ms.png')

'''
Register different motion gates
'''
fig, ax = plt.subplots(1, num_ms//2)
plt.setp(ax, xticks=[], yticks=[])

# Forward motion fields
mf_forward = []
for ind in range(num_ms):
    algo = pReg.NiftyF3dSym()

    # Set up images
    algo.set_reference_image(pReg.NiftiImageData3D(im_ms_rec[ind]))
    algo.set_floating_image(pReg.NiftiImageData3D(im_ms_rec[0]))

    algo.process()
    reg_result = algo.get_output()

    mf_forward.append(algo.get_deformation_field_forward())

    # Test resampler
    resampler = pReg.NiftyResample()
    resampler.set_reference_image(pReg.NiftiImageData3D(im_ms_rec[ind]))
    resampler.set_floating_image(pReg.NiftiImageData3D(im_ms_rec[0]))
    resampler.add_transformation(mf_forward[-1])
    resampler.set_padding_value(0)
    resampler.set_interpolation_type_to_linear()

    im_res = resampler.forward(pReg.NiftiImageData3D(im_ms_rec[0]))
    ax[ind // 2].imshow(np.abs(im_res.as_array()[:, :, sl_idx]))
    ax[ind // 2].plot([50, 150], [85, 85], '--w')

fig.savefig(fpath_output + 'fig_reg_ms.png')

'''
MCIR
'''
# Go through motion states and create k-space
acq_ms = [0] * num_ms
rec_im = [0] * num_ms
E = [0] * num_ms
mf_resampler = [0] * num_ms
for ind in range(num_ms):
    cmidx = ind

    acq_ms[ind] = acq_data_echo0.new_acquisition_data(empty=True)

    # Add motion resolved data
    for jnd in range(len(ms_idx[cmidx])):
        cacq = acq_data_echo0.acquisition(ms_idx[cmidx][jnd])
        acq_ms[ind].append_acquisition(cacq)

    acq_ms[ind].sort_by_time()

    # Create acquisition model
    E_tmp = pMR.AcquisitionModel(acqs=acq_ms[ind], imgs=csm)
    E_tmp.set_coil_sensitivity_maps(csm)

    rec_im[ind] = E_tmp.adjoint(acq_ms[ind])

    E[ind] = pMR.AcquisitionModel(acqs=acq_ms[ind], imgs=rec_im[ind])
    E[ind].set_coil_sensitivity_maps(csm)

    # Create resampler
    mf_resampler[ind] = pReg.NiftyResample()
    mf_resampler[ind].set_reference_image(rec_im[ind])
    mf_resampler[ind].set_floating_image(rec_im[ind])
    mf_resampler[ind].add_transformation(mf_forward[ind])
    mf_resampler[ind].set_padding_value(0)
    mf_resampler[ind].set_interpolation_type_to_linear()


# Set up reconstruction
C = [CompositionOperator(am, res) for am, res in zip(*(E, mf_resampler))]
A = BlockOperator(*C)

# Initial pseudo inverse
acq_ms_block = BlockDataContainer(*acq_ms)
im_xinit = A.adjoint(acq_ms_block)

num_it_fista = 10
f = LeastSquares(A, acq_ms_block, c=1)

if reg_mcir_fista == 'tv':
    G = cilPluginToSIRFFactory.getInstance(FGP_TV, lambdaReg=1e-8, iterationsTV=10,
                                           tolerance=1e-7, methodTV=0, nonnegativity=0,
                                           printing=1, device='cpu')

elif reg_mcir_fista == 'tgv':
    alpha = 1.
    beta = alpha * 2
    lip_const = 12.
    G = cilPluginToSIRFFactory.getInstance(TGV, regularisation_parameter=.01,
                                           LipshitzConstant=lip_const,
                                           alpha1=alpha, alpha2=beta,
                                           iter_TGV=10, torelance=1e-4,
                                           device='cpu')

elif reg_mcir_fista == None:
    G = ZeroFunction()
else:
    assert 0, 'reg_mcir_fista should be None, tv or tgv'

# Run FISTA for least squares
fista = FISTA(x_init=im_xinit, f=f, g=G)
fista.max_iteration = num_it_fista
fista.update_objective_interval = 2
fista.run(100, verbose=True)
fista_mcir = fista.get_output()

fig, ax = plt.subplots(1, 3)
plt.setp(ax, xticks=[], yticks=[])
im_fista_mcir = fista_mcir.as_array()
ax[0].imshow(np.abs(im_fista_mcir[:, :, 96]))
ax[1].imshow(np.rot90(np.abs(im_fista_mcir[:, 110, :]),-1))
ax[2].imshow(np.fliplr(np.rot90(np.abs(im_fista_mcir[sl_idx, :, :]),-1)))
plt.title('FISTA MCIR')
fig.savefig(fpath_output + 'fig_fista_mcir.png')
