import subprocess
import os
import logging
import numpy as np

from stgeorges_utils import change_ismrmrd, to_dicom_folder

from sirf.Gadgetron import AcquisitionData, ImageData
from sirf.Gadgetron import AcquisitionModel
from sirf.Gadgetron import AcquisitionDataProcessor
from sirf.Gadgetron import CartesianGRAPPAReconstructor, FullySampledReconstructor
from sirf.Gadgetron import CoilSensitivityData
from sirf.Gadgetron import preprocess_acquisition_data

from cil.optimisation.functions import LeastSquares
from cil.optimisation.functions import ZeroFunction
from cil.optimisation.algorithms import FISTA, CGLS, GD
from cil.plugins.ccpi_regularisation.functions import FGP_TV
from cil.framework import DataContainer as cilDataContainer
from cil.optimisation.operators import LinearOperator
from cil.optimisation.utilities.callbacks import LogfileCallback
import tempfile

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

command = "siemens_to_ismrmrd"
# data_dir = "/home/jovyan/work/data/"
data_dir = os.path.abspath('/input')
# proc_dir = os.path.join(data_dir, "proc")
proc_dir = tempfile.mkdtemp(prefix="stgeorges_proc_")
# recon_dir = os.path.join(data_dir, "recon")
recon_dir = os.path.abspath('/output')

input_files = [            
                #"meas_MID00614_FID129152_CONVENTIONAL_RECON_SEQD_GF2_AX_RL.dat",
                "meas_MID00619_FID129157_AI_RECON_SEQD_512_GF4_AX_RL.dat"
]

mod_input_files = []
for fname in input_files:
    logger.info(f"Processing file {fname}...")
    file_in = os.path.join(data_dir, fname)
    file_out = os.path.join(proc_dir, os.path.basename(file_in).replace(".dat", ".h5"))
    if os.path.exists(file_out):
        logger.warning(f"Output file {file_out} already exists. Removing it.")
        os.remove(file_out)
    
    # siemens_to_ismrmrd -f meas_MID00614_FID129152_CONVENTIONAL_RECON_SEQD_GF2_AX_RL.dat -o meas_MID00614_FID129152_CONVENTIONAL_RECON_SEQD_GF2_AX_RL.h5 -Z -M
    # https://discord.com/channels/1242028164105109574/1481253905852792962/1481263393733087345

    out = subprocess.run(
        [command, "-f", file_in, "-o", file_out, "-z", "2", "-M"],
        capture_output=True,
        text=True,
    )

    logger.info(out.stdout)
    logger.error(out.stderr)

    # Change ISMRMRD file if needed
    file_out_mod = file_out.replace(".h5", "_mod.h5")
    change_ismrmrd(file_out, file_out_mod, matrixSizeY=512)
    logger.info(f"Modified ISMRMRD file saved as {file_out_mod}")
    mod_input_files.append(file_out_mod)

# reconstruct AI recon

for file in mod_input_files:
    logger.info(f"Reconstructing file {file}...")
    if "AI_RECON" in file:
        # downsampled data
        acq_data_ai = AcquisitionData(file)
        acq_data_ai = preprocess_acquisition_data(acq_data_ai)
        
        csm = CoilSensitivityData()
        csm.smoothness = 100
        csm.calculate(acq_data_ai)

        E = AcquisitionModel(acqs=acq_data_ai, imgs=csm)
        E.set_coil_sensitivity_maps(csm)
        # Use the result of the inverse as our starting point
        x_init = E.inverse(acq_data_ai)

        # We set up our AcquisitionModel
        E = AcquisitionModel(acqs=acq_data_ai, imgs=x_init)
        E.set_coil_sensitivity_maps(csm)

        # Define our objective/loss function as least squares between Ex and y
        f = LeastSquares(E, acq_data_ai, c=1)

        alpha = 0.3
        TV = FGP_TV(alpha=alpha, nonnegativity=False, device='cpu')
        G = TV

        # add logger callback to FISTA
        # lc = LogfileCallback(log_file=os.path.join(recon_dir, "fista_log.txt"))

        # Set up FISTA
        fista = FISTA(initial=x_init.fill(0.0), f=f, g=G)
        fista.update_objective_interval = 5

        # Run FISTA for least squares
        num_iterations = 80
        fista.run(num_iterations)

        to_dicom_folder(
            data=fista.solution.as_array(), 
            foldername=recon_dir, 
            filename_prefix="sirf_recon_" + os.path.basename(file).replace(".h5", ""),
            series_description=f"SIRF recon_{num_iterations} LS+ {alpha} TV"
        )

# delete proc_dir
if os.path.exists(proc_dir):
    logger.info(f"Deleting temporary processing directory {proc_dir}...")
    try:
        for fname in os.listdir(proc_dir):
            os.remove(os.path.join(proc_dir, fname))
        os.rmdir(proc_dir)
        logger.info("Temporary processing directory deleted successfully.")
    except Exception as e:
        logger.error(f"Error deleting temporary processing directory: {e}")