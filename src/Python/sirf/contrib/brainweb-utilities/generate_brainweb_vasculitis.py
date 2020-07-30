"""Generate brainweb data with two simulated temporal arteries

Usage:
  generate_brainweb_vasculitis [--help | options]

Options:
  -i <path>, --out_im=<path>  output image filename [default: im]
  --iIL=<val>                 inner intensity (left) [default: 10]
  --iIR=<val>                 inner intensity (right) [default: 20]
  --oIL=<val>                 outer intensity (left) [default: 50]
  --oIR=<val>                 outer intensity (right) [default: 80]
  --iRL=<val>                 inner radius (left) [default: 3]
  --iRR=<val>                 inner radius (right) [default: 3]
  --oRL=<val>                 outer radius (left) [default: 5]
  --oRR=<val>                 outer radius (right) [default: 5]
  --cL=<val>                  centre (left) [default: -80]
  --cR=<val>                  centre (left) [default: 80]
"""

# CCP PETMR Synergistic Image Reconstruction Framework (SIRF)
# Copyright 2020 University College London.
#
# This is software developed for the Collaborative Computational
# Project in Positron Emission Tomography and Magnetic Resonance imaging
# (http://www.ccppetmr.ac.uk/).
#
# Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#       http://www.apache.org/licenses/LICENSE-2.0
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.

from numba import jit
import brainweb
import numpy as np
from tqdm.auto import tqdm
import sirf.STIR as pet
import sirf.Reg as reg
from sirf.Utilities import examples_data_path
from docopt import docopt

__version__ = '0.1.0'
args = docopt(__doc__, version=__version__)

# Parse input arguments
out_prefix = args['--out_im']
iIL = float(args['--iIL'])
iIR = float(args['--iIR'])
oIL = float(args['--oIL'])
oIR = float(args['--oIR'])
iRL = float(args['--iRL'])
iRR = float(args['--iRR'])
oRL = float(args['--oRL'])
oRR = float(args['--oRR'])
cL = float(args['--cL'])
cR = float(args['--cR'])


def get_brainweb_image():
    """Get brainweb image."""
    fname, url = sorted(brainweb.utils.LINKS.items())[0]
    files = brainweb.get_file(fname, url, ".")
    data = brainweb.load_file(fname)

    brainweb.seed(1337)

    for f in tqdm([fname], desc="mMR ground truths", unit="subject"):
        vol = brainweb.get_mmr_fromfile(
            f, petNoise=0, petSigma=0)

    return vol['PET']


def crop_and_orient(im):
    """Crop and orient image."""
    im = im[:, 105:105+127, 108:108+127]
    im = np.flip(im, 0)
    return im


def get_as_pet_im(arr):
    """Get as PET image."""
    # We'll need a template sinogram
    mMR_template_sino = \
        examples_data_path('PET') + "/mMR/mMR_template_span11.hs"
    templ_sino = pet.AcquisitionData(mMR_template_sino)
    im = pet.ImageData(templ_sino)
    dim = (127, 127, 127)
    voxel_size = im.voxel_sizes()
    im.initialise(dim, voxel_size)
    im.fill(arr)
    return im


def save_nii(im, fname):
    """Save as nii."""
    reg.ImageData(im).write(fname)


def get_cylinder_in_im(im_in, length, radius, intensity, tm=None):
    """Get an image containing a cylinder."""
    cylinder = pet.EllipticCylinder()
    cylinder.set_length(length)
    cylinder.set_origin([0, 0, 0])
    cylinder.set_radii([radius, radius])
    im = im_in.clone()
    im.fill(0)
    im.add_shape(cylinder, intensity)
    if tm:
        # resample
        res = reg.NiftyResample()
        res.set_reference_image(im)
        res.set_floating_image(im)
        res.add_transformation(tm)
        res.set_interpolation_type_to_cubic_spline()
        im = res.forward(im)
    return im


@jit(nopython=True)
def loop_and_replace(arr_out, arr_to_add):
    """JIT loop."""
    arr_shape = arr_out.shape
    arr_to_add_thresh = 0.01 * arr_to_add.max()
    for ix, iy, iz in np.ndindex(arr_shape):
        arr_out[ix, iy, iz] = max(arr_out[ix, iy, iz], arr_to_add[ix, iy, iz])
        if abs(arr_to_add[ix, iy, iz]) > arr_to_add_thresh:
            arr_out[ix, iy, iz] = arr_to_add[ix, iy, iz]


def replace_if_greater(out, to_add):
    """To add."""
    out_arr = out.as_array()
    to_add_arr = to_add.as_array()
    out_arr = np.maximum(out_arr, to_add_arr)
    # loop_and_replace(out_arr, to_add_arr)
    out.fill(out_arr)


def main():
    """Do main function."""

    # Get brainweb image
    FDG_arr = get_brainweb_image()

    # Crop to (127,127,127) and flip component for correct orientation
    FDG_arr = crop_and_orient(FDG_arr)

    # Convert numpy array to STIR image
    FDG = get_as_pet_im(FDG_arr)

    # Save unmodified image
    save_nii(FDG, out_prefix + "_original")

    # Parameters
    side = ('right', 'left')
    distance_from_centre = (cL, cR)
    outer_cylinder_radius = (oRL, oRR)
    inner_cylinder_radius = (iRL, iRR)
    outer_cylinder_intensity = (oIL, oIR)
    inner_cylinder_intensity = (iIL, iIR)

    out = FDG.clone()

    for i in range(2):

        print("Creating " + side[i] + " temporal artery..." +
              "\n\tInner radius: " + str(inner_cylinder_radius[i]) +
              "\n\tOuter radius: " + str(outer_cylinder_radius[i]) +
              "\n\tInner intensity: " + str(inner_cylinder_intensity[i]) +
              "\n\tOuter intensity: " + str(outer_cylinder_intensity[i]) +
              "\n\tDistance from centre: " + str(distance_from_centre[i]))

        np_tm = np.array([[1, 0, 0, distance_from_centre[i]],
                         [0, 0, 1, 150],
                         [0, 1, 0, 0],
                         [0, 0, 0, 1]])
        tm = reg.AffineTransformation(np_tm)

        outer_cylinder = get_cylinder_in_im(
            FDG, length=40, tm=tm,
            radius=outer_cylinder_radius[i],
            intensity=outer_cylinder_intensity[i]
            )
        inner_cylinder = get_cylinder_in_im(
            FDG, length=40, tm=tm,
            radius=inner_cylinder_radius[i],
            intensity=inner_cylinder_intensity[i] - outer_cylinder_intensity[i]
            )

        # For outer cylinder, need it to replace the skin
        # So we can't do a simple add, we need to replace
        replace_if_greater(out, outer_cylinder)
        out += inner_cylinder
    save_nii(out, out_prefix)


if __name__ == "__main__":
    main()
