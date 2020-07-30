"""Generate brainweb data with two simulated temporal arteries

Usage:
  generate_brainweb_vasculitis [--help | options]

Options:
  -i <path>, --out_im=<path>  output image filename prefix [default: im]
  --save-labels               save label images for all non-zero structures and a total background
  --voxel-size=<val>          string specifying the output voxel size (mMR | MR | brainweb) [default: mMR]
  --iIL=<val>                 inner intensity (left) [default: 1]
  --iIR=<val>                 inner intensity (right) [default: 2]
  --oIL=<val>                 outer intensity (left) [default: 5]
  --oIR=<val>                 outer intensity (right) [default: 8]
  --iRL=<val>                 inner radius (left) [default: 3]
  --iRR=<val>                 inner radius (right) [default: 3]
  --oRL=<val>                 outer radius (left) [default: 5]
  --oRR=<val>                 outer radius (right) [default: 5]
  --cL=<val>                  centre (left) [default: -80]
  --cR=<val>                  centre (left) [default: 80]
"""

# CCP SyneRBI Synergistic Image Reconstruction Framework (SIRF)
# Copyright 2020 University College London.
#
# author Richard Brown

# This is software developed for the Collaborative Computational
# Project in Synergistic Image Reconstruction for Biomedical Imaging
# (http://www.ccpsynerbi.ac.uk/).
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
import nibabel

__version__ = '0.1.0'
args = docopt(__doc__, version=__version__)

# Parse input arguments
out_prefix = args['--out_im']
save_labels = args['--save-labels']
outres=args['--voxel-size']
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


def get_brainweb_image(outres=outres, PetClass=brainweb.FDG, save_labels=False):
    """Get brainweb image."""
    fname, url = sorted(brainweb.utils.LINKS.items())[0]
    files = brainweb.get_file(fname, url, ".")
    data = brainweb.load_file(fname)

    brainweb.seed(1337)

    #for f in tqdm([fname], desc="ground truths", unit="subject"):
    vol = brainweb.get_mmr_fromfile(
            fname, petNoise=0, petSigma=0, outres=outres, PetClass=PetClass)
    if save_labels:
        labels = brainweb.get_label_probabilities(fname, outres=outres)
        non_zero_labels = PetClass.attrs
        all_labels = PetClass.all_labels
        non_zero_indices = list(all_labels.index(l) for l in non_zero_labels)
        # keep only non-zero ones
        labels = labels[non_zero_indices, :, :, :]
        return (vol['PET'], vol['res'], labels, non_zero_labels)
    else:
        return (vol['PET'], vol['res'])

def crop_and_orient(im, res):
    """Crop and orient image."""
    # original code for the mMR voxel sizes 
    # im = im[:, 105:105+127, 108:108+127]
    mMR_res = np.array((2.0312, 2.0863, 2.0863))
    org_min=np.array((0, 105, 108))
    org_max=org_min+127
    new_min = np.int32(np.round(org_min*mMR_res/res))
    new_max = np.int32(np.round(org_max*mMR_res/res))
    im = im[new_min[0]:new_max[0], new_min[1]:new_max[1], new_min[2]:new_max[2]]
    im = np.flip(im, 0)
    return im


def get_as_pet_im(arr, res):
    """Get as PET image."""
    # We'll need a template sinogram
    #mMR_template_sino = \
    #    examples_data_path('PET') + "/mMR/mMR_template_span11.hs"
    #templ_sino = pet.AcquisitionData(mMR_template_sino)
    #im = pet.ImageData(templ_sino)
    im = pet.ImageData()
    im.initialise(arr.shape, tuple(res))
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

def weighted_add(out, values, weights):
    """set out to out*(1-sum(weights)) + sum(weights*values).
    
    Currently only works for SIRF/CIL objects
    """
    zero=out.allocate()
    out *= (out.allocate(1)-sum(weights, zero))
    out += sum([weights[i]*values[i] for i in range(len(values))], zero)

def make_4d_nifti(out_filename, all_filenames):
    # first read one to get geometry ok
    template = nibabel.load(all_filenames[0])
    all_data = ( nibabel.load(f).get_fdata() for f in all_filenames)
    nii = nibabel.Nifti1Image(np.array(list(all_data)), template.affine)
    nibabel.save(nii, out_filename)

def main():
    """Do main function."""

    # Get brainweb image
    images = get_brainweb_image(outres=outres, save_labels=save_labels)
    FDG_arr = images[0]
    res = images[1]
    if save_labels:
        labels = images[2]
        label_names = images[3]
    

    # Crop and flip component for correct orientation
    FDG_arr = crop_and_orient(FDG_arr, res)

    # Convert numpy array to STIR image
    FDG = get_as_pet_im(FDG_arr, res)

    # rescale to SUV
    FDG *= 5 / np.max(FDG.as_array())

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
    label_all_vessels = FDG.allocate(0)
    all_vessels = []
    all_vessel_filenames = []
    for i in range(2):

        print("Creating " + side[i] + " temporal artery..." +
              "\n\tInner radius: " + str(inner_cylinder_radius[i]) +
              "\n\tOuter radius: " + str(outer_cylinder_radius[i]) +
              "\n\tInner intensity: " + str(inner_cylinder_intensity[i]) +
              "\n\tOuter intensity: " + str(outer_cylinder_intensity[i]) +
              "\n\tDistance from centre: " + str(distance_from_centre[i]))

        # TODO remove hard-coded 150
        np_tm = np.array([[1, 0, 0, distance_from_centre[i]],
                         [0, 0, 1, 150],
                         [0, 1, 0, 0],
                         [0, 0, 0, 1]])
        tm = reg.AffineTransformation(np_tm)

        outer_cylinder = get_cylinder_in_im(
            FDG, length=40, tm=tm,
            radius=outer_cylinder_radius[i],
            intensity=1
            )
        inner_cylinder = get_cylinder_in_im(
            FDG, length=40, tm=tm,
            radius=inner_cylinder_radius[i],
            intensity=1
            )
        label_all_vessels += outer_cylinder
        outer_cylinder -= inner_cylinder
        if save_labels:
            all_vessels.append(outer_cylinder)
            all_vessels.append(inner_cylinder)
            filename = out_prefix + "_label_outer_cylinder" + str(i)
            save_nii(outer_cylinder, filename)
            all_vessel_filenames.append(filename + ".nii")
            filename = out_prefix + "_label_inner_cylinder" + str(i)
            save_nii(inner_cylinder, filename)
            all_vessel_filenames.append(filename + ".nii")

        weighted_add(out, [outer_cylinder_intensity[i], inner_cylinder_intensity[i]], [outer_cylinder, inner_cylinder])

    save_nii(out, out_prefix)

    if save_labels:
        label_all_vessels_array = label_all_vessels.as_array()
        num_labels = labels.shape[0] + 1 + 2*2 # background + 4 vessel regions
        #all_labels = np.zeros(label_all_vessels_array.shape + (num_labels,), dtype='float32')
        all_label_filenames = []
        # initialise background as everything except the vessels. we'll then subtract the rest as we go along
        total_background = label_all_vessels.allocate(1) - label_all_vessels

        for i in range(labels.shape[0]):
            array = crop_and_orient(labels[i,:,:,:], res)
            # exclude temporal vessels
            array *= (1-label_all_vessels_array)
            this_label = get_as_pet_im(array, res)
            this_filename = out_prefix + "_label" +str(i) + "_" + label_names[i]
            save_nii(this_label, this_filename)
            all_label_filenames.append(this_filename + ".nii")
            #all_labels[i, :, :, :] = this_label.as_array()
            total_background -= this_label

        # store vessels and background in the last 5
        #all_labels[-5:-1, :, :, :] = np.array(list(v.as_array() for v in all_vessels))
        #all_labels[-1, :, :, :] = total_background.as_array()
        all_label_filenames += all_vessel_filenames

        this_filename = out_prefix + "_label_everything_else"
        save_nii(total_background, this_filename)
        all_label_filenames.append(this_filename + ".nii")

        # now make one 4D nifti
        make_4d_nifti(out_prefix + "_alllabels.nii", all_label_filenames)

if __name__ == "__main__":
    main()
