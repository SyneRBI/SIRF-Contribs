"""Generate brainweb data with two simulated temporal arteries

Usage:
  generate_brainweb_vasculitis [--help | options]

Options:
  -i <path>, --out_im=<path>     output image filename prefix [default: im]
  --save-labels               save label images for all non-zero structures and a total background
  --brainweb-cache=<path>       filename prefix for saving brainweb data [default: brainweb_labels]
  --voxel-size=<val>          string specifying the output voxel size (mMR | MR | brainweb) [default: mMR]
  --iIL=<val>                 inner intensity (left) [default: 1]
  --iIR=<val>                 inner intensity (right) [default: 2]
  --oIL=<val>                 outer intensity (left) [default: 5]
  --oIR=<val>                 outer intensity (right) [default: 8]
  --iRL=<val>                 inner radius (left) [default: 3]
  --iRR=<val>                 inner radius (right) [default: 3]
  --oRL=<val>                 outer radius (left) [default: 5]
  --oRR=<val>                 outer radius (right) [default: 5]
  --lL=<val>                  vessel length (left) [default: 40]
  --lR=<val>                  vessel length (right) [default: 40]
  --cL=<val>                  centre (left) [default: -80]
  --cR=<val>                  centre (left) [default: 80]
"""

# CCP SyneRBI Synergistic Image Reconstruction Framework (SIRF)
# Copyright 2020 University College London.
#
# author Richard Brown
# author Kris Thielemans

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

#import MINC
import brainweb
import numpy as np
from tqdm.auto import tqdm
import sirf.STIR as pet
import sirf.Reg as reg
#from sirf.Utilities import examples_data_path
from docopt import docopt
import os
import nibabel

__version__ = '0.3.0'
args = docopt(__doc__, version=__version__)
print(args)
# Parse input arguments
out_prefix = args['--out_im']
save_labels = args['--save-labels']
brainweb_label_prefix = args['--brainweb-cache']
outres=args['--voxel-size']
iIL = float(args['--iIL'])
iIR = float(args['--iIR'])
oIL = float(args['--oIL'])
oIR = float(args['--oIR'])
iRL = float(args['--iRL'])
iRR = float(args['--iRR'])
oRL = float(args['--oRL'])
oRR = float(args['--oRR'])
lL = float(args['--lL'])
lR = float(args['--lR'])
cL = float(args['--cL'])
cR = float(args['--cR'])

def get_brainweb_labels():
    """Get brainweb image."""
    fname, url = sorted(brainweb.utils.LINKS.items())[0]
    brainweb.get_file(fname, url, ".")
    data = brainweb.load_file(fname)
    return data

def get_brainweb_labels_as_pet():
    data=get_brainweb_labels()
    res=getattr(brainweb.Res,'brainweb')
    new_shape=(data.shape[0],512,512)   
    padLR, padR = divmod((np.array(new_shape) - data.shape), 2)
    data = np.pad(data, [(p, p + r) for (p, r)
                         in zip(padLR.astype(int), padR.astype(int))],
                       mode="constant")
    #data = np.flip(data, 0)
    return get_as_pet_im(data,res)

def get_brainweb_image(outres=outres, PetClass=brainweb.FDG, save_labels=False):
    """Get brainweb image. (no longer used)"""
    fname, url = sorted(brainweb.utils.LINKS.items())[0]
    brainweb.get_file(fname, url, ".")
    #data = brainweb.load_file(fname)

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
    im = pet.ImageData()
    im.initialise(arr.shape, tuple(res))
    im.fill(arr)
    return im

def save_nii(im, fname):
    """Save as nii."""
    reg.ImageData(im).write(fname)


def get_cylinder_in_im(im_in, length, radius, origin, intensity, tm=None, num_samples=3):
    """Get an image containing a cylinder."""
    cylinder = pet.EllipticCylinder()
    cylinder.set_length(length)
    cylinder.set_radii([radius, radius])
    geo = im_in.get_geometrical_info()
    centre = (np.array(geo.get_offset()) +
              (np.array(geo.get_size())-1)*np.array(geo.get_spacing())/2.)
    # warning: CURRENTLY NEED TO REVERSE
    cylinder.set_origin(tuple(np.array(origin) + centre[::-1]))
    im = im_in.clone()
    im.fill(0)
    im.add_shape(cylinder, intensity, num_samples)
    if tm:
        # resample
        res = reg.NiftyResample()
        res.set_reference_image(im)
        res.set_floating_image(im)
        res.add_transformation(tm)
        res.set_interpolation_type_to_cubic_spline()
        im = res.forward(im)
    return im


def weighted_add(out, values, weights):
    """set out to out + sum(weights*values) """
    for (w,v) in zip (weights, values):
        out += w*v

def zoom_image(im, new_voxel_size):
    """
    returns an image with new voxel sizes

    It uses the 'preserve_values' option of sirf.STIR.ImageData.zoom_image (appropriate for probabilistic labels)

    This uses internal STIR knowledge such that the zoomed image still has the same STIR offset as the input.
    This is only important once using the image for forward projection etc
    """
    geo=im.get_geometrical_info()
    # warning: need to revert these at present
    voxel_size = np.array(geo.get_spacing()[::-1])
    size = np.array(geo.get_size()[::-1])
    zooms = voxel_size / new_voxel_size
    new_size = np.array(np.ceil(size * zooms), 'int')
    # make odd-sizes
    new_size += 1 - (new_size%2)
    # internal STIR calculations:
    STIR_min_ind = np.array((0,-(size[1]//2), -(size[2]//2)))
    new_STIR_min_ind = np.array((0,-(new_size[1]//2), -(new_size[2]//2)))
    STIR_middle_shift=voxel_size*(2*STIR_min_ind + size - 1)/2. - new_voxel_size*(2*new_STIR_min_ind +  new_size-1)/2.
    return im.zoom_image(zooms=tuple(zooms), offsets_in_mm=tuple(-STIR_middle_shift), size=tuple(new_size), scaling='preserve_values')

def make_4d_nifti(out_filename, all_filenames):
    # first read one to get geometry ok
    template = nibabel.load(all_filenames[0])
    all_data = ( nibabel.load(f).get_fdata() for f in all_filenames)
    nii = nibabel.Nifti1Image(np.array(list(all_data)), template.affine)
    nibabel.save(nii, out_filename)
    
def create_vessel(template,
                  inner_cylinder_radius, outer_cylinder_radius,
                  vessel_length,
                  distance_from_centre):
    """ returns a tuple (inner_cylinder, outer_cylinder) """
    #tm = reg.AffineTransformation(np_tm)

    outer_cylinder = get_cylinder_in_im(
        template, length=vessel_length, #tm=tm,
        radius=outer_cylinder_radius,
        origin=(0,0,distance_from_centre),
        intensity=1)
    inner_cylinder = get_cylinder_in_im(
        template, length=vessel_length, #tm=tm,
        radius=inner_cylinder_radius,
        origin=(0,0,distance_from_centre),
        intensity=1)
    outer_cylinder -= inner_cylinder
    return (inner_cylinder, outer_cylinder)
    
def brainweb_labels_to_4d(brainweb_labels_3d, labels = brainweb.Act.all_labels, output_prefix = ""):
    """ takes a 3D image with brainweb labels and returns them as a list of 3D masks """
    all = []
    # set empty first
    l = []
    for label in tqdm(labels):
        filename = output_prefix + label + ".nii"
        if (output_prefix and os.path.isfile(filename)):
            print("Reading " + filename)
            l = pet.ImageData(filename)
        else:
            value = getattr(brainweb.Act, label)
            if (not l):
                l = brainweb_labels_3d.allocate(0)
                brainweb_labels_array = brainweb_labels_3d.as_array()

            l.fill(brainweb_labels_array == value)
            if (output_prefix):
                save_nii(l, filename)

        all.append(l)

    return all

def get_brainweb_image_from_labels(all_label_images, act=brainweb.FDG):
    all_values = [ getattr(act, l) for l in act.attrs]
    if (len(all_label_images) != len(all_values)):
        raise Exception("get_brainweb_image_from_labels: lengths do not match")
    print("Original activity values in brainweb regions:", all_values)
    out = all_label_images[0].clone() * all_values[0]
    weighted_add(out, all_values[1:], all_label_images[1:])
    return out

def main():
    """Do main function."""

    # Parameters
    side = ('left', 'right')
    distance_from_centre = (cL, cR)
    outer_cylinder_radius = (oRL, oRR)
    inner_cylinder_radius = (iRL, iRR)
    vessel_length = (lL, lR)
    outer_cylinder_intensity = (oIL, oIR)
    inner_cylinder_intensity = (iIL, iIR)

    print("read/construct original segmentations (as 3d)")
    brainweb_labels_filename = brainweb_label_prefix + ".nii";
    if (not os.path.isfile(brainweb_labels_filename)):
        bw=get_brainweb_labels_as_pet()
        save_nii(bw, brainweb_labels_filename)
    else:
        bw=pet.ImageData(brainweb_labels_filename)

    print("convert to 4D")
    all_labels = brainweb.FDG.attrs
    all_label_images = brainweb_labels_to_4d(bw, all_labels, brainweb_label_prefix + "_")


    if (outres != "brainweb"):
        new_voxel_size = getattr(brainweb.Res, outres)
        for i in range(len(all_label_images)):
            all_label_images[i] = zoom_image(all_label_images[i], new_voxel_size)

        out = all_label_images[0].allocate()
        out.get_geometrical_info().print_info()
    else:
        out = bw # reuse the variable, dangerous, but saves a bit of memory

    print("create vessels")
    all_vessels = []
    all_vessel_values = []
    all_vessel_labels = []
    for i in range(len(inner_cylinder_intensity)):
        print("... vessel " + str(i+1))
        inner_cylinder, outer_cylinder = create_vessel(out,
                                                       inner_cylinder_radius[i],
                                                       outer_cylinder_radius[i],
                                                       vessel_length[i],
                                                       distance_from_centre[i])
        all_vessels.append(outer_cylinder)
        all_vessel_values.append(outer_cylinder_intensity[i])
        all_vessel_labels.append("outer_cylinder" + str(i))
        all_vessels.append(inner_cylinder)
        all_vessel_values.append(inner_cylinder_intensity[i])
        all_vessel_labels.append("inner_cylinder" + str(i))

    del inner_cylinder
    del outer_cylinder
        
    print("adjust brainweb labels to exclude vessels")
    # vessels will contribute fractionally to some voxels, so we need to take that fraction away
    # we therefore multiply the original maps with (1-sum(all_vessels))
    one_minus_all_vessels_summed = all_vessels[0].allocate(1)
    weighted_add(one_minus_all_vessels_summed, -np.ones(len(all_vessels)), all_vessels)
    
    for l in all_label_images:
        l *= one_minus_all_vessels_summed

    del one_minus_all_vessels_summed

    print("construct image")
    out = get_brainweb_image_from_labels(all_label_images, brainweb.FDG)
    # rescale to SUV
    out *= 5 / np.max(out.as_array())
    # add in vessels
    weighted_add(out, all_vessel_values, all_vessels)

    save_nii(out, out_prefix)

    if save_labels:
        print("saving actual labels")
        all_label_images += all_vessels
        all_labels += all_vessel_labels
        all_label_filenames = []
        # initialise background as everything. we'll then subtract the rest as we go along
        total_background = all_label_images[0].allocate(1)

        for i in range(len(all_labels)):
            this_label_image = all_label_images[i]
            this_filename = out_prefix + "_label" +str(i) + "_" + all_labels[i] + ".nii"
            save_nii(this_label_image, this_filename)
            all_label_filenames.append(this_filename)
            total_background -= this_label_image

        # store background
        this_filename = out_prefix + "_label" +str(len(all_labels)) + "_everything_else.nii"
        save_nii(total_background, this_filename)
        all_label_filenames.append(this_filename)

        # now make one 4D nifti ( this fails for me)
        #make_4d_nifti(out_prefix + "_alllabels.nii", all_label_filenames)
        # but also write all label-filenames in a text-file
        target = open(out_prefix + "_alllabels.txt", 'w')
        target.writelines((l+os.linesep  for l in all_label_filenames))
        target.close()
if __name__ == "__main__":
    main()
