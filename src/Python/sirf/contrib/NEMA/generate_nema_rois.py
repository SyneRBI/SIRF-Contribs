'''Generate NEMA ROIs.

Usage:
  generate_nema_rois [--help | options]

Options:
  -s <file>, --sino=<file>     raw data file [default: no default you need an input of a NEMA sinogram]
  -o <out_path>, --outpath=<out_path>     path to data files, defaults to current directory
                               subfolder of SIRF root folder
  -xy <xy_size>, --xysize=<xy_size> optional size of image in x and y, default is 150
'''

## SyneRBI Synergistic Image Reconstruction Framework (SIRF)
## Copyright 2024 National Physical Laboratory
##
## This is software developed for the Collaborative Computational
## Project in Synergistic Reconstruction for Biomedical Imaging (formerly CCP PETMR)
## (http://www.ccpsynerbi.ac.uk/).
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

__version__ = '0.1.0'
from docopt import docopt
args = docopt(__doc__, version=__version__)
#print(args)

import sirf
import nibabel as nii
import matplotlib.pyplot as plt
import numpy as np
import sys
import os
import pathlib
from ast import literal_eval

from sirf.Utilities import show_2D_array

import sirf.Reg as Reg
import math as m

from sirf.Utilities import show_2D_array, examples_data_path
import sirf.STIR  as pet

# process command-line option

data_output_path = args['--outpath']
if data_output_path is None:
        data_output_path =  './'
prefix = data_output_path + '/'

sino_file = args['--sino']
if sino_file is None:
     sys.exit('Missing the input sinogram as interfile')

xy_size = args['--xysize']
if xy_size is None:
     print('Warning: I am setting x and y size to 150 make sure your reconstructed image has the same dimension')
     xy_size=150



def extract_nema_ROIs(image_size_xy=xy_size):
     
    acq_data = pet.AcquisitionData(sino_file)
    initial_image =  acq_data.create_uniform_image(1.0, xy=image_size_xy)
    image_size = initial_image.dimensions()
    # need to run a simple recon to register ROI with PET
# using parallelproj
    acq_model = pet.AcquisitionModelUsingParallelproj()
# define objective function to be maximized as
# Poisson logarithmic likelihood (with linear model for mean)
    obj_fun = pet.make_Poisson_loglikelihood(acq_data)
    obj_fun.set_acquisition_model(acq_model)
# create the reconstruction object
    recon = pet.OSMAPOSLReconstructor()
    recon.set_objective_function(obj_fun)

# Choose a number of subsets.
    num_subsets = 21
# let's run 1 full iteration which is ideally enough for the registration
    num_subiterations = 21
    recon.set_num_subsets(num_subsets)
    recon.set_num_subiterations(num_subiterations)
    recon_im = initial_image
    recon.set_up(recon_im)
# set the initial image estimate
    recon.set_current_estimate(recon_im)
# reconstruct
    recon.process()
    osem_image = recon.get_output()

    R=114/2
    z=140
    angle_smallest=210
# create an empty image
    empty_image = acq_data.create_uniform_image(0, xy=int(image_size[1]))
    image=empty_image
# assuming exagon shape

    shape6 = pet.Ellipsoid()
    shape5 = pet.Ellipsoid()
    shape4 = pet.Ellipsoid()
    shape3 = pet.Ellipsoid()
    shape2 = pet.Ellipsoid()
    shape1 = pet.Ellipsoid()
 
#Sphere 6 37 mm
    shape6.set_radius_x((18.5))
    shape6.set_radius_y((18.5))
    shape6.set_radius_z((18.5))
    shape6.set_origin((z, -R*m.sin(m.radians(angle_smallest+300)), R*m.cos(m.radians(angle_smallest+300))))
# add the shape to the image
    image.add_shape(shape6, scale = 1)

#Sphere 5 28 mm
    shape5.set_radius_x((14))
    shape5.set_radius_y((14))
    shape5.set_radius_z((14))
    shape5.set_origin((z, -R*m.sin(m.radians(angle_smallest+240)), R*m.cos(m.radians(angle_smallest+240))))
# add the shape to the image
    image.add_shape(shape5, scale = 1)

#Sphere 4 22 mm 
    shape4.set_radius_x((11))
    shape4.set_radius_y((11))
    shape4.set_radius_z((11))
    shape4.set_origin((z,-R*m.sin(m.radians(angle_smallest+180)), R*m.cos(m.radians(angle_smallest+180))))
# add the shape to the image
    image.add_shape(shape4, scale = 1)

#Sphere 3 17 mm
    shape3.set_radius_x((8.5))
    shape3.set_radius_y((8.5))
    shape3.set_radius_z((8.5))
    shape3.set_origin((z,-R*m.sin(m.radians(angle_smallest+120)), R*m.cos(m.radians(angle_smallest+120))))
# add the shape to the image
    image.add_shape(shape3, scale = 1)

#Sphere 2 13 mm
    shape2.set_radius_x((6.5))
    shape2.set_radius_y((6.5))
    shape2.set_radius_z((6.5))
    shape2.set_origin(( z,-R*m.sin(m.radians(angle_smallest+60)), R*m.cos(m.radians(angle_smallest+60))))
# add the shape to the image
    image.add_shape(shape2, scale = 1)

#Sphere 1 10 mm 
    shape1.set_radius_x((5))
    shape1.set_radius_y((5))
    shape1.set_radius_z((5))
    shape1.set_origin((z,-R*m.sin(m.radians(angle_smallest)), R*m.cos(m.radians(angle_smallest))))
# add the shape to the image
    image.add_shape(shape1, scale = 1)

    image6 = acq_data.create_uniform_image(0, xy=int(image_size[1]))
    image5 = acq_data.create_uniform_image(0, xy=int(image_size[1]))
    image4 = acq_data.create_uniform_image(0, xy=int(image_size[1]))
    image3 = acq_data.create_uniform_image(0, xy=int(image_size[1]))
    image2 = acq_data.create_uniform_image(0, xy=int(image_size[1]))
    image1 = acq_data.create_uniform_image(0, xy=int(image_size[1]))


    image6.add_shape(shape6, scale = 1)
    image5.add_shape(shape5, scale = 1)
    image4.add_shape(shape4, scale = 1)
    image3.add_shape(shape3, scale = 1)
    image2.add_shape(shape2, scale = 1)
    image1.add_shape(shape1, scale = 1)

    parfile=pet.get_STIR_examples_dir()+'/samples/stir_math_ITK_output_file_format.par'

    osem_image.write_par(data_output_path+'osem.nii',parfile)
    osem_nii=Reg.NiftiImageData3D(data_output_path+'osem.nii')
    image.write_par(data_output_path+'sphere.nii',parfile)
    spheres_nii=Reg.NiftiImageData3D(data_output_path+'sphere.nii')

#spheres to nifty
    image6.write_par(data_output_path+'sphere6.nii',parfile)
    image5.write_par(data_output_path+'sphere5.nii',parfile)
    image4.write_par(data_output_path+'sphere4.nii',parfile)
    image3.write_par(data_output_path+'sphere3.nii',parfile)
    image2.write_par(data_output_path+'sphere2.nii',parfile)
    image1.write_par(data_output_path+'sphere1.nii',parfile)
    sphere_nii= []
    for i in range(1,6+1):

        sphere_nii.append(Reg.NiftiImageData3D(data_output_path+'sphere'+str(i)+'.nii'))

#now let's register
# The following has a bug so we have used image1.write_par() to create nii images
# osem_nii = Reg.NiftiImageData(osem_image)
# spheres_nii = Reg.NiftiImageData(image)

# Set to NiftyF3dSym for non-rigid
    algo = Reg.NiftyAladinSym()

# Set images
    algo.set_reference_image(osem_nii)
    algo.set_floating_image(spheres_nii)
#set parameters
    algo.set_parameter('SetPerformRigid','1')
    algo.set_parameter('SetPerformAffine','0')
    algo.process()
    reg_image = algo.get_output()

# reg_image_sirf = Reg.ImageData(reg_image_nii)

    np.set_printoptions(precision=3,suppress=True)
    TM = algo.get_transformation_matrix_forward()
    print(TM.as_array())

    #once we have the registration matrix we can the apply it to the sphere generation?
    resampler = Reg.NiftyResample()

# Make sure we know what the resampled image domain looks like (this can be the same as the image to resample)
    resampler.set_reference_image(reg_image)
# Add the desired transformation to apply
    resampler.add_transformation(TM)
    resampler.set_padding_value(0)
# Use linear interpolation
    resampler.set_interpolation_type_to_linear()

    for j in range(6):
# Set image to resample
        resampler.set_floating_image(sphere_nii[j])
# Go!
        resampler.process()
        Roi = resampler.get_output()
        Roi.write(data_output_path+'S'+str(j+1)+'.nii')#TODO getting sirf imagedata to nifty to work without messing the orientation
        ROIsirf = pet.ImageData(data_output_path+'S'+str(j+1)+'.nii')
        ROIsirf.write('S'+str(j+1))

def main():
    extract_nema_ROIs(xy_size)
    
try:
    main()
    print('\n=== done with %s' % __file__)

except error as err:
    print('%s' % err.value)

