'''Generate NEMA ROIs. The script generates the 6 NEMA spheres as ROI images and a 7th ROI image with a 
sphere at the centre of the phamptom. The 7th ROI is meant to be used as background to calculate contrast,
SD etc. The number of the ROI goes from the smallest to the biggest.
unregistered_sphere<i>.nii is the nift image of the ith unregistered sphere;
unregistered_spheres.nii is the nifti image with all the unregistered spheres;
S<i>.nii/.hv is the nifti/interfile image of the ith sphere. These are the ones you need to use for analysis.

Usage:
  generate_nema_rois [--help | options]

Options:
  -s <file>, --sino=<file>     raw data file [default: no default you need an input of a NEMA sinogram]
  -i <file>, --image=<file>              reconstructed image file if None the script runs a reconstruction
  -o <out_path>, --outpath=<out_path>     path to data files, defaults to current directory
  --xysize=<xy_size> optional size of image in x and y
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


import numpy as np
import sys

import sirf.Reg as Reg
import math as m

import sirf.STIR  as pet

def recon_from_sino(acq_data,image_size):
    # Run a simple recon to use as reference image to register ROI with PET

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
    return recon.get_output() 

def construct_NEMA_spheres_and_save(image):
    # Generates the spheres given the geometry of the input image. An image for each sphere and one with all the spheres
    # are genrated and saved to nii format
    
    R=114/2
    z=140
    angle_smallest=210
    # create an empty image
    empty_image = image.get_uniform_copy(0)
    image=empty_image
    # assuming exagon shape


    shape7 = pet.Ellipsoid()
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

    #Sphere 7 50 mm
    shape7.set_radius_x((25))
    shape7.set_radius_y((25))
    shape7.set_radius_z((25))
    shape7.set_origin((z, 0, 0))

    image7 = image.get_uniform_copy(0)
    image6 = image.get_uniform_copy(0)
    image5 = image.get_uniform_copy(0)
    image4 = image.get_uniform_copy(0)
    image3 = image.get_uniform_copy(0)
    image2 = image.get_uniform_copy(0)
    image1 = image.get_uniform_copy(0)


    image7.add_shape(shape7, scale = 1)
    image6.add_shape(shape6, scale = 1)
    image5.add_shape(shape5, scale = 1)
    image4.add_shape(shape4, scale = 1)
    image3.add_shape(shape3, scale = 1)
    image2.add_shape(shape2, scale = 1)
    image1.add_shape(shape1, scale = 1)

    parfile=pet.get_STIR_examples_dir()+'/samples/stir_math_ITK_output_file_format.par'

    image.write_par(data_output_path+'unregistered_spheres.nii',parfile)

    #unregistered_spheres to nifty
    image7.write_par(data_output_path+'unregistered_sphere7.nii',parfile)
    image6.write_par(data_output_path+'unregistered_sphere6.nii',parfile)
    image5.write_par(data_output_path+'unregistered_sphere5.nii',parfile)
    image4.write_par(data_output_path+'unregistered_sphere4.nii',parfile)
    image3.write_par(data_output_path+'unregistered_sphere3.nii',parfile)
    image2.write_par(data_output_path+'unregistered_sphere2.nii',parfile)
    image1.write_par(data_output_path+'unregistered_sphere1.nii',parfile)

def do_registration(recon_image):
    # run the registration between the reconstructed image in input and  the image containing all the 6 NEMA spheres
    # return the transformation matrix, nifti registered image, and the unregistered image
     
    parfile=pet.get_STIR_examples_dir()+'/samples/stir_math_ITK_output_file_format.par'
    recon_image.write_par(data_output_path+'recon.nii',parfile)
    recon_nii=Reg.NiftiImageData3D(data_output_path+'recon.nii')
    unregistered_spheres_nii=Reg.NiftiImageData3D(data_output_path+'unregistered_spheres.nii')
    unregistered_sphere_nii= []

    for i in range(1,7+1):
        unregistered_sphere_nii.append(Reg.NiftiImageData3D(data_output_path+'unregistered_sphere'+str(i)+'.nii'))

    #now let's register
    # The following has a bug so we have used image1.write_par() to create nii images
    # osem_nii = Reg.NiftiImageData(osem_image)
    # unregistered_spheres_nii = Reg.NiftiImageData(image)

    # Set to NiftyF3dSym for non-rigid
    algo = Reg.NiftyAladinSym()

    # Set images
    algo.set_reference_image(recon_nii)
    algo.set_floating_image(unregistered_spheres_nii)
    #set parameters
    algo.set_parameter('SetPerformRigid','1')
    algo.set_parameter('SetPerformAffine','0')
    algo.process()
    reg_image = algo.get_output()

    # reg_image_sirf = Reg.ImageData(reg_image_nii)

    np.set_printoptions(precision=3,suppress=True)
    TM = algo.get_transformation_matrix_forward()
    print(TM.as_array())
    return TM, reg_image, unregistered_sphere_nii

def generate_nema_rois(recon_image):
    #This actually  calls the different functions and return/save the registered ROI
    
    construct_NEMA_spheres_and_save(recon_image)
    TM, reg_image, unregistered_sphere_nii = do_registration(recon_image)    

    #once we have the registration matrix we can then apply it to the sphere generation
    resampler = Reg.NiftyResample()

    # Make sure we know what the resampled image domain looks like (this can be the same as the image to resample)
    resampler.set_reference_image(reg_image)
    # Add the desired transformation to apply
    resampler.add_transformation(TM)
    resampler.set_padding_value(0)
    # Use nearest neighbour interpolation
    resampler.set_interpolation_type_to_nearest_neighbour()

    for j in range(7):
    # Set image to resample
        resampler.set_floating_image(unregistered_sphere_nii[j])
    # Go!
        resampler.process()
        Roi = resampler.get_output()
        Roi.write(data_output_path+'S'+str(j+1)+'.nii')#TODO getting sirf imagedata to nifty to work without messing the orientation
        ROIsirf = pet.ImageData(data_output_path+'S'+str(j+1)+'.nii')
        ROIsirf.write('S'+str(j+1))

    return ROIsirf

if __name__ == '__main__':
    from docopt import docopt
    args = docopt(__doc__, version=__version__)
    #print(args)
    data_output_path = args['--outpath']
    if data_output_path is None:
            data_output_path =  './'
    prefix = data_output_path + '/'

    
    xy_size = args['--xysize']
    if xy_size is None:
        print('Warning: Setting image xy size to 150. Make sure your reconstructed image has the same dimension as the xy_size')
        xy_size = 150
        
    recon_image_file = args['--image']
    if recon_image_file is None:

        sino_file = args['--sino']
        if sino_file is None:
            sys.exit('Missing the input sinogram as interfile')
        else:
            acq_data = pet.AcquisitionData(sino_file)
            initial_image =  acq_data.create_uniform_image(1.0, xy=xy_size)
            recon_image = recon_from_sino(acq_data, initial_image)
    else:
        recon_image=pet.ImageData(recon_image_file)
        
    generate_nema_rois(recon_image)

