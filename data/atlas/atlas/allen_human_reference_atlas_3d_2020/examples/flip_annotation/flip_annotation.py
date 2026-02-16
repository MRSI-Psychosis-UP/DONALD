import SimpleITK as sitk
import os

#
# Example python code to create a doubled sided annotation file
# Flip is implemented as scale transform with -1.0 scale
#

def resample( image, interpolatorType, transform ) :
    reference_image = image
    interpolator = interpolatorType
    default_value = 0.0
    return sitk.Resample( image, reference_image, transform, interpolator, default_value )

# Open the brain template if it exists
template_directory = '../../../../mni_icbm152_nlin_sym_09b_nifti\mni_icbm152_nlin_sym_09b'
template_path = os.path.join(template_directory,'mni_icbm152_t1_tal_nlin_sym_09b_hires.nii')
template = None

if os.path.exists( template_path ) :
    template = sitk.ReadImage( template_path )


# Open the annotation volume
annotation_directory = '../../'
annotation_path = os.path.join( annotation_directory, 'annotation.nii.gz' )
annotation = None

if os.path.exists( annotation_path ) :

    annotation = sitk.ReadImage( annotation_path )

    # The original annotation crosses the midline by a voxel
    # Zero out annotation beyond midline
    # Implement as for loop assignment
    imageSize = annotation.GetSize()
    x = 197
    for y in range(0, imageSize[1] ) :
        for z in range(0, imageSize[2] ) :
            annotation[x,y,z] = 0
    
    
# Implement the flip as a transform
dimension = 3
scale = sitk.ScaleTransform( dimension, (-1.0,1.0,1.0) )

if template: 
    resampled = resample( template, sitk.sitkLinear, scale )
    sitk.WriteImage( resampled, 'template_flipped.nii.gz', True )

if annotation:
    resampled = resample( annotation, sitk.sitkNearestNeighbor, scale )
    sitk.WriteImage( resampled, 'annotation_flipped.nii.gz', True )
    full = sitk.Maximum( annotation, resampled )
    sitk.WriteImage( full, 'annotation_full.nii.gz', True )
    
