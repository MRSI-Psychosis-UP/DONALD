import json
import pandas as pd
import urllib3
import SimpleITK as sitk
import numpy as np
import csv
import os

#
# Example python code to download structure information and ontology from the Allen Brain API
# and create ITK-SNAP compatible files for visualization of the parcellation for various 
# subset of structures (views).
#
# See ITK-SNAP website for download and documentation: http://www.itksnap.org/
#
# Outputs a ITK-SnAP segmentation file "itksnap_segmentation.nii.gz"
# Outputs a ITK-SnAP label description file for each view
#

views = [
'Br', # brain
'FSS_MSS_HSS', # surface structures
'FWM_MWM_HWM', # white matter
'FGM_MGM_HGM', # gray matter
'FV_MV_HV', # ventricles
'BN_SN_STH' # basel ganglia
] 


output_columns = {
'IDX': 'uint16', # Zero-based index 
'-R-': 'uint8', # Red color component (0..255)
'-G-': 'uint8', # Green color component (0..255)
'-B-': 'uint8', # Blue color component (0..255)
'-A-': 'float', # Label transparency (0.00 .. 1.00)
'VIS': 'uint8', # Label visibility (0 or 1)
'MSH': 'uint8', # Label mesh visibility (0 or 1)
'LABEL': 'str' #Label description
}


# Specify the structure graph
graph_id = 16

# RMA query to fetch structures for the structure graph
query_url = "http://api.brain-map.org/api/v2/data/query.json?criteria=model::Structure"
query_url += ",rma::criteria,[graph_id$eq%d]" % graph_id
query_url += ",rma::options[order$eq'structures.graph_order'][num_rows$eqall]"

# Make http request and create a pandas dataframe
http = urllib3.PoolManager()
r = http.request('GET', query_url)
data = json.loads(r.data.decode('utf-8'))['msg']
structures = pd.read_json( json.dumps(data) )
structures.set_index( 'id', inplace=True )

# Open the annotation volume and count number of annotated voxels per structure
input = sitk.ReadImage( '../../annotation.nii.gz' )
arr = sitk.GetArrayFromImage( input )

# Find unique annotation values
unique_values, unique_counts  = np.unique( arr, return_counts=True )
voxel_counts = dict(zip(unique_values, unique_counts))

if 0 in voxel_counts :
    del voxel_counts[0]

#    
# Assign each unique annotation value a ITK-SNAP segmentation index 
# Create an ITK-SNAP segmentation file using those indices
# Convert reference atlas color from hex to RGB format
#

# initializations
for c in output_columns :
    structures[c] = 0
structures['LABEL'] = ""

itksnap_index = 1

for a in voxel_counts:
        
    structures.at[a,'IDX'] = itksnap_index
    structures.at[a,'LABEL'] = structures.loc[a,'acronym'] + ' - ' + str(a)
    structures.at[a,'-A-'] = 1
    
    h = structures.loc[a,'color_hex_triplet']
    rgb = tuple(int(h[i:i+2], 16) for i in (0, 2, 4))
    
    structures.at[a,'-R-'] = rgb[0]
    structures.at[a,'-G-'] = rgb[1]    
    structures.at[a,'-B-'] = rgb[2]
    
    structures.at[a,'VIS'] = 0
    structures.at[a,'MSH'] = 0
    
    itksnap_index += 1
    

# create annotation volume that matches the ITK-SnAP indices
lut = np.zeros( int(max(structures.index.values)) + 1, np.uint32 )
lut[structures.index] = structures.IDX

arr = np.take( lut, arr )
output = sitk.GetImageFromArray( arr )
output.CopyInformation( input )
sitk.WriteImage( output, 'itksnap_segmentation.nii.gz', True )

# volumes to convert
volumes = ['../flip_annotation/annotation_flipped.nii.gz',
    '../flip_annotation/annotation_full.nii.gz']
for vpath in volumes :

    if os.path.exists( vpath ) :
        inputImage = sitk.ReadImage( vpath )
        arr = sitk.GetArrayFromImage( inputImage )
        arr = np.take( lut, arr )
        outputImage = sitk.GetImageFromArray( arr )
        outputImage.CopyInformation( inputImage )
        output_path = 'itksnap_' + os.path.basename( vpath )
        sitk.WriteImage( outputImage, output_path , True )




#
# Create an ITK-SNAP label description file for each view
#
structures.reset_index( inplace=True )
structures.set_index('acronym', inplace=True )
structures.at['VIS'] = 0  # make all labels invisible
structures.at['MSH'] = 0  # make all meshes invisible

for v in views :

    # make of copy of dataframe with only annotated structures
    filtered  = structures[structures['IDX'] > 0].copy()     
    
    # split each view into its component structures
    slist = v.split('_')
    print( 'view = %s, %s' % (v,slist))
    
    # for each structure, make any structure/mesh within its subgraph visible
    for s in slist :
    
        pattern = '/%d/' % structures.loc[s,'id']
        print( 'structure = %s, pattern = %s' % (s,pattern) )
       
        row_indexer = filtered['structure_id_path'].str.contains(pattern)
        
        filtered.loc[row_indexer,'VIS'] = 1
        filtered.loc[row_indexer,'MSH'] = 1
 
    
    # write label description file
    output_file = 'itksnap_label_description_%s.txt' % v
    filtered.to_csv( output_file, 
                        sep=' ', columns=output_columns, header=False, index=False, float_format='%.0f' )

 



