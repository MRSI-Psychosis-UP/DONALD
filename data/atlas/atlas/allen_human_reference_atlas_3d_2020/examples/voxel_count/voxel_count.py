import json
import pandas as pd
import urllib3
import SimpleITK as sitk
import numpy as np

#
# Example python code to download structure information and ontology from the Allen Brain API
# and count the number voxels annotated for each structure.
#
# Outputs file "voxel_count.csv"
#

output_columns = [
'id', # structure database id
'graph_order', # order of structure in structure_graph
'acronym', # structure name acronym
'name', # structure name
'color_hex_triplet', # structure color
'parent_structure_id', # database id of the parent structure
'structure_id_path', # path to root of structure_graph as slashed separted structure id list
'annotated', # boolean to indicate if this structure id is in the annotation volume
'voxel_count', # number of voxel in annotation volume with this structure id
'subgraph_annotated', # boolean to indicate if structure id of any child (subgraph) structure in the annotation volume
'subgraph_voxel_count', # number of voxel in annotation volume for this structure including subgraph
'volume_mm3', # structure volume in mm^3 based on subgraph_voxel_count
'volume_cm3' # structure volume in cm^3 based on subgraph_voxel_count
]


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

# Open the annotation volume and count number of annotated voxels per structure value
volume = sitk.ReadImage( '../../annotation.nii.gz' )
arr = sitk.GetArrayFromImage( volume )

# Compute conversion from number of voxel to mm^3
conversion_mm3 = np.prod(np.array(volume.GetSpacing()))
conversion_cm3 = conversion_mm3 / 1000.0

# Find unique annotation values and count and compute subgraph aggregations
unique_values, unique_counts  = np.unique( arr, return_counts=True )
voxel_counts = dict(zip(unique_values,unique_counts))

structures['annotated'] = False
structures['voxel_count'] = 0
structures['subgraph_annotated'] = False
structures['subgraph_voxel_count'] = 0

for a in voxel_counts:
    if a == 0 :
        continue
    print( 'processing id = %d' % a )
    structures.at[a,'annotated'] = True
    structures.at[a,'voxel_count'] = voxel_counts[a]
    path = structures.loc[a,'structure_id_path']
    for p in path.split('/') :
        if p.isnumeric() :
            pid = int(p)
            structures.at[pid,'subgraph_annotated'] = True
            current_total = structures.loc[pid,'subgraph_voxel_count']
            structures.at[pid,'subgraph_voxel_count'] = current_total + voxel_counts[a]

# Apply conversion to get volume
structures['volume_mm3'] = structures['subgraph_voxel_count'] * conversion_mm3           
structures['volume_cm3'] = structures['subgraph_voxel_count'] * conversion_cm3           
            
# set unannotated counts to null
structures.at[structures['annotated']==False,'voxel_count'] = ""
structures.at[structures['subgraph_annotated']==False,'subgraph_voxel_count'] = ""
structures.at[structures['subgraph_annotated']==False,'volume_mm3'] = ""
structures.at[structures['subgraph_annotated']==False,'volume_cm3'] = ""

# write out dataframe to csv file
structures.reset_index(inplace=True)
filtered = structures[output_columns]
filtered.set_index('id', inplace=True)
filtered.to_csv('voxel_count.csv')









