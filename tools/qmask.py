from os.path import join, split
from tools.debug import Debug
import nibabel as nib
import numpy as np 

debug = Debug()



class Qmask:
    def __init__(self):
        pass


    def get_low_cov_nodes(self,qmask_path,parcellation_data_np,mrsi_cov=0.65,parcel_labels=None):
        if parcel_labels is None:
            parcel_labels = np.unique(parcellation_data_np).astype(int)
            parcel_labels = parcel_labels[parcel_labels!=0]
        qmask_pop      = nib.load(qmask_path)
        n_voxel_counts_dict = self.count_voxels_inside_parcel(qmask_pop.get_fdata(), 
                                                            parcellation_data_np, 
                                                            parcel_labels)
        ignore_parcel_idx = [index for index in n_voxel_counts_dict if n_voxel_counts_dict[index] < mrsi_cov]
        low_cov_nodes_ids = [np.where(parcel_labels == parcel_idx)[0][0] for parcel_idx in ignore_parcel_idx if len(np.where(parcel_labels == parcel_idx)[0]) != 0]
        low_cov_nodes_ids = np.sort(np.array(low_cov_nodes_ids)) 
        return parcel_labels[low_cov_nodes_ids], low_cov_nodes_ids
    
    def count_voxels_inside_parcel(self,image3D, parcel_image3D, parcel_ids_list,norm=True):
        from connectomics.parcellate import Parcellate
        return Parcellate().count_voxels_inside_parcel(image3D, parcel_image3D, parcel_ids_list,norm)