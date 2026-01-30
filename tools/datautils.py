import numpy as np
import matplotlib as plt
import math, sys, os, re
import nibabel as nib
from tqdm import tqdm
from dotenv import load_dotenv
from os.path import join, split
from tools.debug import Debug


debug=Debug()

class DataUtils:
    def __init__(self):
        load_dotenv(override=True)

        # DEVANALYSEPATH is derived from the current script's location.
        self.DEVANALYSEPATH = os.getenv("DEVANALYSEPATH")
        self.ANARESULTSPATH = join(self.DEVANALYSEPATH, "results")
        self.DEVDATAPATH    = join(self.DEVANALYSEPATH, "data")
        self.ANALOGPATH     = join(self.DEVANALYSEPATH, "logs")
        self.BIDS_STRUCTURE_PATH = join(self.DEVANALYSEPATH, "bids", "structure.json")

        # Create necessary directories if they don't exist
        os.makedirs(self.ANALOGPATH, exist_ok=True)
        os.makedirs(self.ANARESULTSPATH, exist_ok=True)

        # Load BIDSDATAPATH from .env
        if os.getenv("BIDSDATAPATH") is None or os.getenv("BIDSDATAPATH")==".":
            debug.warning("BIDSDATAPATH env empty, set to",join(self.DEVDATAPATH,"BIDS"))
            self.BIDSDATAPATH = join(self.DEVDATAPATH,"BIDS")
        else:
            self.BIDSDATAPATH = os.getenv("BIDSDATAPATH")

        # Backwards-compat convenience path
        self.DATAPATH = self.DEVDATAPATH


if __name__=='__main__':
    u = DataUtils()





