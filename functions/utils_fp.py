from rdkit.Chem import MACCSkeys
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, Descriptors
from json import JSONEncoder
from rdkit.Chem import PandasTools
import json
import numpy as np
import pandas as pd


def check_extention(file, encode = 'latin-1'):
    """ check the file extention and convert to ROMol if necessary """

    if file[-3:] == "sdf":       
        imported_file = PandasTools.LoadSDF(file, smilesName='SMILES', includeFingerprints=False)
        return imported_file
    
    elif file[-4:] == "xlsx":
        imported_file = pd.read_excel(file)
        return imported_file
        
    elif file[-3:] == "csv":
        imported_file = pd.read_csv(file, encoding=encode)
        return imported_file
    
    else:
        return ("file extension not supported, supported extentions are: csv, xlsx and sdf")

def rdkit_numpy_convert(fp):
    """Convert rdkit mol to numpy array"""
    output = []
    for f in fp:
        arr = np.zeros((1,))
        DataStructs.ConvertToNumpyArray(f, arr)
        output.append(arr)
    return np.asarray(output)


class NumpyArrayEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return JSONEncoder.default(self, obj)


def fp_generation(mols, radii, bits, type, maccs = True):

    """ Encodes x variable """

    mols = [Chem.MolFromSmiles(smile, sanitize=True) for smile in mols]

    if maccs == True:
        #generates maccs fp
        maccs_fp = [MACCSkeys.GenMACCSKeys(x) for x in mols]
        maccs_fp = rdkit_numpy_convert(maccs_fp)

        with open(f'./data/fp/{type}/maccs_fp.json', 'w', encoding='utf-8') as f:
            json.dump(maccs_fp, f, cls=NumpyArrayEncoder)
    else:
        pass

    #fcfp & ecfp fp generation
    for radius in radii:
        for bit in bits:
            ecfp_fp = [AllChem.GetMorganFingerprintAsBitVect(m, radius, bit, useFeatures=False) for m in mols]
            fcfp_fp = [AllChem.GetMorganFingerprintAsBitVect(m, radius, bit, useFeatures=True) for m in mols]

            ecfp_fp = rdkit_numpy_convert(ecfp_fp)
            fcfp_fp = rdkit_numpy_convert(fcfp_fp)

            with open('./data/fp/{}/ecfp_fp_{}_{}.json'.format(type, radius, bit), 'w', encoding='utf-8') as f:
                json.dump(ecfp_fp, f, cls=NumpyArrayEncoder)
            
            with open('./data/fp/{}/fcfp_fp_{}_{}.json'.format(type, radius, bit), 'w', encoding='utf-8') as f:
                json.dump(fcfp_fp, f, cls=NumpyArrayEncoder)

def y_generation(data, type):
    """ Encodes y variable """
    y = [y for y in data]
    with open(f'./data/curated_data/y/{type}/y_{type}.json', 'w', encoding='utf-8') as f:
        json.dump(y, f, cls=NumpyArrayEncoder)