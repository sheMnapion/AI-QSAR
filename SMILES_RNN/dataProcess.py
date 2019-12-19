import numpy as np
import pandas as pd

def loadEsolSmilesData():
    """load data from ESOL dataset where only smiles format and the aim is extracted"""
    esol=pd.read_csv('../datasets/delaney-processed.csv')
    smiles=np.array(esol['smiles'])
    properties=np.array(esol['measured log solubility in mols per litre'])
    return smiles, properties

if __name__=='__main__':
    loadEsolSmilesData()