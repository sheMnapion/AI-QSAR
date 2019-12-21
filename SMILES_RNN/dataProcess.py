import numpy as np
import pandas as pd
from rdkit import Chem

def loadEsolSmilesData():
    """load data from ESOL dataset where only smiles format and the aim is extracted"""
    esol=pd.read_csv('../datasets/delaney-processed.csv')
    smileses=np.array(esol['smiles'])
    smilesSplit=[]
    smilesDict=dict()
    for smiles in smileses:
        smilesLength=len(smiles)
        nameStr=[]
        index=0
        while index<smilesLength:
            tempAlpha=smiles[index]
            if index<smilesLength-1 and tempAlpha>='A' and tempAlpha<='Z':
                anotherAlpha=smiles[index+1]
                if anotherAlpha>='a' and anotherAlpha<='z':
                    elements=['He','Li','Be','Na','Br']
                    if smiles[index:index+2] in elements:
                        tempAlpha+=anotherAlpha
                        index+=1
            nameStr.append(tempAlpha)
            index+=1
        smilesSplit.append(nameStr)
    # print(smilesSplit)
    properties=np.array(esol['measured log solubility in mols per litre'])
    return np.array(smilesSplit), properties

if __name__=='__main__':
    loadEsolSmilesData()