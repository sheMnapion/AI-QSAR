import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from rdkit import Chem

def loadAugData():
    """load data from desc_canvas_aug30 dataset"""
    augData=pd.read_csv('../datasets/desc_canvas_aug30.csv')
    columns=augData.columns
    augDataTrain=augData[augData['Model']=='Train']
    augDataTrainX=augDataTrain[columns[5:-1]]
    augDataTrainY=augDataTrain[columns[4]]
    augDataTest=augData[augData['Model']=='Test']
    augDataTestX=augDataTest[columns[5:-1]]
    augDataTestY=augDataTest[columns[4]]
    augDataValid=augData[augData['Model']=='Valid']
    augDataValidX=augDataValid[columns[5:-1]]
    augDataValidY=augDataValid[columns[4]]
    return np.array(augDataTrainX), np.array(augDataTrainY), np.array(augDataTestX), np.array(augDataTestY), \
        np.array(augDataValidX), np.array(augDataValidY)

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
                if anotherAlpha==' ': # error, need cleaning
                    index+=1
                    continue
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
    # loadEsolSmilesData()
    trainX,trainY,testX,testY,valX,valY=loadAugData()
    from sklearn.linear_model import LinearRegression
    from sklearn.ensemble import RandomForestRegressor
    tempRegressor=RandomForestRegressor(n_estimators=200,max_features='log2',verbose=True,n_jobs=2)
    tempRegressor.fit(testX,testY)
    pred=tempRegressor.predict(trainX)
    valScore=r2_score(trainY,pred)
    print(valScore)
    plt.scatter(trainY,pred); plt.show()
