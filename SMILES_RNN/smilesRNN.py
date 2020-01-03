import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from dataProcess import loadEsolSmilesData
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from torch.utils.data import DataLoader, Dataset, TensorDataset
import time

EMBED_DIMENSION = 25
LATENT_DIMENSION = 400

class SmilesRnn(nn.Module):
    """class for performing regression or classification directly from smiles representation
        smiles and property pairs needed to initialize
        using simple RNN structure to finish the work
    """
    def __init__(self,maxLength):
        """RNN with only one variable, the maximal length of a possible smiles"""
        super(SmilesRnn,self).__init__()
        self.embedding=nn.Embedding(33,EMBED_DIMENSION,padding_idx=0)
        self.lstm=nn.GRU(
            input_size=EMBED_DIMENSION,
            hidden_size=LATENT_DIMENSION,
            num_layers=2,
            batch_first=True,
            bidirectional=False
        )
        self.fc1=nn.Linear(LATENT_DIMENSION,LATENT_DIMENSION)
        self.fc2=nn.Linear(LATENT_DIMENSION,1)

    def forward(self, x):
        """pass on hidden state as well"""
        x=self.embedding(x)
        out, _=self.lstm(x)
        out= out[:, -1, :]
        x=F.relu(self.fc1(out))
        x=self.fc2(x)
        return x

class SmilesRNNPredictor(object):
    """wrapper class for receiving data and training"""
    def __init__(self,smiles,properties):
        """get smiles and property pairs for regression"""
        self.origSmiles=smiles
        self.origProperties=properties
        smileStrLength=np.array([len(s) for s in smiles])
        maxLength=np.max(smileStrLength)
        print("Max length for RNN input:",maxLength)
        self.maxLength=maxLength
        self.net=SmilesRnn(maxLength)
        self._processData()

    def train(self,nRounds=1000,lr=0.01,earlyStopEpoch=10,batchSize=12):
        """train the RNN for [nRounds] with learning rate [lr]"""
        trainSet=TensorDataset(self.smilesTrain,self.propTrain)
        testSet=TensorDataset(self.smilesTest,self.propTest)
        print(self.smilesTest.shape)
        trainLoader=DataLoader(trainSet,batch_size=batchSize,shuffle=True,num_workers=2)
        testLoader=DataLoader(testSet,batch_size=batchSize,shuffle=False,num_workers=2)
        optimizer=optim.Adam(self.net.parameters(),lr=lr,weight_decay=1e-8)
        lossFunc=nn.MSELoss()
        consecutiveRounds=0
        bestR2=-1.0
        start=time.time()
        for epoch in range(nRounds):
            losses=[]
            for i, (x,y) in enumerate(trainLoader):
                y.unsqueeze_(1) #; y.unsqueeze_(1)
                prediction=self.net(x)
                loss=lossFunc(prediction,y)
                losses.append(loss.item())
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            losses=np.array(losses)
            valLosses=[]; pred=[]; true=[]
            for i, (x,y) in enumerate(testLoader):
                y.unsqueeze_(1) #; y.unsqueeze_(1)
                prediction=self.net(x)
                # print(y.shape,prediction.shape)
                # print(prediction,prediction.shape)
                for j, p in enumerate(prediction[:,0]):
                    pred.append(p)
                    true.append(y[j][0])
                loss=lossFunc(prediction,y)
                valLosses.append(loss.item())
            print("Round [%d]: {%.5f,%.5f} total time: %.5f seconds" % (epoch+1,np.mean(losses),np.mean(valLosses),time.time()-start))
            tempR2Score=r2_score(true,pred)
            print("r^2 score:",tempR2Score)
            import matplotlib.pyplot as plt
            pred=np.array(pred); true=np.array(true)
            print(pred.shape,true.shape)
            # plt.scatter(pred,true)
            minX=min(np.min(pred),np.min(true))
            maxX=max(np.max(pred),np.max(true))
            # plotX=np.linspace(minX,maxX,1000)
            # plt.plot(plotX,plotX)
            # plt.show()
            if tempR2Score>bestR2:
                consecutiveRounds=0
                bestR2=tempR2Score
            else:
                consecutiveRounds+=1
                if consecutiveRounds>=earlyStopEpoch:
                    print("No better performance after %d rounds, break." % (earlyStopEpoch))
                    break
        print("Best r2 score:",bestR2)

    def _processData(self):
        """
            padding data for smile strings to have same input length;
            encode the input string to have unitary coding format
            also split data
        """
        # split first
        smilesSplit = []
        for smiles in self.origSmiles:
            smilesLength = len(smiles)
            nameStr = []
            index = 0
            while index < smilesLength:
                tempAlpha = smiles[index]
                if index < smilesLength - 1 and ((tempAlpha >= 'A' and tempAlpha <= 'Z') or (tempAlpha>='a' and tempAlpha<='z')):
                    anotherAlpha = smiles[index + 1]
                    if anotherAlpha == ' ':  # error, need cleaning
                        index += 1
                        continue
                    if anotherAlpha >= 'a' and anotherAlpha <= 'z':
                        elements = ['He', 'Li', 'Be','Ne', 'Na','Mg','Al','Si','Br','Ar','Ca','Sc','Ti',
                                    'Cl','Cr','Mn','Fe','Co','Ni','Cu','Zn','Ga','Ge','As','Se','Kr',
                                    'Rb','Sr','Zr','Nb','Mo','Tc','Ru','Rh','Pd','Ag','Cd','In','Sn',
                                    'Sb','Te','Xe','Cs','Ba','Lu','Hf','Ta','Re','Os','Ir','Pt','Au',
                                    'Hg','Tl','Pb','Bi','Po','At','Rn','La','Ce','Pr','Nd','Pm','Sm',
                                    'Eu','Gd','Tb','Dy','Ho','Er','Tm','Yb','se','cl']
                        if smiles[index:index + 2] in elements:
                            tempAlpha += anotherAlpha
                            index += 1
                elif tempAlpha=='n':
                    print(smiles)
                    exit(0)
                nameStr.append(tempAlpha)
                index += 1
            smilesSplit.append(nameStr)
        self.origSmiles=np.array(smilesSplit)
        recodeDict=dict([('@',0)])
        dictKey=1
        nItems=len(self.origSmiles)
        padData=np.zeros((nItems,self.maxLength),dtype=np.int32)
        for i, s in enumerate(self.origSmiles):
            for j, sChar in enumerate(s):
                if sChar not in recodeDict.keys():
                    recodeDict[sChar]=dictKey
                    dictKey+=1
                padData[i][j]=recodeDict[sChar]
        # print(padData,padData.shape)
        print(nItems,dictKey,recodeDict)
        self.nKeys=dictKey
        self.decodeDict=dict([(recodeDict[key],key) for key in recodeDict.keys()])
        self.standardData=padData
        smilesTrain,smilesTest,propTrain,propTest=train_test_split(padData,self.origProperties,test_size=0.2,random_state=2019)
        self.origSmilesTrain=smilesTrain # this stores smiles sequences for molecular design rip out usage
        nTrain=len(smilesTrain); nTest=len(smilesTest)
        print("Train test #:",nTrain,nTest)
        smilesTrain=torch.tensor(smilesTrain).to(torch.long)
        smilesTest=torch.tensor(smilesTest).to(torch.long)
        matrixEncode=False
        if matrixEncode is True:
            self.smilesTrain=torch.zeros(nTrain,self.maxLength,self.nKeys,dtype=torch.float32)
            self.smilesTest=torch.zeros(nTest,self.maxLength,self.nKeys,dtype=torch.float32)
            tempMatrix=np.array(torch.randn(self.nKeys,self.nKeys))
            Q,R=np.linalg.qr(tempMatrix)
            Q=torch.from_numpy(Q)
            with torch.no_grad():
                for i in range(nTrain):
                    for j, s in enumerate(smilesTrain[i]):
                        self.smilesTrain[i][j]=Q[int(s)]
                for i in range(nTest):
                    for j, s in enumerate(smilesTest[i]):
                        self.smilesTest[i][j]=Q[int(s)]
        else:
            self.smilesTrain=smilesTrain #torch.zeros(nTrain,self.maxLength-1,dtype=torch.long)
            self.smilesTest=smilesTest #torch.zeros(nTest,self.maxLength-1,dtype=torch.long)
        self.propTrain=torch.tensor(propTrain,dtype=torch.float32)
        self.propTest=torch.tensor(propTest,dtype=torch.float32)
        print("Dataset prepared, shapes are",self.smilesTrain.shape,self.smilesTest.shape,self.propTrain.shape,self.propTest.shape)

if __name__=='__main__':
    smiles,properties=loadEsolSmilesData()
    predictor=SmilesRNNPredictor(smiles,properties)
    predictor.train(nRounds=1000,lr=3e-4,batchSize=80)