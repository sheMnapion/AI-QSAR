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
import matplotlib.pyplot as plt

EMBED_DIMENSION = 25
LATENT_DIMENSION = 400

useGPU=torch.cuda.is_available()

class CollateFn(object):
    """part for dealing with vari-len inputs"""

    def __init__(self):
        pass

    def _collate(self, batch):
        batchX = [b[0] for b in batch]
        batchY = [b[1].item() for b in batch]
        batchY = np.array(batchY)
        trueLengths=[]
        maxLength=len(batchX[0])
        for bx in batchX:
            start=0
            for j in range(maxLength):
                if bx[j]==0:
                    break
                start+=1
            trueLengths.append(start)
        trueLengths=np.array(trueLengths)
        maxLen = max(trueLengths)
        batchSize=len(batchY)
        finalBatchX = torch.zeros(batchSize,maxLen,dtype=torch.long)
        for i in range(batchSize):
            finalBatchX[i]=batchX[i][:maxLen]
        finalBatchY = torch.tensor(batchY,dtype=torch.float32)
        # print(np.max(trueLengths),np.min(trueLengths))
        # print(finalBatchX.shape,finalBatchY.shape)
        return (finalBatchX, finalBatchY)

    def __call__(self, batch):
        return self._collate(batch)

class SmilesRnn(nn.Module):
    """class for performing regression or classification directly from smiles representation
        smiles and property pairs needed to initialize
        using simple RNN structure to finish the work
    """
    def __init__(self,nKeys):
        """RNN with only one variable, the number of keys of possible smiles representations"""
        super(SmilesRnn,self).__init__()
        self.embedding=nn.Embedding(nKeys,EMBED_DIMENSION,padding_idx=0)
        self.lstm=nn.GRU(
            input_size=EMBED_DIMENSION,
            hidden_size=LATENT_DIMENSION,
            num_layers=1,
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
    def __init__(self):
        """get smiles and property pairs for regression"""
        pass

    def initFromData(self,smiles,properties):
        """init total model from data offered by smiles and properties"""
        self.origSmiles=smiles
        self.origProperties=properties
        smileStrLength=np.array([len(s) for s in smiles])
        maxLength=np.max(smileStrLength)
        print("Max length for RNN input:",maxLength)
        self.maxLength=maxLength
        # self.net=SmilesRnn(maxLength)
        self._processData()

    def train(self,nRounds=1000,lr=0.01,earlyStop=True,earlyStopEpoch=10,batchSize=12):
        """train the RNN for [nRounds] with learning rate [lr]"""
        trainSet=TensorDataset(self.smilesTrain,self.propTrain)
        testSet=TensorDataset(self.smilesTest,self.propTest)
        print(self.smilesTest.shape)
        trainLoader=DataLoader(trainSet,batch_size=batchSize,shuffle=True,num_workers=2,collate_fn=CollateFn())
        testLoader=DataLoader(testSet,batch_size=batchSize,shuffle=False,num_workers=2,collate_fn=CollateFn())
        optimizer=optim.Adam(self.net.parameters(),lr=lr,weight_decay=1e-8)
        lossFunc=nn.MSELoss()
        consecutiveRounds=0
        bestR2=-1.0
        start=time.time()
        for epoch in range(nRounds):
            losses=[]
            for i, (x,y) in enumerate(trainLoader):
                y.unsqueeze_(1) #; y.unsqueeze_(1)
                if useGPU==True:
                    x=x.cuda(); y=y.cuda()
                prediction=self.net(x)
                loss=lossFunc(prediction,y)
                losses.append(loss.item())
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            losses=np.array(losses)
            valLosses=[]; pred=[]; true=[]
            for i, (x,y) in enumerate(testLoader):
                with torch.no_grad():
                    y.unsqueeze_(1) #; y.unsqueeze_(1)
                    if useGPU==True:
                        x=x.cuda(); y=y.cuda()
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
            if earlyStop==True:
                if tempR2Score>bestR2:
                    consecutiveRounds=0
                    bestR2=tempR2Score
                    tempData=[self.net.state_dict(),self.decodeDict]
                    torch.save(tempData,'/tmp/tmpRNNState.pt')
                else:
                    consecutiveRounds+=1
                    if consecutiveRounds>=earlyStopEpoch:
                        print("No better performance after %d rounds, break." % (earlyStopEpoch))
                        break
        print("Best r2 score:",bestR2)

    def loadFromModel(self,modelPath):
        """load model from given path"""
        if useGPU==False:
            modelData=torch.load(modelPath,map_location='cpu')
        else:
            modelData=torch.load(modelPath,map_location='cuda:0')
        stateDict=modelData[0]
        self.decodeDict=modelData[1]
        self.recodeDict=dict([(self.decodeDict[i],i) for i in self.decodeDict.keys()])
        # print(self.decodeDict)
        # print(self.recodeDict)
        self.nKeys=len(self.decodeDict.keys())
        self.net=SmilesRnn(self.nKeys)
        self.net.load_state_dict(stateDict)

    def predict(self,smiles,batchSize=50):
        """make prediction on test smiles strings"""
        smilesSplit=self._parseSmiles(smiles)
        smilesSplit=np.array(smilesSplit)
        lengths=[len(sS) for sS in smilesSplit]
        maxLength=max(lengths)
        nTests=len(smiles)
        padData=np.zeros((nTests,maxLength))
        for i, smiles in enumerate(smilesSplit):
            for j, s in enumerate(smiles):
                padData[i][j]=self.recodeDict[s]
        print(padData)
        padData=torch.from_numpy(padData)
        padProp=torch.zeros(nTests)
        testSet=TensorDataset(padData,padProp)
        testLoader=DataLoader(testSet,batch_size=batchSize,shuffle=False,num_workers=1,collate_fn=CollateFn())
        preds=[]
        for (x,y) in testLoader:
            tempBatchSize=len(y)
            if useGPU==True:
                x=x.cuda(); y=y.cuda()
            with torch.no_grad():
                pred=self.net(x)
                for i in range(tempBatchSize):
                    preds.append(pred[i][0].item())
        preds=np.array(preds)
        return preds

    def _parseSmiles(self,testSmiles):
        """parse smiles and get its valid representation"""
        smilesSplit = []
        print(testSmiles)
        for smiles in testSmiles:
            smiles=''.join(smiles)
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
        return smilesSplit

    def _processData(self):
        """
            padding data for smile strings to have same input length;
            encode the input string to have unitary coding format
            also split data
        """
        # split first
        smilesSplit=self._parseSmiles(self.origSmiles)
        self.origSmiles=np.array(smilesSplit)
        recodeDict=dict([('@',0)])
        dictKey=1
        nItems=len(self.origSmiles)
        self.origLengths=np.array([len(origSmile) for origSmile in self.origSmiles])
        self.maxLength=np.max(self.origLengths)
        padData=[]
        for i, s in enumerate(self.origSmiles):
            tempData=[]
            for j, sChar in enumerate(s):
                if sChar not in recodeDict.keys():
                    recodeDict[sChar]=dictKey
                    dictKey+=1
                tempData.append(recodeDict[sChar])
            padData.append(tempData)
        padData=np.array(padData)
        # print(padData)
        print(nItems,dictKey,recodeDict)
        self.nKeys=dictKey
        self.recodeDict=recodeDict
        self.net=SmilesRnn(self.nKeys)
        if useGPU==True:
            self.net=self.net.cuda()
        self.decodeDict=dict([(recodeDict[key],key) for key in recodeDict.keys()])
        self.standardData=padData
        smilesTrain,smilesTest,propTrain,propTest=train_test_split(padData,self.origProperties,test_size=0.2,random_state=2019)
        self.origSmilesTrain=smilesTrain # this stores smiles sequences for molecular design rip out usage
        nTrain=len(smilesTrain); nTest=len(smilesTest)
        print("Train test #:",nTrain,nTest)
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
            trainLengths=np.array([len(sT) for sT in smilesTrain])
            trainMaxLength=np.max(trainLengths)
            nTrain=len(trainLengths)
            trainPadData=np.zeros((nTrain,trainMaxLength))
            for i in range(nTrain):
                trainPadData[i][:trainLengths[i]]=smilesTrain[i]
            self.smilesTrain=torch.tensor(trainPadData,dtype=torch.long)
            testLengths=np.array([len(sT) for sT in smilesTest])
            testMaxLength=np.max(testLengths)
            nTest=len(testLengths)
            testPadData=np.zeros((nTest,testMaxLength))
            for i in range(nTest):
                testPadData[i][:testLengths[i]]=smilesTest[i]
            self.smilesTest=torch.tensor(testPadData,dtype=torch.long)
        # print(self.smilesTrain)
        # print(self.smilesTest)
        self.propTrain=torch.tensor(propTrain,dtype=torch.float32)
        self.propTest=torch.tensor(propTest,dtype=torch.float32)

if __name__=='__main__':
    smiles,properties=loadEsolSmilesData()
    predictor=SmilesRNNPredictor()
    # predictor.initFromData(smiles,properties)
    # predictor.train(nRounds=1000,lr=3e-4,batchSize=50)
    predictor.loadFromModel('tmpRNN_0.88.pt')
    predictor.predict(smiles)
