import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from dataProcess import loadEsolSmilesData
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from torch.utils.data import DataLoader, Dataset, TensorDataset

class SmilesRnn(nn.Module):
    """class for performing regression or classification directly from smiles representation
        smiles and property pairs needed to initialize
        using simple RNN structure to finish the work
    """
    def __init__(self,maxLength):
        """RNN with only one variable, the maximal length of a possible smiles"""
        super(SmilesRnn,self).__init__()
        self.rnn=nn.RNN(
            input_size=maxLength,
            hidden_size=512,
            num_layers=2,
            bidirectional=False
        )
        self.out=nn.Linear(512,1)

    def forward(self, x, h_state):
        """pass on hidden state as well"""
        out, h=self.rnn(x, h_state)
        prediction=self.out(out)
        return prediction, h

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
        for epoch in range(nRounds):
            losses=[]
            for i, (x,y) in enumerate(trainLoader):
                x.unsqueeze_(1)
                y.unsqueeze_(1); y.unsqueeze_(1)
                prediction, hState=self.net(x,None)
                loss=lossFunc(prediction,y)
                losses.append(loss.item())
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            losses=np.array(losses)
            valLosses=[]; pred=[]; true=[]
            for i, (x,y) in enumerate(testLoader):
                x.unsqueeze_(1)
                y.unsqueeze_(1); y.unsqueeze_(1)
                prediction, hState=self.net(x,None)
                # print(prediction,prediction.shape)
                for j, p in enumerate(prediction[:,0,0]):
                    pred.append(p)
                    true.append(y[j][0][0])
                loss=lossFunc(prediction,y)
                valLosses.append(loss.item())
            print("Round [%d]: {%.5f,%.5f}" % (epoch+1,np.mean(losses),np.mean(valLosses)))
            tempR2Score=r2_score(true,pred)
            print("r^2 score:",tempR2Score)
            # import matplotlib.pyplot as plt
            # pred=np.array(pred); true=np.array(true)
            # print(pred.shape,true.shape)
            # plt.scatter(pred,true)
            # minX=min(np.min(pred),np.min(true))
            # maxX=max(np.max(pred),np.max(true))
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
        """padding data for smile strings to have same input length;
            encode the input string to have unitary coding format
            also split data
        """
        recodeDict=dict(); dictKey=1
        nItems=len(self.origSmiles)
        padData=np.zeros((nItems,self.maxLength),dtype=np.float32)
        for i, s in enumerate(self.origSmiles):
            for j, sChar in enumerate(s):
                if sChar not in recodeDict.keys():
                    recodeDict[sChar]=dictKey
                    dictKey+=1
                padData[i][j]=recodeDict[sChar]
        # print(padData,padData.shape)
        print(nItems,recodeDict)
        self.nWords=len(recodeDict)
        self.embedding=nn.Embedding(self.nWords,10,padding_idx=0)
        self.standardData=padData
        smilesTrain,smilesTest,propTrain,propTest=train_test_split(padData,self.origProperties,test_size=0.2,random_state=2019)
        self.smilesTrain=torch.tensor(smilesTrain,dtype=torch.float32)
        self.smilesTest=torch.tensor(smilesTest,dtype=torch.float32)
        self.propTrain=torch.tensor(propTrain,dtype=torch.float32)
        self.propTest=torch.tensor(propTest,dtype=torch.float32)
        print("Dataset prepared.")

if __name__=='__main__':
    smiles,properties=loadEsolSmilesData()
    predictor=SmilesRNNPredictor(smiles,properties)
    predictor.train(nRounds=100,lr=1e-3)