import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from dataProcess import loadEsolSmilesData
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from torch.utils.data import DataLoader, Dataset, TensorDataset

class SmilesCNN(nn.Module):
    """class for performing regression or classification directly from smiles representation
        smiles and property pairs needed to initialize
        using CNN structure to finish the work
    """
    def __init__(self):
        """RNN with only one variable, the maximal length of a possible smiles"""
        super(SmilesCNN,self).__init__()
        self.conv1=nn.Conv2d(1,2,7,dilation=1)
        self.conv2=nn.Conv2d(2,3,7,dilation=2)
        self.fc1=nn.Linear(867,300)
        self.fc2=nn.Linear(300,1)

    def forward(self, x):
        """pass on hidden state as well"""
        x=F.max_pool2d(self.conv1(x),(2,2))
        x=F.max_pool2d(self.conv2(x),(2,2))
        # print(x.shape)
        x = x.view(-1, self.num_flat_features(x))
        # print(x.shape)
        x= F.relu(self.fc1(x))
        x=self.fc2(x)
        return x

    def num_flat_features(self,x):
        size=x.size()[1:] # all dimensions except the batch dimension
        num_features=1
        for s in size:
            num_features *= s
        return num_features

class SmilesCNNPredictor(object):
    """wrapper class for receiving data and training"""
    def __init__(self,smiles,properties):
        """get smiles and property pairs for regression"""
        self.origSmiles=smiles
        self.origProperties=properties
        smileStrLength=np.array([len(s) for s in smiles])
        maxLength=np.max(smileStrLength)
        print("Max length for RNN input:",maxLength)
        self.maxLength=maxLength
        self.net=SmilesCNN()
        self._processData()

    def train(self,nRounds=1000,lr=0.01,earlyStopEpoch=10,batchSize=12):
        """train the RNN for [nRounds] with learning rate [lr]"""
        trainSet=TensorDataset(self.smilesTrain,self.propTrain)
        testSet=TensorDataset(self.smilesTest,self.propTest)
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
                y.unsqueeze_(1) # ; y.unsqueeze_(1)
                prediction=self.net(x)
                loss=lossFunc(prediction,y)
                losses.append(loss.item())
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            losses=np.array(losses)
            valLosses=[]; pred=[]; true=[]
            with torch.no_grad():
                for i, (x,y) in enumerate(testLoader):
                    x.unsqueeze_(1)
                    y.unsqueeze_(1) #; y.unsqueeze_(1)
                    prediction=self.net(x)
                    for i, p in enumerate(prediction[0]):
                        pred.append(p)
                        true.append(y[0][i])
                    loss=lossFunc(prediction,y)
                    valLosses.append(loss.item())
            print("Round [%d]: {%.5f,%.5f}" % (epoch+1,np.mean(losses),np.mean(valLosses)))
            tempR2Score=r2_score(true,pred)
            print("r^2 score:",tempR2Score)
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
        padData=np.zeros((nItems,self.maxLength),dtype=np.int32)
        for i, s in enumerate(self.origSmiles):
            for j, sChar in enumerate(s):
                if sChar not in recodeDict.keys():
                    recodeDict[sChar]=dictKey
                    dictKey+=1
                padData[i][j]=recodeDict[sChar]
        # print(padData,padData.shape)
        print(nItems,recodeDict)
        self.nWords=len(recodeDict)
        self.embedding=nn.Embedding(self.nWords+1,self.maxLength,padding_idx=0)
        self.standardData=padData
        smilesTrain,smilesTest,propTrain,propTest=train_test_split(padData,self.origProperties,test_size=0.2,random_state=2019)
        nTrain=len(smilesTrain); nTest=len(smilesTest)
        print("Train test #:",nTrain,nTest)
        self.smilesTrain=torch.zeros(nTrain,self.maxLength,self.maxLength,dtype=torch.float32)
        self.smilesTest=torch.zeros(nTest,self.maxLength,self.maxLength,dtype=torch.float32)
        with torch.no_grad():
            for i in range(nTrain):
                self.smilesTrain[i]=self.embedding(torch.Tensor(smilesTrain[i]).to(torch.long))
            for i in range(nTest):
                self.smilesTest[i]=self.embedding(torch.Tensor(smilesTest[i]).to(torch.long))
        self.propTrain=torch.tensor(propTrain,dtype=torch.float32)
        self.propTest=torch.tensor(propTest,dtype=torch.float32)
        print("Dataset prepared.")

if __name__=='__main__':
    smiles,properties=loadEsolSmilesData()
    predictor=SmilesCNNPredictor(smiles,properties)
    predictor.train(nRounds=100,lr=3e-4)