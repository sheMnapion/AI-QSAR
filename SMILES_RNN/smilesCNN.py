import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from dataProcess import loadEsolSmilesData
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
from torch.utils.data import DataLoader, Dataset, TensorDataset
import time

class SmilesCNNVAE(nn.Module):
    """
        class for performing regression/classification and generating new medicines directly
        from smiles representation; using CNN structure to finish the encode and decode
        part
    """
    def __init__(self,featureNum):
        """CNN structures"""
        super(SmilesCNNVAE,self).__init__()
        self.featureNum=featureNum
        self.conv1=nn.Conv2d(1,2,7,dilation=1)
        self.avgPool1=nn.AvgPool2d(2)
        self.conv2=nn.Conv2d(2,3,7,dilation=2)
        self.avgPool2=nn.AvgPool2d(2)
        self.unMaxPool3=nn.MaxUnpool2d(2)
        self.conv3=nn.ConvTranspose2d(3,2,7,dilation=2)
        self.unMaxPool4=nn.MaxUnpool2d(2)
        self.conv4=nn.ConvTranspose2d(2,1,7,dilation=1)
        self.fc1=nn.Linear(867,256)
        self.fc21=nn.Linear(256,100)
        self.fc22=nn.Linear(256,100)
        self.fc3=nn.Linear(100,256)
        self.fc4=nn.Linear(256,867)

    def num_flat_features(self,x):
        size=x.size()[1:] # all dimensions except the batch dimension
        num_features=1
        for s in size:
            num_features *= s
        return num_features

    def encode(self, x):
        """encode the input into two parts, mean mu and log variance"""
        x=self.avgPool1(self.conv1(x))
        x=self.avgPool2(self.conv2(x))
        x=x.view(-1,self.num_flat_features(x))
        x=F.relu(self.fc1(x))
        return self.fc21(x), self.fc22(x)

    def decode(self, z):
        """decode the inner representation vibrated with random noise to the original size"""
        batchSize=z.shape[0]
        z=F.relu(self.fc3(z))
        z=F.relu(self.fc4(z))
        z=z.view(batchSize,3,17,17)
        z=self.conv3(F.interpolate(z,scale_factor=2))
        z=self.conv4(F.interpolate(z,scale_factor=2))
        return torch.sigmoid(z) # temporarily take this as the binary vector format

    def reparameterize(self, mu, logvar):
        """re-parameterization trick in training the net"""
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

    def middleRepresentation(self, x):
        """return representation in latent space"""
        mu, logVar = self.encode(x)
        return mu

def vaeLossFunc(reconstructedX, x, mu, logvar):
    # BCE = F.mse_loss(reconstructedX, x)
    BCE = F.binary_cross_entropy(reconstructedX, x, reduction='sum')
    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return BCE + KLD

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
        self.fc1=nn.Linear(867,400)
        self.fc2=nn.Linear(400,200)
        self.fc3=nn.Linear(200,1)

    def forward(self, x):
        """pass on hidden state as well"""
        x=F.max_pool2d(self.conv1(x),(2,2))
        x=F.max_pool2d(self.conv2(x),(2,2))
        # print(x.shape)
        x = x.view(-1, self.num_flat_features(x))
        # print(x.shape)
        x= F.relu(self.fc1(x))
        x= F.relu(self.fc2(x))
        x= self.fc3(x)
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
        self.vaeNet=SmilesCNNVAE(maxLength)
        self._processData()

    def loadVAE(self,path):
        """load trained VAE from given path"""
        self.vaeNet.load_state_dict(torch.load(path))
        print("VAE LOADED")

    def trainVAE(self,nRounds=1000,lr=0.01,earlyStop=True,earlyStopEpoch=10,batchSize=12):
        """train the vae net"""
        trainSet=TensorDataset(self.smilesTrain,self.propTrain)
        testSet=TensorDataset(self.smilesTest,self.propTest)
        trainLoader=DataLoader(trainSet,batch_size=batchSize,shuffle=True,num_workers=2)
        testLoader=DataLoader(testSet,batch_size=batchSize,shuffle=False,num_workers=2)
        optimizer=optim.Adam(self.vaeNet.parameters(),lr=lr,weight_decay=1e-8)
        consecutiveRounds=0
        bestLoss=1e9
        start=time.time()
        for epoch in range(nRounds):
            losses=[]
            for i, (x,y) in enumerate(trainLoader):
                x.unsqueeze_(1)
                y.unsqueeze_(1) # ; y.unsqueeze_(1)
                reconstructedX,mu,logVar=self.vaeNet(x)
                loss=vaeLossFunc(reconstructedX,x,mu,logVar)
                # print(loss)
                losses.append(loss.item())
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            losses=np.array(losses)
            print("Train mean loss:",np.mean(losses))
            valLosses=[]; pred=[]; true=[]
            with torch.no_grad():
                for i, (x,y) in enumerate(testLoader):
                    x.unsqueeze_(1)
                    y.unsqueeze_(1) #; y.unsqueeze_(1)
                    reconstructedX,mu,logVar=self.vaeNet(x)
                    loss=vaeLossFunc(reconstructedX,x,mu,logVar)
                    valLosses.append(loss.item())
            valLoss=np.mean(valLosses)
            if bestLoss>valLoss:
                consecutiveRounds=0
                bestLoss=valLoss
                torch.save(self.vaeNet.state_dict(),'tmp/tmpBestModel.pt')
            else:
                consecutiveRounds+=1
                if consecutiveRounds>=earlyStopEpoch and earlyStop:
                    print("No better performance after %d rounds, break." % (earlyStopEpoch))
                    break
            print("Round [%d]: {%.5f,%.5f|%.5f} after %.3f seconds" % (epoch+1,np.mean(losses),np.mean(valLosses),bestLoss,time.time()-start))
        print("Best validation loss:",bestLoss)

    def encodeDataset(self):
        """encode dataset with trained VAE to obtain its representation in latent space"""
        trainSet=self.smilesTrain
        trainSet.unsqueeze_(1)
        testSet=self.smilesTest
        testSet.unsqueeze_(1)
        trainRet=self.vaeNet.middleRepresentation(trainSet)
        print(trainRet.shape)
        self.trainRepr=trainRet
        testRet=self.vaeNet.middleRepresentation(testSet)
        print(testRet.shape)
        self.testRepr=testRet

    def trainLatentModel(self):
        """train the prediction model within latent space"""
        from sklearn.neural_network import MLPRegressor
        from sklearn.ensemble import RandomForestRegressor
        import matplotlib.pyplot as plt
        tempRegressor=RandomForestRegressor(n_estimators=1000,n_jobs=2,verbose=True)
        npTrainLatent=self.trainRepr.detach().numpy()
        npTestLatent=self.testRepr.detach().numpy()
        npTrainLabel=self.propTrain.detach().numpy()
        npTestLabel=self.propTest.detach().numpy()
        print(npTrainLatent.shape,npTrainLabel.shape)
        print(npTestLatent.shape,npTestLabel.shape)
        tempRegressor.fit(npTrainLatent,npTrainLabel)
        pred=tempRegressor.predict(npTestLatent)
        print(pred.shape,npTestLabel.shape)
        score=r2_score(npTestLabel,pred)
        plt.scatter(npTestLabel,pred)
        print("r2 score:",score)
        plt.show()

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
                    # print(prediction.shape)
                    for i, p in enumerate(prediction[:,0]):
                        pred.append(p)
                        true.append(y[i][0])
                    loss=lossFunc(prediction,y)
                    valLosses.append(loss.item())
            print("Round [%d]: {%.5f,%.5f}" % (epoch+1,np.mean(losses),np.mean(valLosses)))
            true=np.array(true); pred=np.array(pred)
            print(true.shape,pred.shape)
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
        tempMatrix=np.array(torch.randn(self.maxLength,self.maxLength))
        Q,R=np.linalg.qr(tempMatrix)
        Q=torch.from_numpy(Q)
        with torch.no_grad():
            for i in range(nTrain):
                for j, s in enumerate(smilesTrain[i]):
                # self.smilesTrain[i]=self.embedding(torch.Tensor(smilesTrain[i]).to(torch.long))
                # print(self.smilesTrain[i])
                # print(smilesTrain[i])
                    self.smilesTrain[i][j][int(s)]=1.0
                    # self.smilesTrain[i][j]=Q[int(s)]
                    if s==0.0: break
                # print(self.smilesTrain[i])
                # print(smilesTrain[i])
            for i in range(nTest):
                for j, s in enumerate(smilesTest[i]):
                # self.smilesTest[i]=self.embedding(torch.Tensor(smilesTest[i]).to(torch.long))
                    # self.smilesTest[i][j]=Q[int(s)]
                    self.smilesTest[i][j][int(s)]=1.0
                    if s==0.0: break
        self.propTrain=torch.tensor(propTrain,dtype=torch.float32)
        self.propTest=torch.tensor(propTest,dtype=torch.float32)
        print("Dataset prepared.")

if __name__=='__main__':
    smiles,properties=loadEsolSmilesData()
    predictor=SmilesCNNPredictor(smiles,properties)
    # predictor.train(nRounds=100,lr=3e-4)
    # predictor.trainVAE(nRounds=1000,lr=1e-3,earlyStop=True,earlyStopEpoch=10,batchSize=30)
    predictor.loadVAE('tmp/tmpBestModel.pt')
    predictor.encodeDataset()
    predictor.trainLatentModel()