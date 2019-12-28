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
import matplotlib.pyplot as plt
import copy
from rdkit import Chem
from rdkit.Chem import Draw

class SmilesCNNVAE(nn.Module):
    """
        class for performing regression/classification and generating new medicines directly
        from smiles representation; using embedding structure as the core to encode SMILES sequence
        and decode accordingly
    """
    def __init__(self,featureNum):
        """CNN structures"""
        super(SmilesCNNVAE,self).__init__()
        self.featureNum=featureNum
        self.embedding=nn.Embedding(32,32)
        self.fc1=nn.Linear(3200,1000)
        self.fc21=nn.Linear(1000,200)
        self.fc22=nn.Linear(1000,200)
        self.fc3=nn.Linear(200,1000)
        self.fc4=nn.Linear(1000,3200)
        self.decodeFC=nn.Linear(32,32)

    def num_flat_features(self,x):
        size=x.size()[1:] # all dimensions except the batch dimension
        num_features=1
        for s in size:
            num_features *= s
        return num_features

    def encode(self, x):
        """encode the input into two parts, mean mu and log variance"""
        x=self.embedding(x)
        x=x.view(-1,self.num_flat_features(x))
        x=F.leaky_relu(self.fc1(x))
        return self.fc21(x), self.fc22(x)

    def decode(self, z):
        """decode the inner representation vibrated with random noise to the original size"""
        batchSize=z.shape[0]
        z=F.leaky_relu(self.fc3(z))
        z=F.leaky_relu(self.fc4(z))
        z=z.view(-1,32)
        z=self.decodeFC(z)
        z=z.view(batchSize,-1,32)
        return z # temporarily take this as the binary vector format

    def reparameterize(self, mu, logvar):
        """re-parameterization trick in training the net"""
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar, self.embedding(x) # standard version also included

    def middleRepresentation(self, x):
        """return representation in latent space"""
        mu, logVar = self.encode(x)
        return self.reparameterize(mu,logVar)

def vaeLossFunc(reconstructedX, x, mu, logvar):
    batchSize=x.shape[0]
    reconstructedX=reconstructedX.view(-1,32)
    x=x.view(-1)
    # print(reconstructedX.shape,x.shape)
    # BCE = F.binary_cross_entropy(reconstructedX, x, reduction='sum')
    loss=nn.CrossEntropyLoss()
    BCE=loss(reconstructedX,x)
    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    print(BCE,KLD)
    return BCE + KLD

class DNNRegressor(nn.Module):
    """
        class for performing regression with deep neural networks
        has to use pytorch to enable optimization
    """
    def __init__(self):
        """regressor on given data dimension (100)"""
        super(DNNRegressor,self).__init__()
        self.fc1=nn.Linear(200,200)
        self.fc2=nn.Linear(200,200)
        self.fc3=nn.Linear(200,200)
        self.fc4=nn.Linear(200,1)

    def forward(self, x):
        x=F.relu(self.fc1(x))
        x=F.relu(self.fc2(x))
        x=F.relu(self.fc3(x))
        return self.fc4(x)

class SmilesCNN(nn.Module):
    """class for performing regression or classification directly from smiles representation
        smiles and property pairs needed to initialize
        using CNN structure to finish the work
    """
    def __init__(self):
        """RNN with only one variable, the maximal length of a possible smiles"""
        super(SmilesCNN,self).__init__()
        self.embedding=nn.Embedding(32,30)
        # self.conv1=nn.Conv2d(1,2,7,dilation=1)
        # self.conv2=nn.Conv2d(2,3,7,dilation=2)
        self.fc1=nn.Linear(3000,1000)
        self.fc2=nn.Linear(1000,200)
        self.fc3=nn.Linear(200,1)

    def forward(self, x):
        """pass on hidden state as well"""
        # print(x.shape)
        x=self.embedding(x)
        # print(x.shape)
        x=x.view(-1,self.num_flat_features(x))

        # x=F.max_pool2d(self.conv1(x),(2,2))
        # x=F.max_pool2d(self.conv2(x),(2,2))
        # print(x.shape)
        # x = x.view(-1, self.num_flat_features(x))
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
        self.maxLength=100 # int(np.ceil(maxLength/16.0)*16)
        print("Pad to %d" % (self.maxLength))
        self.net=SmilesCNN()
        self.vaeNet=SmilesCNNVAE(self.maxLength)
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
                w,h=x.shape[:2]
                aimX=torch.zeros(w,h,32,dtype=torch.float32)
                for j in range(w):
                    for k in range(h):
                        aimX[j][k][int(x[j][k])]=1.0
                x.unsqueeze_(1)
                y.unsqueeze_(1) # ; y.unsqueeze_(1)
                reconstructedX,mu,logVar,origX=self.vaeNet(x)
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
                    w,h = x.shape[:2]
                    aimX = torch.zeros(w, h, 32, dtype=torch.float32)
                    for j in range(w):
                        for k in range(h):
                            aimX[j][k][int(x[j][k])] = 1.0
                    x.unsqueeze_(1)
                    y.unsqueeze_(1) #; y.unsqueeze_(1)
                    reconstructedX,mu,logVar,origX=self.vaeNet(x)
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
        print('Train repr shape:',trainRet.shape)
        self.trainRepr=trainRet.detach().clone()
        testRet=self.vaeNet.middleRepresentation(testSet)
        print('Test repr shape:',testRet.shape)
        self.testRepr=testRet.detach().clone()

    def identityRatio(self):
        """check the extent of identity learned through our vae net"""
        temp=self.trainRepr
        decoded=self.vaeNet.decode(temp)
        print(decoded.shape)
        translated=torch.argmax(decoded,dim=2)
        correctRatios=[]
        w,h=self.origSmilesTrain.shape[:2]
        for i in range(w):
            orig=self.origSmilesTrain[i]
            trans=translated[i]
            # print(orig)
            # print(trans)
            correctCount=0
            for j in range(h):
                if orig[j]==0:
                    correctCount/=j
                    correctRatios.append(correctCount)
                    break
                if trans[j]==orig[j]:
                    correctCount+=1
        print(correctRatios)
        correctRatios=np.array(correctRatios)
        print('Average correct translation ratio:',np.mean(correctRatios))

    def molecularRandomDesign(self,aimNumber=100,batchSize=10):
        """design [aimNumber] molecules by randomly sampling from latent space"""
        designedNumber=0
        designedMolecules=set()
        designedMoleculesPairs=[]
        while designedNumber<aimNumber:
            latentVector=torch.randn(batchSize,1,200)
            properties=self.latentRegressor(latentVector)
            translate=self.vaeNet.decode(latentVector)
            validVectors=np.array(torch.argmax(translate,dim=2))
            translations=[]
            for validVector in validVectors:
                translation=''.join([self.decodeDict[vV] for vV in validVector if vV > 0])
                translations.append(translation)
            for i in range(batchSize):
                translation=translations[i]
                if designedNumber==aimNumber: break
                if translation in designedMolecules: continue
                try:
                    mol=Chem.MolFromSmiles(translation)
                    Draw.MolToImageFile(mol,str.format('designed/designed_%d_%.5f.png' % (designedNumber,properties[i].item())))
                    print('[%d]: %s | (%d)' % (designedNumber+1,translation,i))
                    designedMoleculesPairs.append([properties[i].item(),translation])
                    designedNumber+=1
                except ValueError as e:
                    continue
        designedMoleculesPairs=np.array(designedMoleculesPairs)
        propertyIndex=np.argsort(designedMoleculesPairs[:,0])
        designedMoleculesPairs=designedMoleculesPairs[propertyIndex]
        print(designedMoleculesPairs)

    def molecularDesign(self,lr=1e-3,rounds=1000):
        """design molecular using trained latent model"""
        nTrain=self.trainRepr.shape[0]

        latentVector=torch.randn(1,1,200,requires_grad=True)
        optimizer = optim.Adam([latentVector], lr=lr)
        bestScore = -10.0
        designedMolecules = set()
        for epoch in range(rounds):
            tempScore = self.latentRegressor(latentVector)
            optimizer.zero_grad()
            tempScore.backward()
            optimizer.step()
            # print(latentVector)
            translate = self.vaeNet.decode(latentVector)
            validVector = torch.argmax(translate[0], dim=1)
            validVector = np.array(validVector)
            # print(validVector,validVector.shape)
            translation = ''.join([self.decodeDict[vV] for vV in validVector if vV > 0])
            print(epoch,translation,tempScore)
            if translation in designedMolecules: continue
            try:
                # print(translation)
                mol = Chem.MolFromSmiles(translation)
                Draw.MolToImageFile(mol, str.format(
                    'designed/search_%d_%.5f.png' % (epoch + 1, tempScore.item())))
                print(translation)
                print("Epoch [%d]: designed molecular property %.5f" % (epoch + 1, tempScore.item()))
                designedMolecules.add(translation)
            except ValueError as e:
                continue
        exit(0)

        for i in range(nTrain):
            latentVector=torch.randn(1,1,200,requires_grad=True)
            with torch.no_grad():
                for j in range(200):
                    latentVector[0][0][j]=self.trainRepr[i][j]
            origVec=np.array(self.origSmilesTrain[i])
            origTranslation=''.join([self.decodeDict[tR] for tR in origVec if tR>0])
            # print(origTranslation)
            print("Number [%d]:" %(i+1))
            # print(latentVector,latentVector.shape)
            optimizer=optim.SGD([latentVector],lr=lr)
            bestScore=-10.0
            designedMolecules=set()
            for epoch in range(rounds):
                tempScore=self.latentRegressor(latentVector)
                optimizer.zero_grad()
                tempScore.backward()
                optimizer.step()
                # print(latentVector)
                translate=self.vaeNet.decode(latentVector)
                validVector=torch.argmax(translate[0],dim=1)
                validVector=np.array(validVector)
                # print(validVector,validVector.shape)
                translation=''.join([self.decodeDict[vV] for vV in validVector if vV>0])
                print(epoch, origTranslation, translation, tempScore)
                if translation==origTranslation or translation in designedMolecules: continue
                try:
                    mol=Chem.MolFromSmiles(translation)
                    Draw.MolToImageFile(mol,str.format('designed/%d_derivative_%d_%.5f.png' % (i+1,epoch+1,tempScore.item())))
                    print(translation)
                    print(origTranslation)
                    print("Epoch [%d]: designed molecular property %.5f" % (epoch+1,tempScore.item()))
                    designedMolecules.add(translation)
                except ValueError as e:
                    continue

    def trainLatentModel(self,lr=1e-3,batchSize=12,nRounds=1000,earlyStopEpoch=10):
        """train the prediction model within latent space"""
        npTrainX=np.array(self.trainRepr); npTrainY=np.array(self.propTrain)
        npTestX=np.array(self.testRepr); npTestY=np.array(self.propTest)
        from sklearn.ensemble import RandomForestRegressor
        tempRegressor=RandomForestRegressor(n_estimators=200,n_jobs=2,random_state=2019,verbose=True)
        tempRegressor.fit(npTrainX,npTrainY)
        predTestY=tempRegressor.predict(npTestX)
        score=r2_score(npTestY,predTestY)
        print(score)
        plt.scatter(npTestY,predTestY); plt.show()
        tempRegressor=DNNRegressor()
        trainSet=TensorDataset(self.trainRepr,self.propTrain)
        testSet=TensorDataset(self.testRepr,self.propTest)
        trainLoader=DataLoader(trainSet,batch_size=batchSize,shuffle=True)
        testLoader=DataLoader(testSet,batch_size=batchSize,shuffle=False)
        optimizer=optim.Adam(tempRegressor.parameters(),lr=lr,weight_decay=1e-8)
        lossFunc=nn.MSELoss()
        consecutiveRounds=0
        bestR2=-1.0
        for epoch in range(nRounds):
            losses=[]
            for i, (x,y) in enumerate(trainLoader):
                x.unsqueeze_(1)
                y.unsqueeze_(1); y.unsqueeze_(1)
                prediction=tempRegressor(x)
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
                    y.unsqueeze_(1); y.unsqueeze_(1)
                    prediction=tempRegressor(x)
                    # print(prediction.shape)
                    for i, p in enumerate(prediction[:,0,0]):
                        pred.append(p)
                        true.append(y[i][0])
                    # print(pred)
                    # print(true)
                    loss=lossFunc(prediction,y)
                    valLosses.append(loss.item())
            if epoch%10==0:
                print("Round [%d]: {%.5f,%.5f}|%.5f" % (epoch+1,np.mean(losses),np.mean(valLosses),bestR2))
            true=np.array(true); pred=np.array(pred)
            # print(true.shape,pred.shape)
            tempR2Score=r2_score(true,pred)
            # print("r^2 score:",tempR2Score)
            if tempR2Score>bestR2:
                consecutiveRounds=0
                bestR2=tempR2Score
                torch.save(tempRegressor.state_dict(),'tmp/latentModel.pt')
            else:
                consecutiveRounds+=1
                if consecutiveRounds>=earlyStopEpoch:
                    print("No better performance after %d rounds, break." % (earlyStopEpoch))
                    break
        print("Best r2 score:",bestR2)
        self.latentRegressor=DNNRegressor()
        self.latentRegressor.load_state_dict(torch.load('tmp/latentModel.pt'))

    def loadLatentModel(self, path):
        """load pretrained model on latent space from given [path]"""
        self.latentRegressor=DNNRegressor()
        self.latentRegressor.load_state_dict(torch.load(path))

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
        start=time.time()
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
            print("r^2 score:",tempR2Score,'after ',time.time()-start,' seconds.')
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
        print(nItems,recodeDict)
        self.decodeDict=dict([(recodeDict[key],key) for key in recodeDict.keys()])
        self.nWords=42 # len(recodeDict)
        self.embedding=nn.Embedding(self.nWords+1,self.maxLength,padding_idx=0)
        self.standardData=padData
        smilesTrain,smilesTest,propTrain,propTest=train_test_split(padData,self.origProperties,test_size=0.2,random_state=2019)
        self.origSmilesTrain=smilesTrain # this stores smiles sequences for molecular design rip out usage
        nTrain=len(smilesTrain); nTest=len(smilesTest)
        print("Train test #:",nTrain,nTest)
        smilesTrain=torch.tensor(smilesTrain).to(torch.long)
        smilesTest=torch.tensor(smilesTest).to(torch.long)
        matrixEncode=False
        if matrixEncode is True:
            self.smilesTrain=torch.zeros(nTrain,self.maxLength,self.nWords,dtype=torch.float32)
            self.smilesTest=torch.zeros(nTest,self.maxLength,self.nWords,dtype=torch.float32)
            # tempMatrix=np.array(torch.randn(self.maxLength,self.maxLength))
            # Q,R=np.linalg.qr(tempMatrix)
            # Q=torch.from_numpy(Q)
            with torch.no_grad():
                for i in range(nTrain):
                    for j, s in enumerate(smilesTrain[i]):
                    # self.smilesTrain[i]=self.embedding(torch.Tensor(smilesTrain[i]).to(torch.long))
                    # print(self.smilesTrain[i])
                    # print(smilesTrain[i])
                        self.smilesTrain[i][j][int(s)]=1.0
                for i in range(nTest):
                    for j, s in enumerate(smilesTest[i]):
                        # self.smilesTest[i]=self.embedding(torch.Tensor(smilesTest[i]).to(torch.long))
                        # self.smilesTest[i][j]=Q[int(s)]
                        self.smilesTest[i][j][int(s)]=1.0
        else:
            self.smilesTrain=smilesTrain #torch.zeros(nTrain,self.maxLength-1,dtype=torch.long)
            self.smilesTest=smilesTest #torch.zeros(nTest,self.maxLength-1,dtype=torch.long)
        self.propTrain=torch.tensor(propTrain,dtype=torch.float32)
        self.propTest=torch.tensor(propTest,dtype=torch.float32)
        print("Dataset prepared.")

if __name__=='__main__':
    smiles,properties=loadEsolSmilesData()
    predictor=SmilesCNNPredictor(smiles,properties)
    # predictor.train(nRounds=1000,lr=5e-4,batchSize=20)
    # predictor.trainVAE(nRounds=500,lr=3e-4,earlyStop=True,earlyStopEpoch=20,batchSize=20)
    predictor.loadVAE('tmp/tmpTanh1Model.pt')
    predictor.encodeDataset()
    # predictor.identityRatio()
    # predictor.trainLatentModel(lr=3e-4,batchSize=20,nRounds=10000,earlyStopEpoch=100)
    predictor.loadLatentModel('tmp/latentModel.pt')
    # predictor.molecularDesign(lr=1e-3,rounds=100)
    predictor.molecularRandomDesign(aimNumber=100,batchSize=500)