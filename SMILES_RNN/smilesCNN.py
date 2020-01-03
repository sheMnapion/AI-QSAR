import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from torch.utils.data import DataLoader, Dataset, TensorDataset
import time
import matplotlib.pyplot as plt
import copy
from rdkit import Chem
from rdkit.Chem import Draw
from dataProcess import loadEsolSmilesData

class SmilesVAE(nn.Module):
    """
        class for performing regression/classification and generating new medicines directly
        from smiles representation; using embedding structure as the core to encode SMILES sequence
        and decode accordingly
    """
    def __init__(self,maxLength,keyNum):
        """CNN structures"""
        super(SmilesVAE,self).__init__()
        self.maxLength=maxLength
        self.keyNum=keyNum
        self.embedding=nn.Embedding(keyNum,keyNum)
        self.fc1=nn.Linear(keyNum*maxLength,3000)
        # self.fc1_res=nn.Linear(3000,3000)
        self.fc21=nn.Linear(3000,200)
        self.fc22=nn.Linear(3000,200)
        self.fc3=nn.Linear(200,3000)
        # self.fc3_res=nn.Linear(3000,3000)
        self.fc4=nn.Linear(3000,keyNum*maxLength)
        self.decodeFC=nn.Linear(keyNum,keyNum)

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
        x=torch.tanh(self.fc1(x))
        # x=self.fc1_res(z)+z # residual block
        return self.fc21(x), self.fc22(x)

    def decode(self, z):
        """decode the inner representation vibrated with random noise to the original size"""
        batchSize=z.shape[0]
        z=torch.tanh(self.fc3(z))
        # z=self.fc3_res(z)+z
        z=torch.tanh(self.fc4(z))
        z=z.view(-1,self.keyNum)
        z=self.decodeFC(z)
        z=z.view(batchSize,-1,self.keyNum)
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

def vaeLossFunc(reconstructedX, x, mu, logvar, keyNum):
    # batchSize=x.shape[0]
    reconstructedX=reconstructedX.view(-1,keyNum)
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
    return 1000*BCE + KLD

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

class SmilesDesigner(object):
    """wrapper class for receiving data and training"""
    def __init__(self):
        pass

    def initFromSmilesAndProps(self,smiles,properties):
        """get smiles and property pairs for regression"""
        self.origSmiles=smiles
        self.origProperties=properties
        smileStrLength=np.array([len(s) for s in smiles])
        maxLength=np.max(smileStrLength)
        print("Max length for RNN input:",maxLength)
        self.maxLength=int(np.ceil(maxLength/4.0)*4)
        print("Pad to %d" % (self.maxLength))
        self._processData()
        self.vaeNet=SmilesVAE(self.maxLength,self.nKeys)

    def initFromModel(self, path):
        """now only create a model from vae"""
        vaeTotal=torch.load(path)
        vae=vaeTotal[0]
        latentModel=vaeTotal[1]
        decodeDict=vaeTotal[2]
        self.decodeDict=decodeDict
        self.nKeys=vae['embedding.weight'].shape[0]
        self.maxLength=int(vae['fc1.weight'].shape[1]/self.nKeys)
        print("Initialization from model:",self.nKeys,self.maxLength)
        self.vaeNet=SmilesVAE(self.maxLength,self.nKeys)
        self.vaeNet.load_state_dict(vae)
        self.latentRegressor=DNNRegressor()
        self.latentRegressor.load_state_dict(latentModel)
        print("Initialization done.")

    def loadVAE(self,path):
        """load trained VAE from given path"""
        self.vaeNet.load_state_dict(torch.load(path))
        print("VAE LOADED")

    def trainVAE(self,nRounds=1000,lr=0.01,earlyStop=True,earlyStopEpoch=10,batchSize=12,signal=None):
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
                x.unsqueeze_(1)
                y.unsqueeze_(1) # ; y.unsqueeze_(1)
                reconstructedX,mu,logVar,origX=self.vaeNet(x)
                loss=vaeLossFunc(reconstructedX,x,mu,logVar,self.nKeys)
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
                    x.unsqueeze_(1)
                    y.unsqueeze_(1) #; y.unsqueeze_(1)
                    reconstructedX,mu,logVar,origX=self.vaeNet(x)
                    loss=vaeLossFunc(reconstructedX,x,mu,logVar,self.nKeys)
                    valLosses.append(loss.item())
            valLoss=np.mean(valLosses)
            if bestLoss>valLoss:
                consecutiveRounds=0
                bestLoss=valLoss
                torch.save(self.vaeNet.state_dict(),'/tmp/tmpBestModel.pt')
            else:
                consecutiveRounds+=1
                if consecutiveRounds>=earlyStopEpoch and earlyStop:
                    print("No better performance after %d rounds, break." % (earlyStopEpoch))
                    break
            print("Round [%d]: {%.5f,%.5f|%.5f} after %.3f seconds" % (epoch+1,np.mean(losses),np.mean(valLosses),bestLoss,time.time()-start))
            if signal is not None:
                msg=str.format("Round [%d]: (%.5f,%.5f|%.5f) after %.5f seconds" % (epoch+1,np.mean(losses),np.mean(valLosses),bestLoss,time.time()-start))
                # print(msg)
                signal.emit(msg)
        print("Best validation loss:",bestLoss)
        self.vaeNet.load_state_dict(torch.load('/tmp/tmpBestModel.pt'))

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

    def molecularRandomDesign(self,aimNumber=100,batchSize=10,signal=None):
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
                # print(translation)
                if designedNumber==aimNumber: break
                if translation in designedMolecules: continue
                try:
                    mol=Chem.MolFromSmiles(translation)
                    Draw.MolToImageFile(mol,str.format('/tmp/designed_%d_%.5f.png' % (designedNumber,properties[i].item())))
                    print('[%d]: %s | (%d)' % (designedNumber+1,translation,i))
                    designedMolecules.add(translation)
                    designedMoleculesPairs.append([properties[i].item(),translation])
                    designedNumber+=1
                    if signal is not None:
                        signal.emit(str.format("No. %d molecule designed!" % designedNumber))
                except:
                    continue
        designedMoleculesPairs=np.array(designedMoleculesPairs)
        propertyIndex=np.argsort(designedMoleculesPairs[:,0])
        designedMoleculesPairs=designedMoleculesPairs[propertyIndex]
        return designedMoleculesPairs

    def trainLatentModel(self,lr=1e-3,batchSize=12,nRounds=1000,earlyStopEpoch=10):
        """train the prediction model within latent space"""
        from sklearn.ensemble import RandomForestRegressor
        tempRegressor=RandomForestRegressor(n_estimators=200,verbose=True,n_jobs=2)
        tempRegressor.fit(self.trainRepr,self.propTrain)
        pred=tempRegressor.predict(self.testRepr)
        score=r2_score(self.propTest,pred)
        print('predicted score:',score)
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
                torch.save(tempRegressor.state_dict(),'/tmp/latentModel.pt')
            else:
                consecutiveRounds+=1
                if consecutiveRounds>=earlyStopEpoch:
                    print("No better performance after %d rounds, break." % (earlyStopEpoch))
                    break
        print("Best r2 score:",bestR2)
        self.latentRegressor=DNNRegressor()
        self.latentRegressor.load_state_dict(torch.load('/tmp/latentModel.pt'))

    def loadLatentModel(self, path):
        """load pretrained model on latent space from given [path]"""
        self.latentRegressor=DNNRegressor()
        self.latentRegressor.load_state_dict(torch.load(path))

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
            self.smilesTrain=torch.zeros(nTrain,self.maxLength,self.nWords,dtype=torch.float32)
            self.smilesTest=torch.zeros(nTest,self.maxLength,self.nWords,dtype=torch.float32)
            # tempMatrix=np.array(torch.randn(self.maxLength,self.maxLength))
            # Q,R=np.linalg.qr(tempMatrix)
            # Q=torch.from_numpy(Q)
            with torch.no_grad():
                for i in range(nTrain):
                    for j, s in enumerate(smilesTrain[i]):
                        self.smilesTrain[i][j][int(s)]=1.0
                for i in range(nTest):
                    for j, s in enumerate(smilesTest[i]):
                        self.smilesTest[i][j][int(s)]=1.0
        else:
            self.smilesTrain=smilesTrain #torch.zeros(nTrain,self.maxLength-1,dtype=torch.long)
            self.smilesTest=smilesTest #torch.zeros(nTest,self.maxLength-1,dtype=torch.long)
        self.propTrain=torch.tensor(propTrain,dtype=torch.float32)
        self.propTest=torch.tensor(propTest,dtype=torch.float32)
        print("Dataset prepared.")

if __name__=='__main__':
    smiles,properties=loadEsolSmilesData()
    predictor=SmilesDesigner()
    predictor.initFromSmilesAndProps(smiles,properties)
    # predictor.train(nRounds=1000,lr=5e-4,batchSize=20)
    predictor.trainVAE(nRounds=500,lr=3e-4,earlyStop=True,earlyStopEpoch=20,batchSize=20)
    # predictor.loadVAE('tmp/tmpTanh1Model.pt')
    predictor.encodeDataset()
    # predictor.identityRatio()
    predictor.trainLatentModel(lr=3e-4,batchSize=20,nRounds=10000,earlyStopEpoch=100)
    # predictor.loadLatentModel('tmp/latentModel.pt')
    # predictor.molecularDesign(lr=1e-3,rounds=100)
    predictor.molecularRandomDesign(aimNumber=100,batchSize=500)