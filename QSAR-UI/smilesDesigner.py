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
import sys
from rdkit import Chem
from rdkit.Chem import Draw

EMBED_DIMENSION  = 35
LATENT_DIMENSION = 200

useGPU=torch.cuda.is_available()
if useGPU==True:
    torch.cuda.set_device(1)

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

class SmilesRNNVAE(nn.Module):
    """
        class for performing regression and generating new medicines directly
        from smiles representation; using GRU-seq2seq structure as the core part
        to encode SMILES sequence and decode accordingly
    """
    def __init__(self,keyNum):
        """seq2seq structures"""
        super(SmilesRNNVAE,self).__init__()
        self.keyNum=keyNum
        self.embedding=nn.Embedding(keyNum,EMBED_DIMENSION,padding_idx=0)
        self.encodeGRU=nn.GRU(
            input_size=EMBED_DIMENSION,
            hidden_size=LATENT_DIMENSION,
            num_layers=2,
            batch_first=True,
            bidirectional=False
        )
        self.decodeGRU=nn.GRU(
            input_size=LATENT_DIMENSION+EMBED_DIMENSION,
            hidden_size=EMBED_DIMENSION,
            num_layers=2,
            batch_first=True,
            bidirectional=False
        )
        self.fc11=nn.Linear(LATENT_DIMENSION,LATENT_DIMENSION)
        self.fc12=nn.Linear(LATENT_DIMENSION,LATENT_DIMENSION)
        self.predFC1=nn.Linear(LATENT_DIMENSION,LATENT_DIMENSION)
        self.predFC2=nn.Linear(LATENT_DIMENSION,1)
        self.fc2=nn.Linear(EMBED_DIMENSION,keyNum)

    def encode(self, x):
        """encode the input into two parts, mean mu and log variance"""
        x=self.embedding(x)
        # print(x.shape)
        out, _=self.encodeGRU(x)
        out=out[:,-1,:]
        # x=self.fc1_res(z)+z # residual block
        return self.fc11(out), self.fc12(out)

    def decode(self, z, maxLength):
        """decode the inner representation vibrated with normalized noise to the original size"""
        z=z.unsqueeze(1)
        # print(z.shape)
        batchSize=z.shape[0]
        retTensor=torch.randn(batchSize,0,EMBED_DIMENSION,requires_grad=True)
        if useGPU==True:
            retTensor=retTensor.cuda()
        lastWord=torch.zeros(batchSize,1,EMBED_DIMENSION,requires_grad=True) # first word always 0
        if useGPU==True:
            lastWord=lastWord.cuda()
        for i in range(maxLength):
            presentState=torch.cat([z,lastWord],dim=2)
            presentOut, _=self.decodeGRU(presentState)
            retTensor=torch.cat([retTensor,presentOut],dim=1)
            lastWord=presentOut
        pred=self.fc2(retTensor)
        return pred #emporarily take this as the binary vector format

    def reparameterize(self, mu, logvar):
        """re-parameterization trick in training the net"""
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def forward(self, x):
        maxLength=x.shape[1]
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        pred=F.relu(self.predFC1(mu))
        predY=self.predFC2(pred)
        return self.decode(z, maxLength), mu, logvar, predY # standard version also included

    def middleRepresentation(self, x):
        """return representation in latent space"""
        mu, logVar = self.encode(x)
        return mu #self.reparameterize(mu,logVar)

def vaeLossFunc(reconstructedX, x, mu, logvar, keyNum, predY, y, debug=True, weight=1):
    # batchSize=x.shape[0]
    reconstructedX=reconstructedX.contiguous().view(-1,keyNum)
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
    REGRE=nn.MSELoss()
    REGRESS=REGRE(y,predY)
    if debug==True:
        print(BCE.item(),KLD.item(),REGRESS.item())
    return 100*BCE + weight*KLD + 10*REGRESS

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
        # self.vaeNet=SmilesVAE(self.maxLength,self.nKeys)
        # NOW START USING RNN REPRESENTATION
        self.vaeNet=SmilesRNNVAE(self.nKeys)
        if useGPU==True:
            self.vaeNet=self.vaeNet.cuda()

    def initFromModel(self, path):
        """now only create a model from vae
            model content: [vaeState dict, decodeDict]
        """
        vaeTotal=torch.load(path)
        vae=vaeTotal[0]
        decodeDict=vaeTotal[1]
        self.decodeDict=decodeDict
        self.nKeys=vae['embedding.weight'].shape[0]
        print("Initialization from model:",path,self.nKeys)
        self.vaeNet=SmilesRNNVAE(self.nKeys)
        self.vaeNet.load_state_dict(vae)
        print("Initialization done.")

    def loadVAE(self,path):
        """load trained VAE from given path"""
        if useGPU==False:
            self.vaeNet.load_state_dict(torch.load(path,map_location='cpu'))
        else:
            self.vaeNet.load_state_dict(torch.load(path,map_location='cuda'))
        print("VAE LOADED")

    def trainVAE(self,nRounds=1000,lr=0.01,earlyStop=True,earlyStopEpoch=10,batchSize=12,signal=None):
        """train the vae net"""
        trainSet=TensorDataset(self.smilesTrain,self.propTrain)
        testSet=TensorDataset(self.smilesTest,self.propTest)
        trainLoader=DataLoader(trainSet,batch_size=batchSize,shuffle=True,num_workers=2,collate_fn=CollateFn())
        testLoader=DataLoader(testSet,batch_size=batchSize,shuffle=False,num_workers=2,collate_fn=CollateFn())
        optimizer=optim.Adam(self.vaeNet.parameters(),lr=lr,weight_decay=1e-8)
        consecutiveRounds=0
        bestLoss=1e9
        lossFunc1=nn.MSELoss()
        start=time.time()
        for epoch in range(nRounds):
            losses=[]
            if epoch<100:
                epochWeight=0
            elif epoch>=100 and epoch<200:
                epochWeight=(epoch-100)/10000
            else:
                epochWeight=0.01
            for i, (x,y) in enumerate(trainLoader):
                # w,h=x.shape[:2]
                # x.unsqueeze_(1)
                y.unsqueeze_(1) # ; y.unsqueeze_(1)
                if useGPU==True:
                    x=x.cuda(); y=y.cuda()
                reconstructedX,mu,logVar,predY=self.vaeNet(x)
                loss=vaeLossFunc(reconstructedX,x,mu,logVar,self.nKeys,predY,y,debug=False,weight=epochWeight)
                # loss=lossFunc1(predY,y)
                losses.append(loss.item())
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            losses=np.array(losses)
            print("Train mean loss:",np.mean(losses))
            valLosses=[]; pred=[]; true=[]
            with torch.no_grad():
                for i, (x,y) in enumerate(testLoader):
                    # w,h = x.shape[:2]
                    # x.unsqueeze_(1)
                    y.unsqueeze_(1) #; y.unsqueeze_(1)
                    if useGPU==True:
                        x=x.cuda(); y=y.cuda()
                    reconstructedX,mu,logVar,predY=self.vaeNet(x)
                    loss=vaeLossFunc(reconstructedX,x,mu,logVar,self.nKeys,predY,y,weight=epochWeight)
                    # loss=lossFunc1(predY,y)
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
        self.vaeNet.load_state_dict(torch.load('/tmp/tmpBestModel.pt'))

    def encodeDataset(self,batchSize=30):
        """encode dataset with trained VAE to obtain its representation in latent space"""
        trainSet=TensorDataset(self.smilesTrain,self.propTrain)
        testSet=TensorDataset(self.smilesTest,self.propTest)
        trainLoader=DataLoader(trainSet,batch_size=batchSize,shuffle=False,num_workers=2,collate_fn=CollateFn())
        testLoader=DataLoader(testSet,batch_size=batchSize,shuffle=False,num_workers=2,collate_fn=CollateFn())
        trainRet=torch.zeros(0,LATENT_DIMENSION,dtype=torch.float32)
        if useGPU==True:
            trainRet=trainRet.cuda()
        for (x,y) in trainLoader:
            if useGPU==True:
                x=x.cuda(); y=y.cuda()
            tempRet=self.vaeNet.middleRepresentation(x)
            trainRet=torch.cat([trainRet,tempRet],dim=0)
        print('Train repr shape:',trainRet.shape)
        self.trainRepr=trainRet.detach().clone()
        testRet=torch.zeros(0,LATENT_DIMENSION,dtype=torch.float32)
        if useGPU==True:
            testRet=testRet.cuda()
        for (x,y) in testLoader:
            if useGPU==True:
                x=x.cuda(); y=y.cuda()
            tempRet=self.vaeNet.middleRepresentation(x)
            testRet=torch.cat([testRet,tempRet],dim=0)
        print('Test repr shape:',testRet.shape)
        self.testRepr=testRet.detach().clone()

    def identityRatio(self):
        """check the extent of identity in the first 100 characters learned through our vae net"""
        temp=self.trainRepr
        decoded=self.vaeNet.decode(temp,100)
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
            for j in range(100):
                if orig[j]==0:
                    correctCount/=j
                    correctRatios.append(correctCount)
                    break
                if trans[j]==orig[j]:
                    correctCount+=1
        correctRatios=np.array(correctRatios)
        print('Average correct translation ratio:',np.mean(correctRatios))

    def molecularRandomDesign(self,aimNumber=100,batchSize=10,signal=None):
        """design [aimNumber] molecules by randomly sampling from latent space"""
        designedNumber=0
        designedMolecules=set()
        designedMoleculesPairs=[]
        while designedNumber<aimNumber:
            latentVector=torch.randn(batchSize,200)
            # print(latentVector.shape)
            properties=F.relu(self.vaeNet.predFC1(latentVector))
            properties=F.relu(self.vaeNet.predFC2(properties))
            print(properties.shape)
            translate=self.vaeNet.decode(latentVector,100)
            print(translate.shape)
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

    def testLatentModel(self,batchSize=12):
        """train the prediction model within latent space"""
        from sklearn.ensemble import RandomForestRegressor
        tempRegressor=RandomForestRegressor(n_estimators=100,verbose=True,n_jobs=2)
        tempRegressor.fit(self.trainRepr.cpu(),self.propTrain.cpu())
        pred=tempRegressor.predict(self.testRepr.cpu())
        score=r2_score(self.propTest.cpu(),pred)
        print('Random forest prediction score:',score)
        if useGPU==True:
            self.trainRepr=self.trainRepr.cpu()
            self.propTrain=self.propTrain.cpu()
            self.testRepr=self.testRepr.cpu()
            self.propTest=self.propTest.cpu()
        print(self.trainRepr.shape,self.propTrain.shape)
        trainSet=TensorDataset(self.trainRepr,self.propTrain)
        testSet=TensorDataset(self.testRepr,self.propTest)
        trainLoader=DataLoader(trainSet,batch_size=batchSize,shuffle=False,num_workers=1)
        testLoader=DataLoader(testSet,batch_size=batchSize,shuffle=False,num_workers=1)
        true=[]; pred=[]
        for (x,y) in trainLoader:
            if useGPU==True:
                x=x.cuda(); y=y.cuda()
            x=F.relu(self.vaeNet.predFC1(x))
            predY=self.vaeNet.predFC2(x)
            for i in range(len(y)):
                true.append(y[i].item())
                pred.append(predY[i][0].item())
        print("Train score:",r2_score(true,pred))
        true=[]; pred=[]
        for (x,y) in testLoader:
            if useGPU==True:
                x=x.cuda(); y=y.cuda()
            x=F.relu(self.vaeNet.predFC1(x))
            predY=self.vaeNet.predFC2(x)
            for i in range(len(y)):
                true.append(y[i].item())
                pred.append(predY[i][0].item())
        print("Test score:",r2_score(true,pred))

    def _processData(self):
        """
            padding data for smile strings to have same input length;
            encode the input string to have unitary coding format
            also split data
        """
        # split first
        smilesSplit = []
        for smiles in self.origSmiles:
            smiles=''.join(smiles)
            smilesLength = len(smiles)
            nameStr = []
            index = 0
            while index < smilesLength:
                tempAlpha = smiles[index]
                if index < smilesLength - 1 and tempAlpha.isalpha():
                    anotherAlpha = smiles[index + 1]
                    if anotherAlpha == ' ':  # error, need cleaning
                        index += 1
                        continue
                    if tempAlpha.isalpha():
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