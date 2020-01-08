import pandas as pd
import numpy as np
import torch
import time
from torch import nn, optim
from torch.autograd import Variable
from torch.utils.data import DataLoader,Dataset
from torchvision import datasets, transforms
from sklearn.metrics import r2_score 
import matplotlib.pyplot as plt
from PyQt5.QtCore import pyqtSignal
from sklearn.model_selection import train_test_split
from sklearn import preprocessing


#deep neural network: in_dim,n_hidden_1,n_hidden_2,n_hidden_3,out_dim
class DNN(nn.Module):
    def __init__(self,in_dim,n_hidden_1,n_hidden_2,n_hidden_3,out_dim):
        super(DNN, self).__init__()
        self.layer1 = nn.Sequential(nn.Linear(in_dim,n_hidden_1),nn.BatchNorm1d(n_hidden_1),nn.ReLU(True))
        self.layer2 = nn.Sequential(nn.Linear(n_hidden_1,n_hidden_2),nn.BatchNorm1d(n_hidden_2),nn.ReLU(True))
        self.layer3 = nn.Sequential(nn.Linear(n_hidden_2,n_hidden_3),nn.BatchNorm1d(n_hidden_3),nn.ReLU(True))
        self.layer4 = nn.Sequential(nn.Linear(n_hidden_3,out_dim))

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x

class QSARDataset(Dataset):#custom dataset
    def __init__(self,properties,label):
       self.properties = properties
       self.label = label
    def __getitem__(self, index):
        return self.properties[index],self.label[index]
    def __len__(self):
        return self.properties.shape[0]

class QSARDNN():
    #dnn_type 0:regression    1:classification
    def __init__(self,dnn_type=0,property_num=1):
        self.dnn_type = dnn_type
        assert(self.dnn_type == 0)# or self.dnn_type == 1)
        self.model = DNN(property_num, property_num, property_num,property_num,1)

    def setPropertyNum(self, property_num):
        self.model = DNN(property_num, property_num, property_num,property_num,1)

    def train(self,train_set,train_label,batch_size,learning_rate,num_epoches,early_stop,max_tolerance,
                progress_callback):

        progress_callback.emit('train_set.shape: {}'.format(train_set.shape))
        progress_callback.emit('train_label.shape: {}'.format(train_label.shape))
        progress_callback.emit('batch_size: {}'.format(batch_size))
        progress_callback.emit('learning_rate: {}'.format(learning_rate))
        progress_callback.emit('num_epoches: {}'.format(num_epoches))
        progress_callback.emit('early_stop: {}'.format(early_stop))
        progress_callback.emit('max_tolerance: {}'.format(max_tolerance))

        progress_callback.emit('\n***********Start Training***********\n')

        train_set,va_set,train_label,va_label = train_test_split(train_set,train_label,test_size = 0.2,random_state = 0)
        train_set = preprocessing.scale(train_set,axis=0)

        self.loss_list = []
        self.mse_list = []

        #create training data loader
        train_data = QSARDataset(train_set,train_label)
        train_loader = DataLoader(train_data,batch_size=batch_size, shuffle=True)
        if self.dnn_type == 0:
            #use MSE loss function for regression
            self.loss_func = nn.MSELoss()
        else:
            #use CrossEntropy loss function for classification
            self.loss_func = nn.CrossEntropyLoss() 
        #Create Gradient Descent Optimizer
        self.optimizer = torch.optim.Adam(self.model.parameters(), learning_rate)
        num_nongrowth = 0

        for epoch in range(num_epoches):
            avg_loss = 0; avg_mse = 0; cnt = 0
            for i, data in enumerate(train_loader, start=0):
                self.model.train()
                train, label = data
                if train.shape[0] < batch_size:
                    continue
                else:
                    cnt += 1
                train = train.float()
                if self.dnn_type == 0:
                    label = label.float()

                pred = self.model(train)

                if self.dnn_type == 0:
                    loss = self.loss_func(pred,label)
                else:
                    loss = self.loss_func(pred,label.squeeze())    

                avg_mse += loss.item()
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            va_pred = self.test(va_set,va_label)
            avg_loss = r2_score(va_pred,va_label)
            avg_mse /= float(cnt)
            self.mse_list.append(avg_mse)

            self.loss_list.append(avg_loss)
            if epoch == 0:
                old_loss = avg_loss
            else:
                if avg_loss > old_loss:
                    num_nongrowth = 0
                    old_loss = avg_loss
                else:
                    num_nongrowth = num_nongrowth +1
                    if num_nongrowth > max_tolerance and early_stop == 1:
                        progress_callback.emit('\n***********Finish Training***********\n')

                        return epoch + 1, self.loss_list, self.mse_list
            progress_callback.emit(str.format("%d/%d" % (epoch+1,num_epoches)))
            if epoch % 10 == 9:
                progress_callback.emit('epoch [{}/{}]'.format(epoch + 1, num_epoches))
                progress_callback.emit('*' * 10)
                progress_callback.emit('validation R2 score : {}'.format(old_loss))

        progress_callback.emit('\n***********Finish Training***********\n')

        return num_epoches, self.loss_list, self.mse_list

    
    #test_set must be 2d --- data_num*property_num
    def test(self,test_set,test_label):
        self.model.eval()
        single_input = 0
        test_set = preprocessing.scale(test_set,axis=0)
        #batch normalization can not handle single input
        if test_set.shape[0] == 1:
            test_set=np.array([test_set[0],test_set[0]])
            single_input = 1
        test_set = torch.from_numpy(test_set).float()
        test_label = torch.from_numpy(test_label).float() 
        test_output = self.model(test_set)
        if self.dnn_type == 0:
            test_pred = test_output.data.numpy()
        else:
            test_pred = torch.max(test_output, 1)[1].data.numpy().reshape(test_output.shape[0],1)
        if single_input == 1:
            return np.array([test_pred[0]])  
        else:
            return test_pred

    def train_and_test(self,train_set,train_label,test_set,test_label,batch_size,
            learning_rate,num_epoches,early_stop,max_tolerance,progress_callback,result_callback):

        num_epoches, loss_list, mse_list = self.train(train_set,train_label,batch_size,learning_rate,num_epoches,
                                        early_stop,max_tolerance, progress_callback)

        test_pred = self.test(test_set, test_label)
        test_pred = list(test_pred.reshape(-1))

        result = {"numEpochs": num_epoches, "lossList": loss_list,
                  "testPred": test_pred, "mseList": mse_list,
                  "model": self.model.state_dict(), "testLabel": test_label, 'modelName': 'DNN'}

        result_callback.emit(result)

        return result

    def save(self,path):
        torch.save(self.model.state_dict(), path)
    
    def load(self,path):
        self.model.load_state_dict(torch.load(path))


if __name__ == '__main__':
    #csv_data = pd.read_csv("../datasets/delaney-dropped_transformed.csv")#location of the data file
    csv_data = pd.read_csv("../datasets/desc_canvas_aug30_train.csv")
    columns = csv_data.columns
    device = torch.device('cpu')
    a = []
    label = []
    for i in columns:
        #if  i != "ESOL predicted log solubility in mols per litre" and i != "Compound ID" and i != "smiles":
        #    if i == "measured log solubility in mols per litre":
        if  i != "mol" and i != "CID" and i != "Class" and i != "Model":
            if i == "pIC50":
                label = csv_data[i].to_list()
            else:
                a.append(csv_data[i].to_list())

    data_set = np.array(a).T
    property_num = data_set.shape[1]

    #divide training set and test set
    
    data_label = np.array(label).reshape(len(label),1)
    train_set,test_set,train_label,test_label = train_test_split(data_set,data_label,test_size = 0.2,random_state = 0)

    batch_size = 50
    learning_rate = 0.01
    num_epoches = 1000

    model = QSARDNN(0,property_num)
    num_epoches , loss_list = model.train(train_set,train_label,batch_size,learning_rate,num_epoches,1,20)
    pred = model.test(test_set,test_label)
    print('R2 score: {}'.format(r2_score(pred,test_label)))

    #draw R2 score and L2 loss curve
    x1 = pred
    y1 = test_label
    plt.scatter(x1, y1)
    plt.title('pred')
    plt.ylabel('real label')
    plt.show()
