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
    dnn_type = 1
    loss_list = []

    #dnn_type 0:regression    1:classification
    def __init__(self,dnn_type,property_num):
        if dnn_type != 0:
            self.dnn_type = 2
        self.model = DNN(property_num, property_num, property_num,property_num,self.dnn_type)

    def train(self,train_set,train_label,batch_size,learning_rate,num_epoches,early_stop,max_tolerance):
        #self.model.train()
        train_len = int(train_set.shape[0]/batch_size)*batch_size
        train_set = train_set[0:train_len]
        #create training data loader
        train_data = QSARDataset(train_set,train_label)
        train_loader = DataLoader(train_data,batch_size=batch_size, shuffle=True)
        if self.dnn_type == 1:
            #use MSE loss function for regression
            self.loss_func = nn.MSELoss()
        else:
            #use CrossEntropy loss function for classification
            self.loss_func = nn.CrossEntropyLoss() 
        #Create Gradient Descent Optimizer
        self.optimizer = torch.optim.SGD(self.model.parameters(), learning_rate)
        num_nongrowth = 0

        for epoch in range(num_epoches):
            avg_loss = 0
            for i, data in enumerate(train_loader, start=0):
                train, label = data
                train = train.float()
                if self.dnn_type == 1:
                    label = label.float()
                pred = self.model(train)
                if self.dnn_type == 1:
                    loss = self.loss_func(pred,label)
                else:
                    loss = self.loss_func(pred,label.squeeze())    
                avg_loss = avg_loss + (loss.data.numpy()/len(train_loader))
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            
            self.loss_list.append(avg_loss)
            if epoch == 0:
                old_loss = avg_loss
            else:
                if avg_loss < old_loss:
                    num_nongrowth = 0
                    old_loss = avg_loss
                else:
                    num_nongrowth = num_nongrowth +1
                    if num_nongrowth > max_tolerance and early_stop == 1:
                        return epoch +1,self.loss_list
            if epoch%10 == 0:
                print('epoch [{}/{}]'.format(epoch + 10, num_epoches))
                print('*' * 10)
                print('loss : {}'.format(avg_loss))
        return num_epoches,self.loss_list

    
    #test_set must be 2d --- data_num*property_num
    def test(self,test_set,test_label):
        #self.model.eval()
        single_input = 0
        #batch normalization can not handle single input
        if test_set.shape[0] == 1:
            test_set=np.array([test_set[0],test_set[0]])
            single_input = 1
        test_set = torch.from_numpy(test_set).float()
        test_label = torch.from_numpy(test_label).float() 
        test_output = self.model(test_set)
        if self.dnn_type == 1:
            test_pred = test_output.data.numpy()
        else:
            test_pred = torch.max(test_output, 1)[1].data.numpy().reshape(test_output.shape[0],1)
        if single_input == 1:
            return np.array([test_pred[0]])  
        else:
            return test_pred

    def save(self,path):
        torch.save(self.model.state_dict(), path)
    
    def load(self,path):
        self.model.load_state_dict(torch.load(path))


if __name__ == '__main__':
    csv_data = pd.read_csv("../datasets/delaney-processed.csv")#location of the data file
    columns = csv_data.columns
    device = torch.device('cpu')
    a = []
    label = []
    for i in columns:
        #i != "ESOL predicted log solubility in mols per litre" and
        if  i != "Compound ID" and i != "smiles":
            if i == "measured log solubility in mols per litre":
                label = csv_data[i].to_list()
            else:
                a.append(csv_data[i].to_list())

    data_set = np.array(a).T
    property_num = data_set.shape[1]

    train_len = 1000
    test_len = data_set.shape[0] - train_len
    #divide training set and test set
    train_set = data_set[0:train_len:1]
    test_set = data_set[train_len:data_set.shape[0]:1]
    data_label = np.array(label).reshape(len(label),1)
    train_label = data_label[0:train_len:1]
    test_label = data_label[train_len:data_set.shape[0]:1]

    batch_size = 50
    learning_rate = 0.01
    num_epoches = 1000

    model = QSARDNN(0,property_num)
    num_epoches , loss_list = model.train(train_set,train_label,batch_size,learning_rate,num_epoches,0,50)
    pred = model.test(test_set,test_label)
    print('R2 score: {}'.format(r2_score(pred,test_label)))

    #draw R2 score and L2 loss curve
    x1 = range(0,num_epoches)
    y1 = loss_list
    plt.plot(x1, y1, '.-')
    plt.title('loss vs. epoches')
    plt.ylabel('loss')
    plt.show()
