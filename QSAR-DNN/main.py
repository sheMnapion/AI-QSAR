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


batch_size = 50
learning_rate = 0.01
num_epoches = 1000

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
in_dim, n_hidden_1, n_hidden_2,n_hidden_3, out_dim = property_num, property_num, property_num,property_num,1 

train_len = int((data_set.shape[0] - int(data_set.shape[0]/10))/batch_size)*batch_size
test_len = data_set.shape[0] - train_len
#divide training set and test set
train_set = data_set[0:train_len:1]
test_set = data_set[train_len:data_set.shape[0]:1]
data_label = np.array(label).reshape(len(label),1)
train_label = data_label[0:train_len:1]
test_label = data_label[train_len:data_set.shape[0]:1]

test_set = torch.from_numpy(test_set)
test_label = torch.from_numpy(test_label)
#create training data loader
train_data = QSARDataset(train_set,train_label)
train_loader = DataLoader(train_data,batch_size=batch_size, shuffle=True)
#use MSE loss function
l2_loss = torch.nn.MSELoss()
#Create DNN network
dnn = DNN(in_dim, n_hidden_1,n_hidden_2,n_hidden_3, out_dim).to(device)
#Create Gradient Descent Optimizer
optimizer = torch.optim.SGD(dnn.parameters(), learning_rate)

r2 = 0
r2_list = []
l2_list = []

print(len(train_loader))
for epoch in range(num_epoches):
    for i, data in enumerate(train_loader, start=0):
        train, label = data
        train = train.float()
        label = label.float()
        pred = dnn(train)
        loss =l2_loss(pred,label)
        if i == len(train_loader) - 1:
            l2_list.append(loss.data.numpy())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    test_set = test_set.float()
    test_label = test_label.float()
    test_pred = dnn(test_set)
    r2 = r2_score(test_label.data.numpy(),test_pred.data.numpy())
    r2_list.append(r2)
    if epoch%10 == 0:
        print('epoch [{}/{}]'.format(epoch , num_epoches))
        print('*' * 10)
        print('R2 score: {}'.format(r2))

#draw R2 score and L2 loss curve
x1 = range(0,num_epoches) 
y1 = r2_list 
plt.subplot(2, 1, 1) 
plt.plot(x1, y1, '.-') 
plt.title('R2 score vs. epoches') 
plt.ylabel('R2 score')
x2 =  range(0,num_epoches)
y2 = l2_list
plt.subplot(2, 1, 2) 
plt.plot(x2, y2, '.-') 
plt.title('L2 loss vs. epoches') 
plt.ylabel('L2 loss')
plt.show() 
plt.savefig("loss.jpg")