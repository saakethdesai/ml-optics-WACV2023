import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle as pkl
import numpy as np
from torch.utils.data import Dataset, DataLoader
from matplotlib import pyplot as plt

torch.manual_seed(0)
np.random.seed(0)


class neural_net(nn.Module):
    def __init__(self):
        super(neural_net, self).__init__()
        self.ninputs = 3 
        self.noutputs = 1 
        
        self.linear1 = nn.Linear(self.ninputs, 1000)
        self.linear2 = nn.Linear(1000, 1000)
        self.linear3 = nn.Linear(1000, self.noutputs)
        
    def forward(self, x):
        x = self.linear1(x)
        x = torch.relu(x)
        x = self.linear2(x)
        x = torch.relu(x)
        x = self.linear3(x)
        x = torch.relu(x)
        return x 


def nn_loss(pred_x, gt_x):
    mse = F.mse_loss(pred_x, gt_x, reduction='none')
    mse = torch.sum(mse, axis=1)
    mse = torch.mean(mse)
    return mse 


#----------------------------------------------#
#load dataset
with open("2.5mm.p", "rb") as f:
    data = pkl.load(f) 

raw_inputs = np.array(data['inputs'], dtype='float32')
output = np.array(data['img_result'], dtype='float32')

#filter relevant variables
#idx = [1, 2, 8]
inputs = raw_inputs#[:, idx]
output = output.reshape((-1, 1)) 
dataset = np.concatenate((inputs, output), axis=1) 
print (inputs.shape, output.shape, dataset.shape)

#split into train and test sets
np.random.shuffle(dataset)
idx1 = int(0.8*len(dataset))
idx2 = int(0.9*len(dataset))
train_data = dataset[:idx1, :]
val_data = dataset[idx1:idx2, :]
test_data = dataset[idx2:, :]
print (train_data.shape, val_data.shape, test_data.shape)

#create dataloaders
class MyDataset(Dataset):
    def __init__(self, data):
        self.data = data
    def __len__(self):
        return len(self.data)
    def __getitem__(self, index):
        X = self.data[index, :3]
        y = self.data[index, 3:]
        return X, y

train_dataset = MyDataset(train_data)
val_dataset = MyDataset(val_data)
test_dataset = MyDataset(test_data)

batch_size = 32
train_data_gen = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_data_gen = DataLoader(val_dataset, batch_size=len(val_dataset), shuffle=True)
test_data_gen = DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=True)


#create model
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

net = neural_net().to(device=device)
#vae.load_state_dict(torch.load("vae_low_freq.pth"))
num_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
print('Number of parameters: %d' % num_params)

learning_rate = 1e-4
optimizer = torch.optim.Adam(params=net.parameters(), lr=learning_rate)

EPOCHS = 200 
print('Training ...')


for epoch in range(EPOCHS):

    train_loss = 0 
    num_batches = 0
    
    for data, labels in train_data_gen:
        data = data.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        pred = net(data)
        loss = nn_loss(pred, labels)
        loss.backward()
        curr_loss = loss.item()
        train_loss += curr_loss 
        optimizer.step()
        num_batches += 1
        #print('Batch [%d / %d] train loss: %f' % (num_batches, len(train_data_gen), curr_loss))
    
    for data, labels in val_data_gen:
        data = data.to(device)
        labels = labels.to(device)
        pred = net(data)
        loss = nn_loss(pred, labels)
        val_loss = loss.item()
    
    train_loss /= num_batches
    print('Epoch [%d / %d] train loss: %f, val loss: %f' % (epoch+1, EPOCHS, train_loss, val_loss))

    torch.save(net.state_dict(), "nn.pth")
    #np.savetxt("loss.txt", np.array(train_loss_avg))

net = neural_net().to(device=device)
net.load_state_dict(torch.load("nn.pth"))

#make predictions on train, val, and test dataset
train_data_gen = DataLoader(train_dataset, batch_size=len(train_dataset), shuffle=True)
val_data_gen = DataLoader(val_dataset, batch_size=len(val_dataset), shuffle=True)
test_data_gen = DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=True)

gt_train_labels = []
pred_train_labels = []
gt_val_labels = []
pred_val_labels = []
gt_test_labels = []
pred_test_labels = []

for data, labels in train_data_gen:
    data = data.to(device)
    labels = labels.to(device)
    pred = net(data)
    gt_train_labels.append(labels.detach().numpy().flatten())
    pred_train_labels.append(pred.detach().numpy().flatten())

for data, labels in val_data_gen:
    data = data.to(device)
    labels = labels.to(device)
    pred = net(data)
    gt_val_labels.append(labels.detach().numpy().flatten())
    pred_val_labels.append(pred.detach().numpy().flatten())

for data, labels in test_data_gen:
    data = data.to(device)
    labels = labels.to(device)
    pred = net(data)
    loss = nn_loss(pred, labels)
    test_loss = loss.item()
    gt_test_labels.append(labels.detach().numpy().flatten())
    pred_test_labels.append(pred.detach().numpy().flatten())

print ("Loss on test dataset: ", test_loss)
plt.plot(np.array(gt_train_labels), np.array(pred_train_labels), 'ro')
plt.plot(np.array(gt_val_labels), np.array(pred_val_labels), 'bo')
plt.plot(np.array(gt_test_labels), np.array(pred_test_labels), 'go')
plt.savefig("preds.png")
