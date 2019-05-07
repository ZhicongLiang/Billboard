import torch
import torch.nn as nn
import torch.utils.data as data
from torch.autograd import Variable
import pandas as pd
import tqdm
import matplotlib.pyplot as plt
import time
import numpy as np

class Dataset(data.Dataset):
    '''
    Setup the dataset
    '''
    train_set = './train_set/train.xlsx'
    test_set = './test_set/test.xlsx'
    
    def __init__(self, train=True):
        if train:
            self.data = pd.read_excel(self.train_set)
        else:
            self.data = pd.read_excel(self.test_set)
        
    def __getitem__(self, index):
        row = self.data.iloc[index]
        x = row[['Danceability', 
                 'Energy', 
                 'Speechiness', 
                 'Acousticness', 
                 'Instrumentalness', 
                 'Liveness',
                 'Valence',
                 'Loudness',
                 'Tempo',
                 'Artist_Score']]
        x = torch.Tensor(x)
        
        y = np.array(row['label'])
        y = torch.from_numpy(y)
                
        return x, y
    
    def __len__(self):
        return self.data.shape[0]
    

class Model(nn.Module):
    '''Define 1-hidden layer Neural Network model'''
    def __init__(self):
        super(Model, self).__init__()
    
        layers = [
                    nn.Linear(10, 6),
                    nn.ReLU(),
                    nn.Linear(6,1),
                    nn.Sigmoid()
                ]
        self.classifier = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.classifier(x)

    
def accuracy(model, loader):
    '''
    Calculate the accuracy of give model and dataloader
    '''
    total = 0
    correct = 0
    for j, (xx, yy) in enumerate(loader):
        xx, yy = Variable(xx, requires_grad=False), Variable(yy.to(torch.float)[:,None], requires_grad=False)
        pred_logits = model(xx)
        pred_yy = pred_logits.ge(threshold).float()    
        correct += sum(pred_yy == yy).item()
        total += yy.shape[0]
    return correct/total


    
if __name__ == '__main__':
    
    # hyperparameter setting
    batch_size = 128
    epochs = 100
    threshold = 0.5
    
    learning_rate = 5e-4
    decay = 1e-4
    step_size = 10
    momentum = 0.95
    
    # get the dataset
    train_set = Dataset(train=True)
    train_loader = data.DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True)
    test_set = Dataset(train=False)
    test_loader = data.DataLoader(dataset=test_set, batch_size=batch_size, shuffle=False)
    
    # model instance
    model = Model()
    
    criterion = nn.BCELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=decay, momentum=momentum)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=0.5)
    
    hist = pd.DataFrame(columns=['epoch', 'loss', 'train_accuracy', 'test_accuracy', 'time'])
    for epoch in range(epochs):
        tic = time.time()
        
        for i, (x,y) in enumerate(train_loader):
            
            x, y = Variable(x, requires_grad=False), Variable(y.to(torch.float)[:,None], requires_grad=False)
            
            optimizer.zero_grad()
            
            logits = model(x)
            loss = criterion(logits, y)
            
            loss.backward()
            optimizer.step()
        
        # evaluation
        with torch.no_grad():
            train_accuracy = accuracy(model, train_loader)
            test_accuracy = accuracy(model, test_loader)
        
        toc = time.time()
        
        print('epoch:{:d} loss:{:.4f}  train_acc:{:.4f} test_acc:{:4f} time:{:.4}'.format(epoch, loss.item(), train_accuracy, test_accuracy, toc-tic))
            
        
        hist.loc[hist.shape[0]+1] = [epoch, loss.item(), train_accuracy, test_accuracy, toc-tic]
        hist.to_excel('./history/nn.xlsx')
    
#     save the model weights
    state_dict = model.state_dict()
    torch.save(state_dict, './weights/nn.pth')
    
    # plot the loss and accuracy history
    f = plt.figure()
    plt.title('Train loss')
    plt.xlabel('epoch')
    plt.plot(np.arange(hist.shape[0]), hist['loss'])
    
    g = plt.figure()
    plt.plot(np.arange(hist.shape[0]), hist['train_accuracy'])
    plt.plot(np.arange(hist.shape[0]), hist['test_accuracy'])
    plt.title('Accuracy')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')