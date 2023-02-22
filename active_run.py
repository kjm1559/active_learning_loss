import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch.nn.functional as F
from source.layers import active_learning_model
import numpy as np
import pandas as pd
from datetime import datetime

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def loss_pred_metric(l_hat_i, l_hat_j, l_i, l_j, xi=1):
    st = torch.where(l_i > l_j, torch.ones_like(l_i), -torch.ones_like(l_i))
    # st = torch.reshape(st, (-1, 1))
    return torch.maximum(torch.zeros_like(l_hat_i), -st*(l_hat_i - l_hat_j) + xi)


def loss_pred_fn(loss, loss_hat, lambda_=1):
    # make pair
    loss_hat_i, loss_hat_j = loss_hat.chunk(2, dim=0)
    loss_i, loss_j = loss.chunk(2, dim=0)
    loss_pred = loss_pred_metric(loss_hat_i, loss_hat_j, loss_i, loss_j)
    batch_size = loss.size(dim=0)
    return lambda_ * (2*loss_pred.sum()/batch_size)
    
    

def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    losses = []
    losses_cross = []
    losses_pred = []
    for batch, (X, y) in enumerate(dataloader):
        X = X.to(device)
        y = y.to(device)
        pred, pred_loss = model(X)
        loss_cr = loss_fn(pred, y)
        # print(loss_cr.shape)
        loss_pred = loss_pred_fn(loss_cr, torch.flatten(pred_loss, 0))
        loss = loss_cr.mean() + loss_pred
        
        # training
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        losses.append(loss.item())#.cpu().detach().numpy())
        losses_cross.append(loss_cr.mean().item())#cpu().detach().numpy())
        losses_pred.append(loss_pred.item())#.cpu().detach().numpy())
        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}, loss_cross: {torch.mean(loss_cr):>7f}, loss_pred: {loss_pred:>7f} [{current:->5d}/{size:>5d}]")
    # print(losses, losses_cross)
    return np.array(losses).mean(), np.array(losses_cross).mean(), np.array(losses_pred).mean()
            
def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0
    
    with torch.no_grad():
        for X, y in dataloader:
            X = X.to(device)
            y = y.to(device)
            pred, _ = model(X)
            test_loss += torch.mean(loss_fn(pred, y)).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f}\n")
    return correct, test_loss

def get_loss_pred(dataloader, model):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    loss_pred_data = []
    
    with torch.no_grad():
        for X, y in dataloader:
            X = X.to(device)
            y = y.to(device)
            _, loss_pred = model(X)
            loss_pred_data += loss_pred.tolist()
    loss_pred_data = np.array(loss_pred_data)
    return loss_pred_data.reshape(-1) 

if __name__ == '__main__':
    learning_rate = 1e-4#1e-5
    batch_size = 128
    epochs = 200
    steps = 10
    K = 1000
    
    log = []
    date_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    columns = ['datetime', 'step', 'epoch', 'loss', 'loss_cross', 'loss_pred', 'val_loss', 'val_acc']
    
    remain_training_data = datasets.CIFAR10(
        root='data',
        train=True,
        download=True,
        transform = transforms.ToTensor(),
        # target_transform=transforms.Compose([
        #                          lambda x:torch.LongTensor([x]), # or just torch.tensor
        #                          lambda x:F.one_hot(x,10)])
    )
    
    training_data = datasets.CIFAR10(
        root='data',
        train=True,
        download=True,
        transform = transforms.Compose([
            transforms.RandomResizedCrop(32),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        # target_transform=transforms.Compose([
        #                          lambda x:torch.LongTensor([x]), # or just torch.tensor
        #                          lambda x:F.one_hot(x,10)])
    )
    test_data = datasets.CIFAR10(
        root='data',
        train=False,
        download=True,
        transform=transforms.ToTensor(),
        # target_transform=transforms.Compose([
        #                          lambda x:torch.LongTensor([x]), # or just torch.tensor
        #                          lambda x:F.one_hot(x,10)])
    )
    
    test_dataloader = DataLoader(test_data, batch_size=64)
    
    model = active_learning_model(3, 10, first_kernel=3, first_stride=1, stem_pooling=False)
    model = model.cuda()
    print(model, device)
    
    loss_fn = nn.CrossEntropyLoss(reduction='none')
    # optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    for step in range(steps):
        if step == 0:
            random_index = np.arange(remain_training_data.data.shape[0])
            np.random.shuffle(random_index)
            # select K samples randomly
            training_data.data = remain_training_data.data[random_index[:K]]
            training_data.targets = np.array(remain_training_data.targets)[random_index[:K]].tolist()
            remain_training_data.data = remain_training_data.data[random_index[K:]]
            remain_training_data.targets = np.array(remain_training_data.targets)[random_index[K:]].tolist()
            train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True)
        else:
            # select K samples by loss predct order
            remain_dataloader = DataLoader(remain_training_data, batch_size=batch_size)
            loss_pred_data = get_loss_pred(remain_dataloader, model)
            loss_pred_data = np.array(loss_pred_data)
            selected_index = loss_pred_data.argsort()
            training_data.data = np.concatenate((training_data.data, remain_training_data.data[selected_index[-K:]]), axis=0)
            training_data.targets += np.array(remain_training_data.targets)[selected_index[-K:]].tolist()
            remain_training_data.data = remain_training_data.data[selected_index[:-K]]
            remain_training_data.targets = np.array(remain_training_data.targets)[selected_index[:-K]].tolist()
            train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True)
        # optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=5e-4)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)#, momentum=0.9, weight_decay=5e-4)
        model.freeze_loss_predict(True)
        for i in range(epochs):
            if i == 160:
                # optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate/10, momentum=0.9, weight_decay=5e-4)
                optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate/10)#, momentum=0.9, weight_decay=5e-4)
            if i == 120:
                model.freeze_loss_predict(False)
            print(f"{step + 1} step, {i+1}/{epochs}")
            loss, loss_cross, loss_pred = train(train_dataloader, model, loss_fn, optimizer)
            val_acc, val_loss = np.nan, np.nan
            if i % 10 == 0:
                val_acc, val_loss = test(test_dataloader, model, loss_fn)
            log.append([datetime.now().strftime("%Y%m%d_%H:%M:%S.%f"), step, i, loss, loss_cross, loss_pred, val_loss, val_acc])
                       
    # save logs
    pd.DataFrame(log, columns=columns).to_csv(f'active_result_{date_str}.csv', index=False)
        
    
