import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch.nn.functional as F
from source.layers import resNet18
import numpy as np
import pandas as pd
from datetime import datetime

# cuda = torch.device('cuda')
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def train(dataloader, model, loss_fn, optimizer):
    losses = []
    size = len(dataloader.dataset)
    for batch, (X, y) in enumerate(dataloader):
        X = X.to(device)
        y = y.to(device)
        pred, _, _, _, _ = model(X)
        loss = loss_fn(pred, y)
        
        # training
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        
        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f} [{current:->5d}/{size:>5d}]")
    return np.array(losses).mean()

def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0
    
    with torch.no_grad():
        for X, y in dataloader:
            X = X.to(device)
            y = y.to(device)
            pred, _, _, _, _ = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f}\n")
    return correct, test_loss
                  

if __name__ == '__main__':
    learning_rate = 1e-4
    batch_size = 128
    epochs = 200
    log = []
    date_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    columns = ['datetime', 'epoch', 'loss', 'val_loss', 'val_acc']
    
    training_data = datasets.CIFAR10(
        root='data',
        train=True,
        download=True,
        transform = transforms.Compose([
            transforms.RandomResizedCrop(36),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        # target_transform=transforms.Compose([
        #                          lambda x:torch.LongTensor([x]), # or just torch.tensor
        #                          lambda x:F.one_hot(x,10)])
        # target_transform = transforms.Compose([
        #     transforms.RandomResizedCrop(36),
        #     transforms.RandomHorizontalFlip(),
        #     transforms.ToTensor(),
        #     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        # ]),
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
    
    # random selection
    index = np.arange(training_data.data.shape[0]).astype('int32')
    np.random.shuffle(index)
    print(index)
    training_data.data = training_data.data[index[:10000]]
    training_data.targets = np.array(training_data.targets)[index[:10000]].tolist()
    
    train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_data, batch_size=batch_size)
    
    model = resNet18(3, 10, first_kernel=3, first_stride=1, stem_pooling=False)
    model = model.cuda()
    print(model, device)
    
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    for i in range(epochs):
        print(f"{i+1}/{epochs}")
        loss = train(train_dataloader, model, loss_fn, optimizer)
        val_loss, val_acc = test(test_dataloader, model, loss_fn)
        log.append([datetime.now().strftime("%Y%m%d_%H:%M:%S.%f"), i, loss, val_loss, val_acc])
                       
    # save logs
    pd.DataFrame(log, columns=columns).to_csv(f'result_random_selection_{date_str}.csv', index=False)
        
        
    
