import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, random_split, DataLoader

from torch.utils.tensorboard import SummaryWriter
import torchmetrics

import pandas as pd
import matplotlib.pyplot as plt

class Model(nn.Module):
    def __init__(self, input_dim):
        super(Model, self).__init__()
        self.layer1 = nn.Linear(in_features=input_dim, out_features=50)
        self.layer2 = nn.Linear(in_features=50, out_features=50)
        self.layer3 = nn.Linear(in_features=50, out_features=3)

    def forward(self, x): 
        x = F.relu(self.layer1(x)) 
        x = F.relu(self.layer2(x))
        x = F.softmax(self.layer3(x), dim=1) 
        return x

class StandardScaler:
    def __init__(self, mean=None, std=None, epsilon=1e-7):
        self.mean = mean
        self.std = std
        self.epsilon = epsilon

    def fit(self, values):
        dims = list(range(values.dim() -1))
        self.mean = torch.mean(values, dim=dims)
        self.std = torch.std(values, dim=dims)
    
    def transform(self, values):
        return (values - self.mean) / (self.std + self.epsilon)

    def fit_transform(self, values):
        self.fit(values)
        return self.transform(values)

    def __repr__(self):
        return f'mean: {self.mean}, std:{self.std}, epsilon:{self.epsilon}'

class IrisDataset(Dataset):
    def __init__(self, src_file, transform=None):
        iris_dataset = pd.read_csv(src_file, names=['sepallength','sepalwidth','petallength','petalwidth','class'])
        X = iris_dataset[iris_dataset.columns.intersection(['sepallength','sepalwidth','petallength','petalwidth'])]
        Y = iris_dataset[iris_dataset.columns.intersection(['class'])]

        y_conversion = pd.DataFrame()
        for name in Y['class'].unique():
            y_conversion[name] = (Y['class']==name).apply(lambda x: 1.0 if x else 0.0)
        y_tensor = torch.as_tensor(y_conversion.to_numpy()).type(torch.float32)

        x_tensor = torch.tensor(X.values)

        scaler = StandardScaler()
        x_scaled = scaler.fit_transform(x_tensor).type(torch.float32)

        self.data = torch.cat((x_scaled, y_tensor), 1)
        self.transform = transform

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()
        x = self.data[index, 0:4]
        y = self.data[index, 4:]
        sample = (x, y)
        if self.transform:
            sample = self.transform(sample)
        return sample

def train_one_epoch(model, train_ldr, optimizer, loss_fn):
    for data in train_ldr:
        inputs, labels = data
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()

def test_one_epoch(model, val_ldr, epoch, tb_writer):
    acc_metric = torchmetrics.classification.Accuracy(task="multiclass", num_classes=3)
    for data in val_ldr:
        inputs, labels = data
        outputs = model(inputs)
        acc_metric(outputs, labels)
    acc = acc_metric.compute()
    tb_writer.add_scalar("Accuracy/Train", acc, epoch)
        
if __name__ == '__main__':
    dataset = IrisDataset('pytorch_pandas/doc/iris.data')

    dataset_len = len(dataset)
    train_size = int(dataset_len * 0.8)
    val_size = int(dataset_len - train_size)
    train_set, val_set = random_split(dataset, [train_size, val_size])
    train_ldr = DataLoader(train_set, batch_size=4, shuffle=True, drop_last=False)
    val_ldr = DataLoader(val_set, batch_size=4, shuffle=True, drop_last=False)

    model = Model(4)
    model = torch.compile(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.CrossEntropyLoss()
    tb_writer = SummaryWriter()

    EPOCHS = 200
    for epoch in range(EPOCHS):
        model.train(True)
        train_one_epoch(model, train_ldr, optimizer, loss_fn)
        model.train(False)
        if epoch in (0, 25, 50, 75, 100, 125, 150, 175, 199):
            test_one_epoch(model, val_ldr, epoch, tb_writer)
        if epoch % 50 == 0 and epoch != 0:
            torch.save(
                {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }, 
                f"pytorch_pandas/doc/checkpoint-{epoch}.pt"
            )
    
    torch.save(model.state_dict(), "pytorch_pandas/doc/iris_weights.pt")

    model = Model(4)
    model = torch.compile(model)
    model.load_state_dict(torch.load("pytorch_pandas/doc/iris_weights.pt"))
    model.eval()

    inference_tensor = pd.read_csv('pytorch_pandas/doc/iris.inference', names=['sepallength','sepalwidth','petallength','petalwidth'])
    inference_tensor = torch.tensor(inference_tensor.values)
    
    scaler = StandardScaler(mean=torch.tensor([5.84, 3.05, 3.76, 1.20]), std=torch.tensor([0.83, 0.43, 1.76, 0.76]))
    inference_tensor = scaler.transform(inference_tensor).type(torch.float32)
    inference_tensor = model(inference_tensor)
    print(inference_tensor)
    