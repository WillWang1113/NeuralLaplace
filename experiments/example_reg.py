import torch
from torch import nn
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
import matplotlib.pyplot as plt

class CustomDataset(Dataset):
    def __init__(self, samples, labels):
        self.samples = samples
        self.labels = labels

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx], self.labels[idx]


def get_data(dataset, batch_size=64):
    df = pd.read_csv("SOLETE_data/SOLETE_clean_5min.csv",
                     index_col=0,
                     parse_dates=True,
                     infer_datetime_format=True)
    df = df.sort_index()
    if dataset == "wind":
        features = ["u", "v", "P_Gaia[kW]"]
        # features = [
        #     "TEMPERATURE[degC]", "HUMIDITY[%]", "GHI[kW1m2]", "POA Irr[kW1m2]",
        #     "P_Gaia[kW]", "Pressure[mbar]", "u", "v"
        # ]
        df = df['2018-08':'2019-05']
    else:
        features = ["TEMPERATURE[degC]", "POA Irr[kW1m2]", "P_Solar[kW]"]

    df = df[features].values
    train_set, test_set = train_test_split(df, test_size=0.2, shuffle=False)
    train_set, val_set  = train_test_split(train_set, test_size=0.2, shuffle=False)
    scaler = StandardScaler()
    train_set = scaler.fit_transform(train_set)
    val_set = scaler.transform(val_set)
    test_set = scaler.transform(test_set)
    # train_ds = CustomDataset(train_set[...,:-1], train_set[...,[-1]])
    # val_ds = CustomDataset(val_set[...,:-1], val_set[...,[-1]])
    # test_ds = CustomDataset(test_set[...,:-1], test_set[...,[-1]])
    # dltrain = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    # dlval = DataLoader(val_ds, batch_size=batch_size, shuffle=True)
    # dltest = DataLoader(test_ds, batch_size=batch_size, shuffle=True)
    return train_set, val_set, test_set, scaler


train_set, val_set, test_set, scaler = get_data("wind")
model = [LinearRegression, MLPRegressor, GradientBoostingRegressor]
names = ["LR", "MLP", "GBRT"]
for i, m in enumerate(model):
    reg = m()
    reg.fit(train_set[:,:-1], train_set[:,[-1]])
    preds = reg.predict(test_set[:,:-1])
    preds = preds * scaler.scale_[-1] + scaler.mean_[-1]
    real = test_set[:,[-1]] * scaler.scale_[-1] + scaler.mean_[-1]
    test_loss = mean_squared_error(real, preds, squared=False)
    print(test_loss)
    fig, ax = plt.subplots()
    ax.plot(real.ravel(), label="real")
    ax.plot(preds.ravel(), label="preds")
    ax.legend()
    fig.savefig(f"{names[i]}.png")

