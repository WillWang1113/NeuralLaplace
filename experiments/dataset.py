###########################
# Neural Laplace: Learning diverse classes of differential equations in the Laplace domain
# Author: Samuel Holt
###########################
import shelve
from functools import partial
import numpy as np
import scipy.io as sio
import torch
from ddeint import ddeint
from torch.utils.data import DataLoader, SubsetRandomSampler
from tqdm import tqdm
import pandas as pd
from utils import setup_seed
from statsmodels.tsa.arima_process import ArmaProcess

# from torchlaplace.data_utils import (
#     basic_collate_fn
# )
from NeuralLaplace.torchlaplace.data_utils import basic_collate_fn

from pathlib import Path

local_path = Path(__file__).parent

# DE Datasets


def mfred(device, double=False, window_width=24 * 12 * 2, choose=0):
    df = pd.read_csv("datasets/MFRED.csv",
                     parse_dates=True,
                     infer_datetime_format=True,
                     index_col=0)
    df = df.resample("5min").mean()
    select_site = df.columns[choose]
    df = df[[select_site]].values

    trajs = []
    start = 0
    while start + window_width < len(df):
        end = start + window_width
        trajs.append(df[start:end])
        start += 12
    if len(trajs[-1]) != len(trajs[-2]):
        trajs.pop()
    trajs = np.stack(trajs, axis=0)
    # t = torch.linspace(20 / window_width, 20, window_width)
    t = torch.arange(window_width)
    sample_rate = window_width / (t.numpy().max() -
                                  t.numpy().min()) * 2 * np.pi
    if double:
        t = t.to(device).double()
        trajs = torch.from_numpy(trajs).to(device).double()
    else:
        t = t.to(device)
        trajs = torch.from_numpy(trajs).to(device)
    return trajs, t.unsqueeze(0), sample_rate


def nrel(device,
         double=False,
         window_width=24 * 12 * 2,
         choose=9,
         transformed=False):
    df_wind = pd.read_csv("datasets/wtk_wind.csv",
                          parse_dates=True,
                          infer_datetime_format=True,
                          index_col=0)
    df_power = pd.read_csv("datasets/NREL_power.csv",
                           parse_dates=True,
                           infer_datetime_format=True,
                           index_col=0)
    select_site = df_wind.columns[choose]
    df_wind_select = df_wind[select_site].values
    if transformed:
        df_power_select = df_power[select_site + "-LNT"].values

    else:
        df_power_select = df_power[select_site + "[pu]"].values

    df = np.concatenate(
        [df_wind_select.reshape(-1, 1),
         df_power_select.reshape(-1, 1)],
        axis=1)
    trajs = []
    start = 0
    while start + window_width < len(df):
        end = start + window_width
        trajs.append(df[start:end])
        start += 12
    if len(trajs[-1]) != len(trajs[-2]):
        trajs.pop()
    trajs = np.stack(trajs, axis=0)
    # t = torch.linspace(20 / window_width, 20, window_width)
    t = torch.arange(window_width)
    sample_rate = window_width / (t.numpy().max() -
                                  t.numpy().min()) * 2 * np.pi
    if double:
        t = t.to(device).double()
        trajs = torch.from_numpy(trajs).to(device).double()
    else:
        t = t.to(device)
        trajs = torch.from_numpy(trajs).to(device)
    return trajs, t.unsqueeze(0), sample_rate


def sine(device,
         double=False,
         trajectories_to_sample=100,
         t_nsamples=200,
         num_pi=4):
    t_nsamples_ref = 1000
    t_nsamples = int(t_nsamples_ref / 4 * num_pi)

    t_end = num_pi * np.pi
    t_begin = t_end / t_nsamples

    if double:
        ti = torch.linspace(t_begin, t_end, t_nsamples).to(device).double()
    else:
        ti = torch.linspace(t_begin, t_end, t_nsamples).to(device)

    def sampler(t, x0=0):
        # return torch.sin(t + x0)
        return torch.sin(t + x0) + torch.sin(
            4 * (t + x0)) + 0.5 * torch.sin(12 * (t + x0))

    x0s = torch.linspace(0, 16 * torch.pi, trajectories_to_sample)
    trajs = []
    for x0 in x0s:
        trajs.append(sampler(ti, x0))
    y = torch.stack(trajs)
    trajectories = y.view(trajectories_to_sample, -1, 1)
    sample_rate = t_nsamples / (t_end - t_begin) * 2 * np.pi
    return trajectories, ti, sample_rate


def time_sine(device,
              double=False,
              trajectories_to_sample=100,
              t_nsamples=201):
    """Generate sine data in time series fashion

    Args:
        device (_type_): _description_
        double (bool, optional): _description_. Defaults to False.
        trajectories_to_sample (int, optional): _description_. Defaults to 100.
        t_nsamples (int, optional): _description_. Defaults to 201.

    Returns:
        _type_: _description_
    """
    # (total_length - window_width) / stride - 1 = n_windows
    t_end = 20.0
    t_begin = t_end / t_nsamples
    window_width = t_nsamples - (trajectories_to_sample + 1)
    if double:
        ti = torch.linspace(t_begin, t_end, t_nsamples).to(device).double()
    else:
        ti = torch.linspace(t_begin, t_end, t_nsamples).to(device)

    def sampler(t):
        return torch.sin(t)
        return torch.sin(t) + torch.sin(2 * (t)) + 0.5 * torch.sin(11 * (t))

    traj = sampler(ti)
    # traj = torch.cat([traj.reshape(-1,1), ti.reshape(-1,1)], axis=-1)
    start = 0
    trajs, t = [], []
    for i in range(trajectories_to_sample):
        end = window_width + start
        trajs.append(traj[start:end].unsqueeze(-1))
        t.append(ti[start:end])
        start += 1
    if trajs[-1].shape != trajs[0].shape:
        trajs.pop()
        t.pop()
    y, t = torch.stack(trajs), torch.stack(t)
    if double:
        y = y.to(device).double()
        t = t.to(device).double()
    else:
        y = y.to(device)
        t = y.to(device)

    return y, t


# def arima(device, double=False, window_width=1000):
#     arparams = np.array([.85, -.15])
#     maparams = np.array([])
#     ar = np.r_[1, -arparams]  # add zero-lag and negate
#     ma = np.r_[1, maparams]  # add zero-lag
#     AR_object = ArmaProcess(ar, ma)
#     df = AR_object.generate_sample(nsample=100001).reshape(-1, 1)

#     trajs = []
#     start = 0
#     while start + window_width < len(df):
#         end = start + window_width
#         trajs.append(df[start:end])
#         start += 12
#     if len(trajs[-1]) != len(trajs[-2]):
#         trajs.pop()
#     trajs = np.stack(trajs, axis=0)
#     # t = torch.linspace(20 / window_width, 20, window_width)
#     t = torch.arange(window_width)
#     if double:
#         t = t.to(device).double()
#         trajs = torch.from_numpy(trajs).to(device).double()
#     else:
#         t = t.to(device)
#         trajs = torch.from_numpy(trajs).to(device)
#     return trajs, t.unsqueeze(0)


#  Real-world dataset
def solete_solar(
        device,
        double=False,
        features=['TEMPERATURE[degC]', 'POA Irr[kW1m2]', 'P_Solar[kW]'],
        window_width=24 * 12 * 2):
    df = pd.read_csv("datasets/SOLETE_clean_5min.csv",
                     index_col=0,
                     parse_dates=True,
                     infer_datetime_format=True)
    df = df.sort_index()
    df = df[features].values
    trajs = []
    start = 0
    while start + window_width < len(df):
        end = start + window_width
        trajs.append(df[start:end])
        start += window_width // 8
    if len(trajs[-1]) != len(trajs[-2]):
        trajs.pop()
    trajs = np.stack(trajs, axis=0)
    # t = torch.linspace(20 / window_width, 20, window_width)
    t = torch.arange(window_width) / 12
    if double:
        t = t.to(device).double()
        trajs = torch.from_numpy(trajs).to(device).double()
    else:
        t = t.to(device)
        trajs = torch.from_numpy(trajs).to(device)
    return trajs, t.unsqueeze(0)


def guangdong(device, double, features=["value"], window_width=24 * 4 * 2):
    df = pd.read_csv("datasets/gd_wind_site.csv")
    df = df[features].values[:96 * 300]
    trajs = []
    # trajs, ts = [], []
    start = 0
    while start + window_width < len(df):
        end = start + window_width
        trajs.append(df[start:end])
        # ts.append(t[start:end])
        start += window_width // 2
    if len(trajs[-1]) != len(trajs[-2]):
        trajs.pop()
        # ts.pop()
    trajs = np.stack(trajs, axis=0)
    # ts = torch.stack(ts, axis=0)
    t = torch.arange(window_width) / 96
    # t = torch.linspace(window_width / window_width, 20, window_width)
    if double:
        # ts = ts.to(device).double()
        t = t.to(device).double()
        trajs = torch.from_numpy(trajs).to(device).double()
    else:
        # ts = ts.to(device)
        t = t.to(device)
        trajs = torch.from_numpy(trajs).to(device)
    # trajs = torch.cat([trajs, ts.unsqueeze(-1)], axis=-1)
    return trajs, t.unsqueeze(0)


def GEF(device, double, features=["temp", "load"], window_width=24 * 2):
    df = pd.read_csv("datasets/GEF_11_14.csv")
    df = df[features].values
    trajs = []
    # trajs, ts = [], []
    start = 0
    while start + window_width < len(df):
        end = start + window_width
        trajs.append(df[start:end])
        # ts.append(t[start:end])
        start += window_width // window_width
    if len(trajs[-1]) != len(trajs[-2]):
        trajs.pop()
        # ts.pop()
    trajs = np.stack(trajs, axis=0)
    # ts = torch.stack(ts, axis=0)
    t = torch.arange(window_width) / 48
    # t = torch.linspace(window_width / window_width, 20, window_width)
    if double:
        # ts = ts.to(device).double()
        t = t.to(device).double()
        trajs = torch.from_numpy(trajs).to(device).double()
    else:
        # ts = ts.to(device)
        t = t.to(device)
        trajs = torch.from_numpy(trajs).to(device)
    # trajs = torch.cat([trajs, ts.unsqueeze(-1)], axis=-1)
    return trajs, t.unsqueeze(0)


def generate_data_set(name,
                      device,
                      double=False,
                      batch_size=128,
                      extrap=0,
                      trajectories_to_sample=100,
                      percent_missing_at_random=0.0,
                      normalize=True,
                      test_set_out_of_distribution=True,
                      noise_std=None,
                      t_nsamples=200,
                      observe_stride=1,
                      predict_stride=1,
                      observe_steps=200,
                      seed=0,
                      **kwargs):
    setup_seed(seed)
    if name == "nrel":
        trajectories, t, sample_rate = nrel(
            device,
            double,
            transformed=kwargs.get("transformed"),
            window_width=kwargs.get("window_width"))
    elif name == "sine":
        trajectories, t, sample_rate = sine(device, double,
                                            trajectories_to_sample, t_nsamples)
    # elif name == "arima":
    #     trajectories, t, sample_rate = arima(
    #         device, double, window_width=kwargs.get("window_width"))
    elif name == "time_sine":
        trajectories, t, sample_rate = time_sine(device, double,
                                                 trajectories_to_sample,
                                                 t_nsamples)
    elif name == "solete_solar":
        trajectories, t, sample_rate = solete_solar(device, double)

    elif name == "gef":
        trajectories, t, sample_rate = GEF(
            device,
            double,
        )
    elif name == "guangdong":
        trajectories, t, sample_rate = guangdong(
            device,
            double,
        )
    elif name == "mfred":
        trajectories, t, sample_rate = mfred(
            device, double, window_width=kwargs.get("window_width"))

    else:
        raise ValueError("Unknown Dataset To Test")

    if not extrap:
        bool_mask = torch.FloatTensor(
            *trajectories.shape).uniform_() < (1.0 - percent_missing_at_random)
        if double:
            float_mask = (bool_mask).float().double().to(device)
        else:
            float_mask = (bool_mask).float().to(device)
        trajectories = float_mask * trajectories

    if noise_std:
        trajectories += torch.randn(trajectories.shape).to(device) * noise_std

    train_split = int(0.8 * trajectories.shape[0])
    test_split = int(0.9 * trajectories.shape[0])
    if test_set_out_of_distribution:
        train_trajectories = trajectories[:train_split, :, :]
        val_trajectories = trajectories[train_split:test_split, :, :]
        test_trajectories = trajectories[test_split:, :, :]
        if name.__contains__("time"):
            train_t = t[:train_split]
            val_t = t[train_split:test_split]
            test_t = t[test_split:]
        else:
            train_t = t
            val_t = t
            test_t = t

    else:
        traj_index = torch.randperm(trajectories.shape[0])
        train_trajectories = trajectories[traj_index[:train_split], :, :]
        val_trajectories = trajectories[
            traj_index[train_split:test_split], :, :]
        test_trajectories = trajectories[traj_index[test_split:], :, :]
        if name.__contains__("time"):
            train_t = t[traj_index[:train_split]]
            val_t = t[traj_index[train_split:test_split]]
            test_t = t[traj_index[test_split:]]
        else:
            train_t = t
            val_t = t
            test_t = t
    if normalize:
        len_train, len_val, len_test = len(train_trajectories), len(
            val_trajectories), len(test_trajectories)
        dim = trajectories.shape[2]
        train_mean = torch.reshape(train_trajectories, (-1, dim)).mean(0)
        train_std = torch.reshape(train_trajectories, (-1, dim)).std(0)
        train_trajectories = (torch.reshape(train_trajectories, (-1, dim)) -
                              train_mean) / train_std
        val_trajectories = (torch.reshape(val_trajectories,
                                          (-1, dim)) - train_mean) / train_std
        test_trajectories = (torch.reshape(test_trajectories,
                                           (-1, dim)) - train_mean) / train_std
        train_trajectories = train_trajectories.reshape((len_train, -1, dim))
        val_trajectories = val_trajectories.reshape((len_val, -1, dim))
        test_trajectories = test_trajectories.reshape((len_test, -1, dim))
    else:
        train_std = 1
        train_mean = 0

    rand_idx = torch.randperm(len(train_trajectories)).tolist()
    train_trajectories = train_trajectories[rand_idx]
    dltrain = DataLoader(
        train_trajectories,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=lambda batch: basic_collate_fn(
            batch,
            train_t,
            data_type="train",
            extrap=extrap,
            observe_stride=observe_stride,
            predict_stride=predict_stride,
            observe_steps=observe_steps),
    )
    dlval = DataLoader(
        val_trajectories,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=lambda batch: basic_collate_fn(
            batch,
            val_t,
            data_type="test",
            extrap=extrap,
            observe_stride=observe_stride,
            predict_stride=predict_stride,
            observe_steps=observe_steps),
    )
    dltest = DataLoader(
        test_trajectories,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=lambda batch: basic_collate_fn(
            batch,
            test_t,
            data_type="test",
            extrap=extrap,
            observe_stride=observe_stride,
            predict_stride=predict_stride,
            observe_steps=observe_steps),
    )

    b = next(iter(dltrain))
    input_dim = b["observed_data"].shape[-1]
    output_dim = b["data_to_predict"].shape[-1]
    input_timesteps = b["observed_data"].shape[1]
    output_timesteps = b["data_to_predict"].shape[1]
    print(train_mean)
    print(train_std)
    return (input_dim, output_dim, sample_rate, t, dltrain, dlval, dltest,
            input_timesteps, output_timesteps, train_mean, train_std)


# (input_dim, output_dim, dltrain, dlval, dltest, input_timesteps,
#             output_timesteps, train_mean, train_std) = generate_data_set("nrel",
#                       0,
#                       extrap=1,
#                       normalize=False,
#                       batch_size=3, window_width=24*12*2, transformed=False)

# for b in dltrain:
#     for k in b:
#         print(k)
#         # print(b[k].shape)
#     break

# print(
#     input_dim,
#     output_dim,
#     dltrain,
#     dlval,
#     dltest,
#     input_timesteps,
#     output_timesteps,
# )