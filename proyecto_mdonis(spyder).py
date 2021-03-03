# -*- coding: utf-8 -*-
"""
Created on Mon Mar  1 16:59:23 2021

@author: mdonis
"""

import os
import numpy as np
import pandas as pd
os.chdir('C:/Users/mdonis/Documents/Maestría Data Science/Ciencia de Datos en Python/Proyecto')

ds = np.load('proyecto_training_data.npy')

np.random.shuffle(ds)

n = int(ds.shape[0]*0.8)

train_data = ds[0:n,:]
test_data = ds[n:,:]

# Mismos sets de datos convertidos en dataframes
df = pd.DataFrame(ds, columns = ['SalePrice','OverallQual','1stFlrSF','TotRmsAbvGrd','YearBuilt','LotFrontage'])
train_df = pd.DataFrame(train_data, columns = ['SalePrice','OverallQual','1stFlrSF','TotRmsAbvGrd','YearBuilt','LotFrontage'])
test_df = pd.DataFrame(test_data, columns = ['SalePrice','OverallQual','1stFlrSF','TotRmsAbvGrd','YearBuilt','LotFrontage'])

# Análisis exploratorio numpy
mean = train_data.mean(axis = 0)
maxx = train_data.max(axis = 0)
minn = train_data.min(axis = 0)

# Análisis exploratorio pandas
mean_pandas = df.mean(axis = 0)
max_pandas = df.max(axis = 0)
min_pandas = df.min(axis = 0)
std_pandas = df.std(axis = 0)
rng_pandas = max_pandas - min_pandas
