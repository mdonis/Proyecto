# -*- coding: utf-8 -*-
"""
Created on Sat Mar 13 10:06:16 2021

@author: m_don
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Mar  1 16:59:23 2021

@author: mdonis
"""

import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

#os.chdir('C:/Users/mdonis/Documents/Maestría Data Science/Ciencia de Datos en Python/Proyecto')
os.chdir('C:/Users/m_don/Documents/DataScience/Ciencia de Datos en Python/Proyecto')


class regression():
    def __init__(self, npy_filename):
        self.npy_filename = np.load('{}.npy'.format(npy_filename))

# Descarga
ds = np.load('proyecto_training_data.npy')

# Mismos sets de datos convertidos en dataframes
varss = ['SalePrice','OverallQual','1stFlrSF','TotRmsAbvGrd','YearBuilt','LotFrontage']
df = pd.DataFrame(ds, columns = varss)

# Relación columnas ds-df
d = {'SalePrice':0,'OverallQual':1,'1stFlrSF':2,'TotRmsAbvGrd':3,'YearBuilt':4,'LotFrontage':5}

#%% 2. Random-Slicing:
# Random
np.random.shuffle(ds)

# Slicing 80%-20%
s = int(ds.shape[0]*0.8)

train = ds[0:s,:]
test = ds[s:,:]

# Análogos en dfs
train_df = pd.DataFrame(train, columns = varss)
test_df = pd.DataFrame(test, columns = varss)

#%% 3. Análisis Exploratorio:
# Análisis exploratorio pandas
med = df.median(axis = 0)
maxx = df.max(axis = 0)
minn = df.min(axis = 0)
rang = maxx - minn
std = df.std(axis = 0)


# Resultado análisis exploratorio
exp_analysis = pd.concat([med,maxx,minn,rang,std], keys = ['median','max','min','range','std'], axis = 1)
print(exp_analysis)

#%% 4. histogramas
h = {}
sns.set_theme();

for i in df.columns:
    h['{}'.format(i)] = sns.displot(df['{}'.format(i)], kde=True)
    plt.show()
    

#%% 5. Para cada variable independiente x:
    
# Correlación vectorizada
r = {}
p = {}
for i in d:
    if np.isnan(train[:,d[i]].sum()) == False:
        y = train[:,0]
        x = train[:,d[i]]
        n = y.shape[0]
    else:
        train_nonna = train[~np.isnan(train).any(axis=1)]
        y = train_nonna[:,0]
        x = train_nonna[:,d[i]]
        n = y.shape[0]
        
    r['{}'.format(i)] = (n*(x*y).sum() - x.sum()*y.sum()) / ((n*(x**2).sum() - x.sum()**2)*(n*(y**2).sum() - y.sum()**2))**(1/2)
    
    # scatterplot
    p['{}'.format(i)] = plt.scatter(x, y, s = 0.3)
    plt.title('{} r = {}'.format(varss[d[i]],r[i]))
    plt.xlabel(varss[d[i]])
    plt.ylabel(varss[0])
    plt.show()
    

#%% 6 y 7. Función de entrenamiento para el modelo de regresión lineal


y = train[:,d['SalePrice']].reshape(-1, 1)
x = train[:,d['OverallQual']].reshape(-1, 1)

#y = np.array(list(range(1,100))).reshape(-1, 1)
#x = np.array(list(range(1,100))).reshape(-1, 1)

b = np.ones_like(x)
mat_a = np.hstack([x,b])
# Inicialización de parámetros
#b1, b0 = 43656.32107902, -84340.59865125
b1, b0 = 1, 1
# Iteraciones
epochs = 20
impr_error = 1
learn_rate = 0.01
n = y.shape[0]
bi = {}
err = list(range(0,epochs+1,impr_error))

for i in range(epochs):
    # Parámetros iniciales de la iteración i
    if i == 0:
        vect = np.array([[b1],[b0]])
    else:
        vect = bi[i-1]
    
    # Prediciones
    y_h = np.matmul(mat_a, vect)
    
    # Error
    e = (1/(2*n))*((y-y_h)**2).sum()
    
    # Almacenar el error en un vector
    if i in err:
        if i == 0:
            errors = np.array([e])
        else:
            errors = np.append(errors,[e])
    
    # Gradientes
    mat_b = np.transpose(y_h-y)
    b_grad = (np.matmul(mat_b,mat_a)/n).reshape(-1,1)
    
    # Betas
    mat1 = np.hstack([vect,b_grad])
    mat2 = np.array([[1],[-learn_rate]])
    
    # Parámetros resultantes de la iteración i
    bi[i] = np.matmul(mat1,mat2)

# Gráfica del error

plt.plot(range(1,epochs+1,impr_error),errors)
plt.show()


# scatterplot
plt.plot(x, y_h)
plt.scatter(x, y, s = 0.3)
plt.show()


# Avance del modelo
n = 1
models = list(range(0,epochs,n))
for i in models:
    if i == n*(epochs/n-1):
        plt.plot(x,np.matmul(mat_a, bi[i]), color='r')
    else:
        plt.plot(x,np.matmul(mat_a, bi[i]), alpha=0.3)
plt.scatter(x, y, s = 0.3)


regr = LinearRegression().fit(train[:,1],train[:,0])



import numpy as np
from sklearn.linear_model import LinearRegression
X = np.array([[1, 1], [1, 2], [2, 2], [2, 3]])
# y = 1 * x_0 + 2 * x_1 + 3
y = np.dot(X, np.array([1, 2])) + 3
reg = LinearRegression().fit(X, y)
reg.score(X, y)
1.0
reg.coef_
array([1., 2.])
reg.intercept_
reg.predict(np.array([[3, 5]]))
array([16.])




import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score

# Load the diabetes dataset
diabetes_X, diabetes_y = datasets.load_diabetes(return_X_y=True)

# Use only one feature
diabetes_X = diabetes_X[:, np.newaxis, 2]

# Split the data into training/testing sets
diabetes_X_train = diabetes_X[:-20]
diabetes_X_test = diabetes_X[-20:]

# Split the targets into training/testing sets
diabetes_y_train = diabetes_y[:-20]
diabetes_y_test = diabetes_y[-20:]

# Create linear regression object
regr = linear_model.LinearRegression()

# Train the model using the training sets
regr.fit(diabetes_X_train, diabetes_y_train)

# Make predictions using the testing set
diabetes_y_pred = regr.predict(diabetes_X_test)

# The coefficients
print('Coefficients: \n', regr.coef_)
# The mean squared error
print('Mean squared error: %.2f'
      % mean_squared_error(diabetes_y_test, diabetes_y_pred))
# The coefficient of determination: 1 is perfect prediction
print('Coefficient of determination: %.2f'
      % r2_score(diabetes_y_test, diabetes_y_pred))

# Plot outputs
plt.scatter(diabetes_X_test, diabetes_y_test,  color='black')
plt.plot(diabetes_X_test, diabetes_y_pred, color='blue', linewidth=3)

plt.xticks(())
plt.yticks(())

plt.show()


