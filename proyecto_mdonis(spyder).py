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
from sklearn import linear_model
from sklearn import metrics

#os.chdir('C:/Users/mdonis/Documents/Maestría Data Science/Ciencia de Datos en Python/Proyecto')
os.chdir('C:/Users/m_don/Documents/DataScience/Ciencia de Datos en Python/Proyecto')


class regression():
    varss = ['SalePrice','OverallQual','1stFlrSF','TotRmsAbvGrd','YearBuilt','LotFrontage']
    d = {'SalePrice':0,'OverallQual':1,'1stFlrSF':2,'TotRmsAbvGrd':3,'YearBuilt':4,'LotFrontage':5}
    predictable = d['SalePrice']
    r = {}
    
    def __init__(self, npy_filename):
        self.npy = np.load('{}.npy'.format(npy_filename))
        np.random.seed(30)
        np.random.shuffle(self.npy)
        s = int(self.npy.shape[0]*0.8)
        self.train = self.npy[0:s,:]
        self.test = self.npy[s:,:]
        
        self.y_train = self.train[:,self.predictable].reshape(-1, 1)
        self.y_test = self.test[:,self.predictable].reshape(-1,1)
        
        self.df = pd.DataFrame(self.npy, columns = self.varss)
        self.train_df = pd.DataFrame(self.train, columns = self.varss)
        self.test_df = pd.DataFrame(self.test, columns = self.varss)
    
    def exp_analysis(self):
        med = self.df.median(axis = 0)
        maxx = self.df.max(axis = 0)
        minn = self.df.min(axis = 0)
        rang = maxx - minn
        std = self.df.std(axis = 0)
        result = pd.concat([med,maxx,minn,rang,std], keys = ['median','max','min','range','std'], axis = 1)
        return result
    
    def hist(self):
        for i in self.df.columns:
            sns.displot(self.df['{}'.format(i)], kde=True)
            plt.show()
    
    def vars_select(self):
        for i in self.d:
            if np.isnan(self.train[:,self.d[i]].sum()) == False:
                y = self.train[:,self.predictable]
                x = self.train[:,self.d[i]]
                n = y.shape[0]
            else:
                train_nonna = self.train[~np.isnan(self.train).any(axis=1)]
                y = train_nonna[:,self.predictable]
                x = train_nonna[:,self.d[i]]
                n = y.shape[0]
                
            self.r['{}'.format(i)] = (n*(x*y).sum() - x.sum()*y.sum()) / ((n*(x**2).sum() - x.sum()**2)*(n*(y**2).sum() - y.sum()**2))**(1/2)
            
            # scatterplot
            plt.scatter(x, y, s = 0.3)
            plt.title('{} Where r = {}'.format(self.varss[self.d[i]],self.r[i]))
            plt.xlabel(self.varss[self.d[i]])
            plt.ylabel(self.varss[self.predictable])
            plt.show()
        return self.r
        
    @classmethod
    def change_var_to_predict(cls, var):
        regression.predictable = regression.d[var]
        
    def manual_model(self,x_name,epochs,error_freq,learn_rate):
        self.x_train = self.train[:,self.d[x_name]].reshape(-1, 1)
        b = np.ones_like(self.x_train)
        self.mat_a = np.hstack([self.x_train,b])
        b1, b0 = 1, 1
        self.epochs = epochs
        self.error_freq = error_freq
        self.learn_rate = learn_rate
        n = self.y_train.shape[0]
        self.bi = {}
        err = list(range(0,self.epochs+1,self.error_freq))
        
        for i in range(self.epochs):
            if i == 0:
                vect = np.array([[b1],[b0]])
            else:
                vect = self.bi[i-1]
            self.y_train_h_mm = np.matmul(self.mat_a, vect)
            e = (1/(2*n))*((self.y_train-self.y_train_h_mm)**2).sum()
            if i in err:
                if i == 0:
                    self.errors = np.array([e])
                else:
                    self.errors = np.append(self.errors,[e])
            mat_b = np.transpose(self.y_train_h_mm-self.y_train)
            b_grad = (np.matmul(mat_b,self.mat_a)/n).reshape(-1,1)
            mat1 = np.hstack([vect,b_grad])
            mat2 = np.array([[1],[-self.learn_rate]])
            self.bi[i] = np.matmul(mat1,mat2)
        
        # Mean Absolute Error (Para comparar con sklearn)
        error = (1/n)*(np.absolute(self.y_train-self.y_train_h_mm)).sum()
        plt.plot(range(1,self.epochs+1,self.error_freq),self.errors)
        plt.show()
        # scatterplot
        plt.plot(self.x_train, self.y_train_h_mm)
        plt.scatter(self.x_train, self.y_train, s = 0.3)
        plt.show()
        return str('Betas: '), self.bi, str('Errores: '), self.errors, str('Mean Absolute Error: '), error


    def model_evol(self,n):
        self.n = n
        models = list(range(0,self.epochs,self.n))
        for i in models:
            if i == self.n*(self.epochs/self.n-1):
                plt.plot(self.x_train,np.matmul(self.mat_a, self.bi[i]), color='r')
            else:
                plt.plot(self.x_train,np.matmul(self.mat_a, self.bi[i]), alpha=0.3)
        plt.scatter(self.x_train, self.y_train, s = 0.3)
    
    def sklearn_model(self,x_name):
        self.x_train = self.train[:,self.d[x_name]].reshape(-1,1)
        self.reg = linear_model.LinearRegression().fit(self.x_train, self.y_train)
        self.y_train_h_sm = self.reg.predict(self.x_train)
        error = metrics.mean_absolute_error(self.y_train, self.y_train_h_sm)
        b0 = self.reg.intercept_
        b1 = self.reg.coef_
        plt.plot(self.x_train, self.y_train_h_sm)
        plt.scatter(self.x_train, self.y_train, s = 0.3)
        plt.show()
        return b0, b1, error
    
    def both_models(self,x_name,number_of_y_predictions):
        x = number_of_y_predictions
        y_h_mm_subset = self.y_train_h_mm[0:x,]
        y_h_sm_subset = self.y_train_h_sm[0:x,]
        mean = (y_h_mm_subset + y_h_sm_subset)/2
        return str('manual model'),y_h_mm_subset, str('sklearn model'),y_h_sm_subset, str('mean_prediction'),mean
    
    def test_models(self,x_name):
        self.x_test = self.test[:,self.d[x_name]].reshape(-1,1)
        y_test_h_sm = self.reg.predict(self.x_test)
        error_sm = metrics.mean_absolute_error(self.y_test, y_test_h_sm)
        b = np.ones_like(self.x_test)
        mat_a = np.hstack([self.x_test,b])
        vect = self.bi[self.epochs-1]
        y_test_h_mm = np.matmul(mat_a, vect)
        error_mm = metrics.mean_absolute_error(self.y_test, y_test_h_mm)
        d = {'sklearn model': error_sm, 'manual model': error_mm}
        df = pd.DataFrame(d, index=[0])
        values = df.values.flatten()
        sns.barplot(x = df.columns,y=values).set_title('{} prediction errors'.format(x_name))
        plt.show()
        return str('{} sklearn error: '.format(x_name)),error_sm, str('{} manual error: '.format(x_name)), error_mm
        
obj = regression('proyecto_training_data')

obj.exp_analysis()
obj.hist()
obj.vars_select()

obj.manual_model('OverallQual', 200, 10, 0.049)
obj.model_evol(1)

obj.sklearn_model('OverallQual')

obj.both_models('OverallQual',3)

obj.test_models('OverallQual')

regression.predictable
regression.change_var_to_predict('SalePrice')



