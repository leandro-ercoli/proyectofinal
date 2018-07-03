# coding: utf-8

# Machine Learning aplicado a la predicción de precio de criptomonedas, utilizando valores anteriores de las principales critpomonedas, valores de índices de stocks, y trends de búsqueda. 

# Manipulación de datos
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt # Plots

# Keras
from keras.preprocessing.sequence import TimeseriesGenerator
from keras.layers.core import Dense,Activation,Dropout
from keras.layers.recurrent import LSTM
from keras.models import Sequential
import time
import warnings
from sklearn import preprocessing
from sklearn.metrics import mean_squared_error
from math import sqrt

#Manejo de archivos
import os
import tempfile
import shutil

class Predictor:    
    __dir_predicciones = "predicciones"
    def __init__(self,dataset):
        """
        Constructor de la clase Predictor (preprocesador, entrenamiento y prediccion)
        
        :param dataset: un conjunto de datos de la clase Data 
        """
        self.x = dataset['x'].copy()
        self.y = dataset['y'].copy()
        self.moneda = dataset['moneda']
        print("New Predictor")       
          
    def __data_preprocessing(self,test_size,window_size=1,dia_futuro=1, test_skip=0):
        # Dividir el conjunto de x e y en conjuntos de training y testing
        # La y se corre dia_futuro-1 porque el generator del predictor asigna cada sample de x con un valor de y +1
        # (es decir, que genera un target de un dia a futuro)
        train_x = self.x[0:len(self.x)-test_size]
        train_y = self.y[(dia_futuro-1):len(self.y)-test_size+(dia_futuro-1)]
        test_x  = self.x[len(self.x)-test_size:len(self.x)-(dia_futuro-1)]
        test_y  = self.y[len(self.y)-test_size+(dia_futuro-1):]
        self.__real_testprice = test_y[window_size:]
        
        # Normalización: Escalar los datos entre [0,1]. Hay que escalar con respecto al training set, como si el test set no estuviera.
        self.__scalerX = preprocessing.MinMaxScaler(feature_range = (0,1)).fit(train_x)
        train_x_scaled = self.__scalerX.transform(train_x)
        test_x_scaled = self.__scalerX.transform(test_x)
        self.__scalerY = preprocessing.MinMaxScaler(feature_range = (0,1)).fit(train_y.values.reshape(-1,1))
        train_y_scaled = self.__scalerY.transform(train_y.values.reshape(-1,1))
        test_y_scaled = self.__scalerY.transform(test_y.values.reshape(-1,1))
    
        '''
        TimeseriesGenerator crea una secuencia de samples, cada una de longitud *length*, 
        tomadas de a *sampling_rate* timesteps. 
        El end_index es la ultima posición que se va a tomar en los datos (tiene que ser +1 que la que se quiere)
        El start_index es la primera posición desde la que se van a tomar los datos.
        '''
        validation_split = 0.8        
        batch_s = 15
        self.__training_sequence = TimeseriesGenerator(train_x_scaled, train_y_scaled,
                               length=window_size, sampling_rate=1, 
                                end_index=int(len(train_x)*validation_split),
                               batch_size=batch_s)
        self.__validation_sequence = TimeseriesGenerator(train_x_scaled, train_y_scaled,
                                       length=window_size, sampling_rate=1, 
                                        start_index=int(len(train_x)*validation_split), end_index=len(train_x),
                                       batch_size=batch_s)
        self.__testing_sequence = TimeseriesGenerator(test_x_scaled, test_y_scaled,
                                       length=window_size, sampling_rate=1, 
                                        end_index=len(test_x),
                                       batch_size=1)
        
    def __red_neuronas(self,neuronas,epochs,verbose):
        self.model = Sequential()
        self.model.add(LSTM(units = neuronas, dropout=0, 
                       input_shape=(self.__training_sequence[0][0].shape[1], #window_size
                                    self.__training_sequence[0][0].shape[2]))) #features
        self.model.add(Dense(units = 1, activation = 'linear'))
        self.model.compile(optimizer = 'adam', loss = 'mean_squared_error')

        self.history = self.model.fit_generator(self.__training_sequence,
                                                validation_data=self.__validation_sequence,
                                                epochs=epochs,verbose=verbose)        
    
    def predict(self):
        predicted = self.model.predict_generator(self.__testing_sequence)
        predicted = predicted.reshape(-1,1)
        predicted = self.__scalerY.inverse_transform(predicted)

        return predicted
    
    def entrenar_predecir(self,test_size,window_size,dia_futuro,neuronas,epochs,verbose=1,visualizar=1):               
        self.__data_preprocessing(test_size=test_size,window_size=window_size,dia_futuro=dia_futuro)
        print("Datos Preprocesados. Entrenando...")
        self.__red_neuronas(neuronas=neuronas,epochs=epochs,verbose=verbose)
        predicted = self.predict()
              
        # Armar dataframes con predicciones e historial de entrenamiento
        training_history = pd.DataFrame() # Dataframe de entrenamiento
        dataset_name = 'f' + str(dia_futuro) + 'v' + str(window_size) + 'n' + str(neuronas) + 'e' + str(epochs)
        self.__real_testprice['prediccion_'+dataset_name] = predicted
        training_history = pd.concat([training_history, pd.DataFrame(data=self.history.history['loss'], columns=[dataset_name +'_loss'])], axis=1)
        training_history = pd.concat([training_history, pd.DataFrame(data=self.history.history['val_loss'], columns=[dataset_name + 'val_loss'])], axis=1)

        # Guardar datos        
        if not os.path.exists(self.__dir_predicciones):
            os.makedirs(self.__dir_predicciones) # Crear carpeta de predicciones
        datasetfolder = self.__dir_predicciones + "/"  + self.moneda + "/"+ dataset_name
        if (os.path.exists(datasetfolder)):
            tmp = tempfile.mktemp(dir=os.path.dirname(datasetfolder))
            shutil.move(datasetfolder, tmp) # Mover el dir a una ubicacion temporal
            shutil.rmtree(tmp) # Eliminar la carpeta que existía
        os.makedirs(datasetfolder)   
        
        self.__real_testprice.to_csv(datasetfolder + '/predicciones.csv', index=True)
        self.__real_testprice.to_json(datasetfolder + '/predicciones.json',orient="columns")
        print('Predicciones guardadas en carpeta: ' + datasetfolder)

        training_history.to_csv(datasetfolder +  '/train_history.csv', index=False)
        training_history.to_json(datasetfolder + '/train_history.json',orient="columns")
        print('Entrenamiento guardado en carpeta: ' + datasetfolder)
        
        p = {'predicciones':self.__real_testprice, 
                'training_history':training_history,
                'neuronas':neuronas,
                'epochs':epochs,
                'test_size':test_size,
                'dataset_name':dataset_name,
                'window':window_size,
                'dia_futuro':dia_futuro } 
        
        if visualizar:
            self.visualizar(p)
        
        return p
    
    '''
    VISUALIZACION DE RESULTADOS
    '''
    def visualizar(self,prediccion):
        #Grafico de entrenamiento
        plt.figure(figsize=(20,5))
        plt.plot(prediccion['training_history'][prediccion['training_history'].columns[0]], label='loss')
        plt.plot(prediccion['training_history'][prediccion['training_history'].columns[1]], label='val_loss')
        plt.title("Training History", fontsize=40)
        plt.legend()
        plt.grid(color='grey', linestyle='-', linewidth=0.5)
        #plt.savefig(self.__datasetfolder + "/entrenamiento.png")
        plt.show()    

        # El precio real es la ultima columna del dataframe
        val_prediccion = prediccion['predicciones'][prediccion['predicciones'].columns[1]]
        val_real =  prediccion['predicciones'][prediccion['predicciones'].columns[0]]
        rmse = sqrt(mean_squared_error(val_real, val_prediccion))
        print("RMSE: " + str(rmse))
        
        plt.figure(figsize=(20,10))
        plt.plot(val_prediccion, color = (1, 0, 0), label = str(prediccion['dataset_name']))
        plt.plot(val_real, color = (0, 0, 1), label = 'Precio Real')
        plt.title("Predicción " + self.moneda + "(" + str(prediccion['dia_futuro']) + " dias en el futuro)", fontsize=40)
        plt.legend()
        plt.grid(color='grey', linestyle='-', linewidth=0.5)
        #plt.savefig(self.__datasetfolder + "/predicciones.png")
        plt.show()