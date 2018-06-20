# coding: utf-8

# Machine Learning aplicado a la predicción de precio de criptomonedas, utilizando valores anteriores de las principales critpomonedas, valores de índices de stocks, y trends de búsqueda. 

# Manipulación de datos
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt # Plots

import requests #Scraping
from bs4 import BeautifulSoup #Scraping
from pytrends.request import TrendReq #GoogleTrends

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

class Data:
    "Recibe una lista de nombres archivos csv de stocks y administra sus datos."
    __dir_data = "data"
    
    def __init__(self,datafile=''):
        if not datafile == '':
            self.setdata(datafile)
        
    def setdata(self,datafile):
        self.data = pd.read_csv(datafile)
        self.data['Date'] = pd.to_datetime(self.data['Date'])
        self.data.set_index('Date', inplace=True)
        self.fecha_comienzo = self.data.index[0]
        self.fecha_fin = self.data.index[-1]
        
    ''''
    Funcion para actualizar el dataset completo (todas las criptomonedas, stocks y keywords)
    '''
    def update_datasets(self,crypto_names,stock_names,fecha_comienzo,fecha_fin,filename = 'data'):  
        self.crypto_names = crypto_names
        self.stock_names = stock_names
        self.fecha_comienzo = fecha_comienzo
        self.fecha_fin = fecha_fin
        
        cryptodata = self.__get_crypto()
        stockdata = self.__get_stocks()
        trendsdata = self.__get_trends()
        
        # Seleccionar los datos dentro del rango de fechas
        self.data = pd.DataFrame(pd.date_range(start=self.fecha_comienzo, end=self.fecha_fin, freq='D'),columns=['Date'])
        self.data.set_index('Date', inplace=True)
        
        # Hacer join de los dataframes (fechas - precios) para tener precios para las fechas deseadas
        self.data = pd.merge(self.data, cryptodata,  how='left', left_index=True, right_index=True)
        self.data = pd.merge(self.data, stockdata,  how='left', left_index=True, right_index=True)
        self.data = pd.merge(self.data, trendsdata,  how='left', left_index=True, right_index=True)
        self.data.replace('-',np.nan,inplace=True) 
        self.data = self.data.interpolate().ffill().fillna(0) #Completar campos faltantes
    
        # Guardar Datos
        if not os.path.exists(self.__dir_data):
            os.makedirs(self.__dir_data)
        path = self.__dir_data + "/" + filename + '.csv'
        self.data.to_csv(path, index=True)
        print('Datos recolectados, archivo: ' + path)
    
    def __get_stocks(self):    
        "Arma un dataframe con todos los datos de los archivos de stocks"
        stock_dataframe =  pd.DataFrame()
        
        for stock in self.stock_names:
            df = pd.read_csv('data/original/' + stock + '.csv')

            df.drop(['Open','High','Low','Adj Close','Volume'], axis=1, inplace=True)
            df.rename(columns={'Close': stock}, inplace=True)
            df.set_index('Date', inplace=True)
    
            stock_dataframe = pd.concat([stock_dataframe, df], axis=1)

            print('Stock procesado: ' + stock)
        
        stock_dataframe.index.name = 'Date'
        
        return stock_dataframe
    
    '''
    Toma una fecha de comienzo, una fecha de fin, y un arreglo de critpomonedas a buscar.
    Retorna un dataframe con toda la información.

    Modo de utilización de la función:
    marketcaps = get_crypto(criptomonedas=['bitcoin','ethereum','ripple','dash','litecoin','monero','ethereum-classic','nem','augur'],
                          fecha_comienzo='2017-04-02', fecha_fin='2018-03-25')
    #marketcaps.to_csv('csv/marketcaps.csv', index=True)
    '''
    #Función para hacer scraping en coinmarketcap - solo usada por la función get_crypto - 
    def __scrape_crypto(self,criptomoneda):
        # Reformattear las fechas para hacer el request HTTP
        start = self.fecha_comienzo.replace('-', '')
        end = self.fecha_fin.replace('-', '')

        # Retrieve historical snapshot data from date
        page = requests.get('https://coinmarketcap.com/currencies/'+ 
                            criptomoneda +'/historical-data/?start='+ 
                            start + '&end='+ end)
        soup = BeautifulSoup(page.content, 'html.parser')

        print('Criptomoneda procesada: ' + criptomoneda)

        # Pasar la tabla de precios a un csv
        table = soup.find_all('table')[0] # Buscar la tabla en el html

        rows = []
        for row in table.find_all('tr')[1:]:
            rows.append([val.text.replace(',', '') for val in row.find_all('td')]) # Obtener cada valor   

        return rows

    def __get_crypto(self):          
        # Crear rango de fechas
        crypto_dataframe = pd.DataFrame(pd.date_range(start=self.fecha_comienzo, end=self.fecha_fin, freq='D'),columns=['Date'])
        crypto_dataframe.set_index('Date', inplace=True)
        headers = ['Date','Open','High','Low','Close','Volume','Market Cap']

        for moneda in self.crypto_names:
            rows = self.__scrape_crypto(moneda)

            for row in rows:
                row[0] = pd.to_datetime(row[0]) # Dar formato de fecha 

            # Crear data frame con datos
            df = pd.DataFrame(data=rows, columns=headers)
            df = df.iloc[::-1] # Invertir, para que quede del más viejo al más nuevo
            df.set_index('Date', inplace=True)

            # Juntar el dataframe actual con el resto para formar un dataframe de marketcaps de todas las monedas
            df.drop(['Open','High','Low','Volume','Market Cap'], axis=1, inplace=True)
            df.rename(columns={'Close': moneda}, inplace=True)
            crypto_dataframe = pd.merge(crypto_dataframe, df,  how='left', left_index=True, right_index=True)

        return crypto_dataframe

    def __get_trends(self):    
        "Arma un dataframe con todos los datos de los archivos de stocks"
        trends_dataframe =  pd.DataFrame()
        
        #for moneda in self.crypto_names:
        for moneda in ['bitcoin']:
            df = pd.read_csv('data/original/trends_' + moneda + '.csv')
          
            df.rename(columns={df.columns[0]: 'Date',df.columns[1]:'trend_' + moneda}, inplace=True)
            df.set_index('Date', inplace=True)
    
            trends_dataframe = pd.concat([trends_dataframe, df], axis=1)

            print('Moneda procesada: ' + moneda)
        
        trends_dataframe.index.name = 'Date'
        trends_dataframe.replace('<1',0,inplace=True) 
        
        return trends_dataframe
    
    '''
    (DEPRECADO - GOOGLE BLOQUEA IP)
    Toma una fecha de comienzo, una fecha de fin, y un arreglo de keywords a buscar.
    Retorna un dataframe con toda la información.

    Modo de utilización de la función:
    alltrends = get_trends(keywords=['bitcoin','ethereum','ripple','dash','litecoin','monero','ethereum classic','nem','augur'],
                          fecha_comienzo='2017-04-02', fecha_fin='2018-03-25')
    alltrends.to_csv('csv/trends.csv', index=True)

    '''
    def __get_trends_pytrends(self):
        # Login to Google. Only need to run this once, the rest of requests will use the same session.
        pytrend = TrendReq()
        
        # Reformatear arreglo de keywords (Google Trends acepta de a 5 por vez)
        keywords = self.crypto_names# + self.stock_names
        keywords_reformatted=[]
        for i in range(0, len(keywords), 5):
            keywords_reformatted.append(keywords[i:i+5])

        # Crear rango de fechas
        trendsdata = pd.DataFrame(pd.date_range(start=self.fecha_comienzo, end=self.fecha_fin, freq='D'),columns=['Date'])
        trendsdata.set_index('Date', inplace=True)

        for k in keywords_reformatted:
            # Create payload and capture API tokens. Only needed for interest_over_time(), interest_by_region() & related_queries()
            pytrend.build_payload(kw_list=k, timeframe='' + self.fecha_comienzo + ' ' + self.fecha_fin)
            # Interest Over Time
            interest_over_time_df = pytrend.interest_over_time()
            interest_over_time_df.drop('isPartial', axis=1, inplace=True) #Sacar la columna de isPartial
            for keyword in interest_over_time_df.columns:
                interest_over_time_df.rename(columns={keyword: 'trend_' + keyword}, inplace=True)
            
            print('Keywords procesadas: ' + str(k))
            # Mezclar los dataframes, agregando el precio del segundo dataset
            trendsdata = pd.merge(trendsdata, interest_over_time_df,  how='left', left_index=True, right_index=True)

        return trendsdata

    def plot(self):
        "Graficar los datos del DataManager"
        # Graficar TODAS las columnas juntas
        plt.figure(figsize=(20,5))
        plt.title("Stock Prices", fontsize=25)
        for stock in np.arange(0, len(self.data.columns), 1):
            plt.plot(self.data[self.data.columns[stock]], label=self.data.columns[stock])  
        plt.legend()
        plt.show()
        
        # Graficar cada columna
        i = 1
        # plot each column
        plt.figure(figsize=(20,5*len(self.data.columns)))
        for stock in np.arange(0, len(self.data.columns), 1):
            plt.subplot(len(self.data.columns), 1, i)
            plt.plot(self.data[self.data.columns[stock]], label=self.data.columns[stock])   
            plt.legend()
            i+=1
        plt.show()
        
    def create_dataset(self,moneda,fecha_comienzo,fecha_fin):
        """
        Convierte el conjunto de datos en dos conjuntos X e Y (input/output)
 
        :param moneda: nombre de la columna en el dataframe del target
        :returns: conjuntos X e Y como numpy array
        """
        # Seleccionar los datos dentro del rango de fechas
        dataset = self.data[fecha_comienzo:fecha_fin]
        dataset = dataset.interpolate().ffill().fillna(0) #Completar campos faltantes
        
        x = dataset
        y = dataset[['bitcoin']]
        
        return {'x': x, 'y': y, 'moneda':moneda}

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