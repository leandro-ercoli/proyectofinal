# coding: utf-8

# Manipulación de datos
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt # Plots
import requests #Scraping
from bs4 import BeautifulSoup #Scraping
from pytrends.request import TrendReq #GoogleTrends

def get_stocks(stock_names=[]):    
        "Arma un dataframe con todos los datos de los archivos de stocks. Los archivos tienen que estar descargados en data/original/STOCK_NAME.csv"
        stock_dataframe =  pd.DataFrame()
        
        for stock in stock_names:
            df = pd.read_csv('data/original/' + stock + '.csv')

            df.drop(['Open','High','Low','Adj Close','Volume'], axis=1, inplace=True)
            df.rename(columns={'Close': stock}, inplace=True)
            df['Date'] = pd.to_datetime(df['Date'])
            df.set_index('Date', inplace=True)
    
            stock_dataframe = pd.concat([stock_dataframe, df], axis=1)

            print('Stock procesado: ' + stock)
                
        return stock_dataframe
    
#Función para hacer scraping en coinmarketcap - solo usada por la función get_crypto - 
def __scrape_crypto(criptomoneda,fecha_comienzo,fecha_fin):
        # Reformattear las fechas para hacer el request HTTP
        start = fecha_comienzo.replace('-', '')
        end = fecha_fin.replace('-', '')

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

def get_crypto(crypto_names,fecha_comienzo,fecha_fin):          
        # Crear rango de fechas
        crypto_dataframe = pd.DataFrame(pd.date_range(start=fecha_comienzo, end=fecha_fin, freq='D'),columns=['Date'])
        crypto_dataframe.set_index('Date', inplace=True)
        headers = ['Date','Open','High','Low','Close','Volume','Market Cap']

        for moneda in crypto_names:
            rows = __scrape_crypto(moneda,fecha_comienzo,fecha_fin)
            
            for row in rows:
                row[0] = pd.to_datetime(row[0]) # Dar formato de fecha 
                row[1:] = pd.to_numeric(row[1:])
                
            # Crear data frame con datos
            df = pd.DataFrame(data=rows, columns=headers)
            df = df.iloc[::-1] # Invertir, para que quede del más viejo al más nuevo
            df.drop(['Open','High','Low','Volume','Market Cap'], axis=1, inplace=True)
            df.rename(columns={'Close': moneda}, inplace=True)
            df.set_index('Date', inplace=True)            
            # Juntar el dataframe actual con el resto para formar un dataframe de marketcaps de todas las monedas
            crypto_dataframe = pd.merge(crypto_dataframe, df,  how='left', left_index=True, right_index=True)
                
        return crypto_dataframe

def get_trends(terminos,fecha_comienzo,fecha_fin):    
        "Arma un dataframe con todos los datos de los archivos de stocks"
        # Crear rango de fechas
        trends_dataframe = pd.DataFrame(pd.date_range(start=fecha_comienzo, end=fecha_fin, freq='D'),columns=['Date'])
        trends_dataframe.set_index('Date', inplace=True)
        
        #for moneda in self.crypto_names:
        for t in terminos:
            df = pd.read_csv('data/original/trends_' + t + '.csv')
          
            df.rename(columns={df.columns[0]: 'Date',df.columns[1]:'trend_' + t}, inplace=True)
            df.set_index('Date', inplace=True)       
        
            trends_dataframe = pd.merge(trends_dataframe, df,  how='left', left_index=True, right_index=True)

            print('Termino procesado: ' + t)
        
        trends_dataframe.replace('<1',0,inplace=True) 
        trends_dataframe.replace('-',np.nan,inplace=True) 
        trends_dataframe = trends_dataframe.interpolate().ffill().fillna(0) #Completar campos faltantes
        
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
def __get_trends_pytrends():
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