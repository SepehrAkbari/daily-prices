import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from preprocess import preprocess

df = pd.read_csv('../data/gas_price_data.txt', delimiter='\t', header=2)
df = preprocess(df)

def hotDeckApproximate(column: np.ndarray) -> np.ndarray:
    weekly_prices = column.copy() 
    original_prices = weekly_prices[~pd.isna(weekly_prices)]
    
    for i in range(1, len(column)):
        if pd.isna(column[i]):  
            column[i] = np.random.choice(original_prices)

    return column

GasPrices_HotDeck = df.copy()

GasPrices_HotDeck['Regular'] = hotDeckApproximate(GasPrices_HotDeck['Regular'].values)
GasPrices_HotDeck['Diesel'] = hotDeckApproximate(GasPrices_HotDeck['Diesel'].values)

x_regular = np.arange(len(GasPrices_HotDeck)).reshape(-1, 1)
y_regular = GasPrices_HotDeck['Regular']
x_diesel = np.arange(len(GasPrices_HotDeck)).reshape(-1, 1) 
y_diesel = GasPrices_HotDeck['Diesel']

regression_regular = LinearRegression()
regression_regular.fit(x_regular, y_regular)
regression_diesel = LinearRegression()
regression_diesel.fit(x_diesel, y_diesel)

y_predict_regular = regression_regular.predict(x_regular)
y_predict_diesel = regression_diesel.predict(x_diesel)