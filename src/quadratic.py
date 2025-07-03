import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from preprocess import preprocess

df = pd.read_csv('../data/gas_price_data.txt', delimiter='\t', header=2)
df = preprocess(df)

def quadraticApproximate(column: np.ndarray) -> np.ndarray:
    weekly_prices = column.copy()
    length = len(column)

    for i in range(1, length):
        if pd.isna(column[i]):
            prevIndex = i - 1

            nextIndex = i + 1
            while nextIndex < length and pd.isna(weekly_prices[nextIndex]):
                nextIndex += 1

            farNextIndex = nextIndex + 1
            while farNextIndex < length and pd.isna(weekly_prices[farNextIndex]):
                farNextIndex += 1
    
            farPrevIndex = prevIndex - 1
            while pd.isna(weekly_prices[farPrevIndex]):
                farPrevIndex -= 1
          
            farFarPrevIndex = farPrevIndex - 1
            while pd.isna(weekly_prices[farFarPrevIndex]):
                farFarPrevIndex -= 1
           
            if nextIndex >= length:
                y1, y2, y3 = weekly_prices[prevIndex], weekly_prices[farPrevIndex], weekly_prices[farFarPrevIndex]
                x1, x2, x3 = prevIndex, farPrevIndex, farFarPrevIndex
            elif farNextIndex >= length and nextIndex < length:
                y1, y2, y3 = weekly_prices[prevIndex], weekly_prices[nextIndex], weekly_prices[farPrevIndex]
                x1, x2, x3 = prevIndex, nextIndex, farPrevIndex
            elif farNextIndex < length:
                y1, y2, y3 = weekly_prices[nextIndex], weekly_prices[prevIndex], weekly_prices[farNextIndex]
                x1, x2, x3 = nextIndex, prevIndex, farNextIndex
            else:
                raise ValueError("Invalid State")

            slope1 = (y2 - y1) / (x2 - x1)
            slope2 = (y3 - y2) / (x3 - x2)
            delta_x = (x3 - x1) / 2 
            f_prime = slope1  
            f_double_prime = (slope2 - slope1) / delta_x 
            
            for j in range(prevIndex + 1, nextIndex):
                if pd.isna(column[j]):    
                    column[j] = y1 + f_prime * (j - x1) + 0.5 * f_double_prime * (j - x1) ** 2
    return column

GasPrices_Quadratic = df.copy()

GasPrices_Quadratic['Regular'] = quadraticApproximate(GasPrices_Quadratic['Regular'].values)
GasPrices_Quadratic['Diesel'] = quadraticApproximate(GasPrices_Quadratic['Diesel'].values)

x_regular = np.arange(len(GasPrices_Quadratic)).reshape(-1, 1)
y_regular = GasPrices_Quadratic['Regular']
x_diesel = np.arange(len(GasPrices_Quadratic)).reshape(-1, 1) 
y_diesel = GasPrices_Quadratic['Diesel']

regression_regular = LinearRegression()
regression_regular.fit(x_regular, y_regular)
regression_diesel = LinearRegression()
regression_diesel.fit(x_diesel, y_diesel)

y_predict_regular = regression_regular.predict(x_regular)
y_predict_diesel = regression_diesel.predict(x_diesel)


def quadraticNoisyApproximate(column: np.ndarray, mean, std) -> np.ndarray:
    weekly_prices = column.copy() 
    length = len(column)

    noise = np.random.normal(mean, std, size=column.shape)

    for i in range(1, length):  
        if pd.isna(column[i]):
            prevIndex = i - 1
            
            nextIndex = i + 1
            while nextIndex < length and pd.isna(weekly_prices[nextIndex]):
                nextIndex += 1
            
            farNextIndex = nextIndex + 1
            while farNextIndex < length and pd.isna(weekly_prices[farNextIndex]):
                farNextIndex += 1
            
            farPrevIndex = prevIndex - 1
            while pd.isna(weekly_prices[farPrevIndex]):
                farPrevIndex -= 1
            
            farFarPrevIndex = farPrevIndex - 1
            while pd.isna(weekly_prices[farFarPrevIndex]):
                farFarPrevIndex -= 1

            if nextIndex >= length: 
                y1, y2, y3 = weekly_prices[prevIndex], weekly_prices[farPrevIndex], weekly_prices[farFarPrevIndex]
                x1, x2, x3 = prevIndex, farPrevIndex, farFarPrevIndex
            elif farNextIndex >= length and nextIndex < length:    
                y1, y2, y3 = weekly_prices[prevIndex], weekly_prices[nextIndex], weekly_prices[farPrevIndex]
                x1, x2, x3 = prevIndex, nextIndex, farPrevIndex          
            elif farNextIndex < length:
                y1, y2, y3 = weekly_prices[nextIndex], weekly_prices[prevIndex], weekly_prices[farNextIndex]
                x1, x2, x3 = nextIndex, prevIndex, farNextIndex
            else:
                raise ValueError("Invalid State")
            
            slope1 = (y2 - y1) / (x2 - x1)
            slope2 = (y3 - y2) / (x3 - x2)
            delta_x = (x3 - x1) / 2
            f_prime = slope1
            f_double_prime = (slope2 - slope1) / delta_x

            for j in range(prevIndex + 1, nextIndex):
                if pd.isna(column[j]): 
                    column[j] = (y1 + f_prime * (j - x1) + 0.5 * f_double_prime * (j - x1) ** 2) + noise[j]

    return column

GasPrices_Quadratic_Noisy = df.copy()
mean = 0
standard_deviation = 0.1
GasPrices_Quadratic_Noisy['Regular'] = quadraticNoisyApproximate(GasPrices_Quadratic_Noisy['Regular'].values, mean, standard_deviation)
GasPrices_Quadratic_Noisy['Diesel'] = quadraticNoisyApproximate(GasPrices_Quadratic_Noisy['Diesel'].values, mean, standard_deviation)

x_regular = np.arange(len(GasPrices_Quadratic_Noisy)).reshape(-1, 1)
y_regular = GasPrices_Quadratic_Noisy['Regular']
x_diesel = np.arange(len(GasPrices_Quadratic_Noisy)).reshape(-1, 1) 
y_diesel = GasPrices_Quadratic_Noisy['Diesel']

regression_regular = LinearRegression()
regression_regular.fit(x_regular, y_regular)
regression_diesel = LinearRegression()
regression_diesel.fit(x_diesel, y_diesel)

y_predict_regular = regression_regular.predict(x_regular)
y_predict_diesel = regression_diesel.predict(x_diesel)