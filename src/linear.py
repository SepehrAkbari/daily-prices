import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from preprocess import preprocess

df = pd.read_csv('../data/gas_price_data.txt', delimiter='\t', header=2)
df = preprocess(df)

def linearApproximate(column: np.ndarray) -> np.ndarray:
    weekly_prices = column.copy()
    length = len(column)

    for i in range(1, length):
        if pd.isna(column[i]):
            prevIndex = i - 1

            nextIndex = i + 1
            while nextIndex < length and pd.isna(weekly_prices[nextIndex]):
                nextIndex += 1

            if nextIndex >= length:
                farPrevIndex = prevIndex - 1
                while pd.isna(weekly_prices[farPrevIndex]):
                    farPrevIndex -= 1

                if farPrevIndex >= 0 and not pd.isna(weekly_prices[prevIndex]):
                    y1, y2 = weekly_prices[farPrevIndex], weekly_prices[prevIndex]
                    x1, x2 = farPrevIndex, prevIndex

            elif nextIndex < length:
                y1, y2 = weekly_prices[prevIndex], weekly_prices[nextIndex]
                x1, x2 = prevIndex, nextIndex
            else:
                raise ValueError("Invalid state")

            slope = (y2 - y1) / (x2 - x1)

            for j in range(prevIndex + 1, nextIndex):
                column[j] = weekly_prices[prevIndex] + (slope * (j - prevIndex))
                ## print(f"index {j} estimated with {x1}, {x2} and values {y1}, {y2}")

    return column

GasPrices_Linear = df.copy()
GasPrices_Linear['Regular'] = linearApproximate(GasPrices_Linear['Regular'].values)
GasPrices_Linear['Diesel'] = linearApproximate(GasPrices_Linear['Diesel'].values)

x_regular = np.arange(len(GasPrices_Linear)).reshape(-1, 1)
y_regular = GasPrices_Linear['Regular']
x_diesel = np.arange(len(GasPrices_Linear)).reshape(-1, 1) 
y_diesel = GasPrices_Linear['Diesel']

regression_regular = LinearRegression()
regression_regular.fit(x_regular, y_regular)
regression_diesel = LinearRegression()
regression_diesel.fit(x_diesel, y_diesel)

y_predict_regular = regression_regular.predict(x_regular)
y_predict_diesel = regression_diesel.predict(x_diesel)


def linearNoisyApproximate(column: np.ndarray, mean, std) -> np.ndarray:
    weekly_prices = column.copy()
    length = len(column) 

    noise = np.random.normal(mean, std, size = column.shape)

    for i in range(1, length):
        if pd.isna(column[i]):
            prevIndex = i - 1

            nextIndex = i + 1
            while nextIndex < length and pd.isna(weekly_prices[nextIndex]):
                nextIndex += 1

            if nextIndex >= length:
                farPrevIndex = prevIndex - 1
                while pd.isna(weekly_prices[farPrevIndex]):
                    farPrevIndex -= 1
                
                if farPrevIndex >= 0 and not pd.isna(weekly_prices[prevIndex]):
                    y1, y2 = weekly_prices[farPrevIndex], weekly_prices[prevIndex]
                    x1, x2 = farPrevIndex, prevIndex

            elif nextIndex < length:
                y1, y2 = weekly_prices[prevIndex], weekly_prices[nextIndex]
                x1, x2 = prevIndex, nextIndex
            else:
                raise ValueError("Invalid state")

            slope = (y2 - y1) / (x2 - x1)

            for j in range(prevIndex + 1, nextIndex):
                column[j] = (weekly_prices[prevIndex] + (slope * (j - prevIndex))) + noise[j] # Adding noise to each estimated value

    return column

GasPrices_Linear_Noisy = df.copy()
mean = 0
standard_deviation = 0.1
GasPrices_Linear_Noisy['Regular'] = linearNoisyApproximate(GasPrices_Linear_Noisy['Regular'].values, mean, standard_deviation)
GasPrices_Linear_Noisy['Diesel'] = linearNoisyApproximate(GasPrices_Linear_Noisy['Diesel'].values, mean, standard_deviation)

x_regular = np.arange(len(GasPrices_Linear_Noisy)).reshape(-1, 1)
y_regular = GasPrices_Linear_Noisy['Regular']
x_diesel = np.arange(len(GasPrices_Linear_Noisy)).reshape(-1, 1) 
y_diesel = GasPrices_Linear_Noisy['Diesel']

regression_regular = LinearRegression()
regression_regular.fit(x_regular, y_regular)
regression_diesel = LinearRegression()
regression_diesel.fit(x_diesel, y_diesel)

y_predict_regular = regression_regular.predict(x_regular)
y_predict_diesel = regression_diesel.predict(x_diesel)