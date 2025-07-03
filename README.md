# Imputing Daily Gas Prices

This project explores how we can impute daily gas prices using weekly gas prices. This can be efficient for forecasting, reducing the amount of data that needs to be stored, and reducing the amount of data that needs to be processed. The goal of the project is to explore three different methods of imputation: a simple linear interpolation, a more complex polynomial interpolation using Taylor series, and a hot deck imputation.

## Data

The data used in this project is from the U.S. Energy Information Administration (EIA) and contains weekly gas prices from Aug 29th 2022 to August 26th 2024. The data contains prices for both regular and diesel gas. The data can be downloaded as text file in the [data](/data/) folder.

## Usage

To run each method, you can first clone this repository:

```bash
git clone https://github.com/SepehrAkbari/daily-prices.git
cd daily-prices
```

Then, you can run each method using the following commands:

```bash
cd src
python linear.py
python quadratic.py
python hotdeck.py
cd ../
```
To see the results and our analysis, read through the notebook in the [notebook](/notebook/) folder.

```bash
cd notebooks
jupyter daily_prices.ipynb
```
You can also check out the results visually in the [results](/results/) folder.

## Contributing

To contribute to this project, you can fork this repository and create pull requests. You can also open an issue if you find a bug or wish to make a suggestion.

## License

This project is licensed under the [GNU General Public License (GPL)](LICENSE).
