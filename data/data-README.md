# Data Directory

## Source
All data downloaded from **Yahoo Finance** using the `yfinance` Python library.

## Files

### Individual Country Data
| File | Ticker | Description | Observations | Period |
|------|--------|-------------|--------------|--------|
| `chile_ech_daily.csv` | ECH | iShares MSCI Chile ETF | ~2,317 | 2013-06-01 to 2022-08-31 |
| `peru_epu_daily.csv` | EPU | iShares MSCI Peru ETF | ~2,317 | 2013-06-01 to 2022-08-31 |
| `colombia_icol_daily.csv` | ICOL | iShares MSCI Colombia ETF | ~2,317 | 2013-06-01 to 2022-08-31 |

## CSV Format

Each file contains the following columns:
- `Date`: Trading date (YYYY-MM-DD)
- `Open`: Opening price (USD)
- `High`: Daily high price (USD)
- `Low`: Daily low price (USD)
- `Close`: Closing price (USD)
- `Adj Close`: Adjusted closing price (USD)
- `Volume`: Trading volume

## Download Command

To regenerate data:
```bash
python scripts/complete_analysis.py
```

## Data Quality
- No missing values in critical columns
- Dates are continuous (business days only)
- Outliers filtered (>20% daily returns removed)
- All prices in USD

## References
- Data Provider: Yahoo Finance
- Library: yfinance (https://pypi.org/project/yfinance/)
- ETF Provider: iShares (BlackRock)

## Notes
- ETFs track MSCI indices for each country
- Data represents broad market exposure for each country
- Adjusted closing prices used for analysis to account for dividends and splits