# FX Fair Value Dashboard

Interactive dashboard analyzing FX fair values using equity sector performance as predictors.

## Live Dashboard
View the dashboard at: [https://datavizhonduran.github.io/fx_fair_value_dashboard](https://datavizhonduran.github.io/fx_fair_value_dashboard)

## Features
- Fair value models for 25+ major currencies
- Visual cheap/rich indicators with horizontal bar chart
- Individual currency analysis with time series
- Daily automatic updates via GitHub Actions
- Lasso regression models using equity sector ETF ratios

## Methodology
- Uses 10+ equity sector ETF performance ratios as predictors
- Lasso regression to predict FX movements
- Residuals show deviation from fair value (cheap/rich signals)
- Models trained on 10 years of historical data

## Data Sources
- FX data: Alpha Vantage API
- Equity data: Stooq via pandas-datareader
- Daily updates at 7 AM EST

## Currencies Covered
EUR, AUD, CAD, GBP, JPY, SEK, NOK, NZD, CHF, MXN, CLP, BRL, COP, PEN, KRW, IDR, INR, THB, PHP, SGD, PLN, HUF, CZK, ZAR, TRY
