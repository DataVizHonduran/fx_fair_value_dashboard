import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.offline as pyo
from plotly.subplots import make_subplots
from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import datetime
from datetime import date
from pandas_datareader import data

# Configuration
exclude_recent = True
years = 10

# Currency symbols from FRED data
fx_labels = [
    "USDNOK", "USDSEK", 
    "USDMXN", "USDBRL", "USDCLP", 
    "USDZAR", 
    "USDINR", "USDKRW", "USDTHB", "USDSGD"
]

# Equity sector ETFs as predictors
equity_sector_etfs = ["XLE", "XLU", "XLI", "XLV", "XLY", "XLP", "XLK", "XLF", "XLB", "SPY", "XRT", "XLRE", "XLC"]
equity_sector_etfs = [ticker + ".US" for ticker in equity_sector_etfs]

equity_sector_focus = [
    "Energy", "Utilities", "Industrials", "Healthcare", "Consumer Discretionary",
    "Consumer Staples", "Technology", "Financials", "Materials", "SP500",
    "Retail", "Real Estate", "Communications"
]

if exclude_recent:
    equity_sector_etfs = equity_sector_etfs[:-3]
    equity_sector_focus = equity_sector_focus[:-3]

# Date range
start_date = datetime.datetime.now() - datetime.timedelta(days=365*years)
end_date = date.today()

print("Fetching equity sector data...")
# Get equity sector data
etf_data = data.DataReader(equity_sector_etfs, 'stooq', start_date, end_date)
final_df = etf_data['Close'].sort_index(ascending=True)
indexed_df = final_df.apply(lambda col: col / col.dropna().iloc[0] * 100)
indexed_df = indexed_df.bfill()

# Create sector ratios vs SPY
for sector in equity_sector_etfs:
    indexed_df[f"{sector} / SPY"] = indexed_df[sector] / indexed_df["SPY.US"] * 100

ratios = [f"{sector} / SPY" for sector in equity_sector_etfs[:-1]]

def get_fred_fx(years=10):
    import datetime
    from pandas_datareader import data
    import pandas as pd
    
    datalist = [
        "DEXNOUS", "DEXSDUS",
        "DEXMXUS", "DEXBZUS", "DEXCHUS",
        "DEXSFUS",
        "DEXINUS","DEXKOUS", "DEXTHUS", "DEXSIUS"
    ]
    fx_labels = [
        "USDNOK", "USDSEK", 
        "USDMXN", "USDBRL", "USDCLP", 
        "USDZAR", 
        "USDINR", "USDKRW", "USDTHB", "USDSGD"
    ]
    end_date = datetime.date.today()
    print(f"Fetching FX data from FRED for date range ending: {end_date}")
    start_date = end_date - datetime.timedelta(days=365*years)
    df = data.DataReader(datalist, 'fred', start_date, end_date)
    df.columns = fx_labels
    df = df.apply(pd.to_numeric, errors='coerce')
    df.index = pd.to_datetime(df.index, errors='coerce')
    df = df.bfill()
    print(f"Successfully loaded FX data: {df.shape}")
    print(df.tail())
    return df

def calculate_fair_value(currency, indexed_df, fx_df, ratios):
    """Calculate fair value for a single currency"""
    try:
        # Merge equity and FX data
        chart_df = indexed_df[ratios].merge(fx_df[currency], left_index=True, right_index=True, how="inner")
        chart_df = chart_df.bfill()
        chart_df = chart_df.loc[:, chart_df.isnull().sum() <= 10]
        
        # Prepare data for model
        y = chart_df[currency]
        valid_ratios = [col for col in ratios if col in chart_df.columns]
        X = chart_df[valid_ratios].bfill()
        
        if len(X) < 100:  # Need sufficient data
            return None
            
        # Train model
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        lasso = Lasso(max_iter=10000, alpha=0.9)
        lasso.fit(X_train, y_train)
        
        # Calculate predictions and residuals
        chart_df["ypred"] = lasso.predict(chart_df[valid_ratios].bfill())
        chart_df["resids"] = (chart_df[currency] - chart_df["ypred"]) / chart_df[currency] * 100
        
        # Model metrics
        y_pred = lasso.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r_squared = r2_score(y_test, y_pred)
        
        # Get model coefficients info
        coefficients = lasso.coef_
        non_zero_coefficients = [round(x,3) for x in coefficients if x != 0]
        selected_vars = [valid_ratios[i] for i in range(len(coefficients)) if coefficients[i] != 0]
        
        # Current fair value metrics
        current_actual = chart_df[currency].iloc[-1]
        current_predicted = chart_df["
