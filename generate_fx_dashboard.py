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
    "EURUSD", "USDJPY", "GBPUSD", "USDCHF", "USDCAD", "AUDUSD", "NZDUSD","USDNOK", "USDSEK", 
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
        "DEXUSEU", "DEXJPUS", "DEXUSUK", "DEXSZUS", "DEXCAUS", "DEXUSAL","DEXUSNZ", "DEXNOUS", "DEXSDUS",
        "DEXMXUS", "DEXBZUS", "DEXCHUS",
        "DEXSFUS",
        "DEXINUS","DEXKOUS", "DEXTHUS", "DEXSIUS"
    ]
    fx_labels = [
        "EURUSD", "USDJPY", "GBPUSD", "USDCHF", "USDCAD", "AUDUSD", "NZDUSD","USDNOK", "USDSEK", 
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
        r_squared = r2_score(y_test, y_pred)
        
        # Current fair value metrics
        current_actual = chart_df[currency].iloc[-1]
        current_predicted = chart_df["ypred"].iloc[-1]
        current_residual = chart_df["resids"].iloc[-1]
        
        return {
            'currency': currency,
            'current_actual': current_actual,
            'current_predicted': current_predicted,
            'current_residual': current_residual,
            'r_squared': r_squared,
            'chart_data': chart_df,
            'model': lasso,
            'valid_ratios': valid_ratios
        }
        
    except Exception as e:
        print(f"Error calculating fair value for {currency}: {e}")
        return None

# Main execution
print("Starting FX fair value analysis...")

# Get FX data using FRED
df_fx = get_fred_fx(years)

# Calculate fair values for all currencies
results = {}
fair_value_summary = []

for currency in fx_labels:
    if currency in df_fx.columns:
        print(f"Calculating fair value for {currency}...")
        result = calculate_fair_value(currency, indexed_df, df_fx, ratios)
        
        if result:
            results[currency] = result
            fair_value_summary.append({
                'Currency': currency,
                'Current_Rate': result['current_actual'],
                'Fair_Value': result['current_predicted'],
                'Residual_%': result['current_residual'],
                'R_Squared': result['r_squared']
            })

# Create summary DataFrame
summary_df = pd.DataFrame(fair_value_summary)
summary_df = summary_df.sort_values('Residual_%', ascending=True)

print(f"\nProcessed {len(summary_df)} currencies successfully")
print("\nFair Value Summary:")
print(summary_df)

# Create main dashboard
def create_dashboard():
    """Create the main FX fair value dashboard with dropdown functionality"""
    
    # Main bar chart showing current fair values
    colors = ['red' if x < 0 else 'green' for x in summary_df['Residual_%']]
    
    # Create the main figure structure
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=('FX Fair Value Overview - Current Residuals', 'Individual Currency Analysis'),
        vertical_spacing=0.15,
        row_heights=[0.6, 0.4]
    )
    
    # Main horizontal bar chart
    fig.add_trace(go.Bar(
        y=summary_df['Currency'],
        x=summary_df['Residual_%'],
        orientation='h',
        marker_color=colors,
        name='Cheap/Rich (%)',
        text=[f"{x:.1f}%" for x in summary_df['Residual_%']],
        textposition='outside',
        hovertemplate='<b>%{y}</b><br>Residual: %{x:.1f}%<br>R²: %{customdata:.2f}<extra></extra>',
        customdata=summary_df['R_Squared']
    ), row=1, col=1)
    
    # Add vertical line at zero
    fig.add_vline(x=0, line_dash="dash", line_color="gray", row=1, col=1)
    
    # Create traces for all currencies (initially hidden)
    for i, currency in enumerate(results.keys()):
        chart_data = results[currency]['chart_data']
        r_squared = results[currency]['r_squared']
        
        # Actual rate trace
        fig.add_trace(go.Scatter(
            x=chart_data.index,
            y=chart_data[currency],
            mode='lines',
            name=f'{currency} Actual',
            line=dict(color='blue', width=2),
            visible=(i == 0),  # Only first currency visible initially
            legendgroup=f'group{i}',
            showlegend=True if i == 0 else False
        ), row=2, col=1)
        
        # Fair value trace
        fig.add_trace(go.Scatter(
            x=chart_data.index,
            y=chart_data['ypred'],
            mode='lines',
            name=f'{currency} Fair Value (R²={r_squared:.2f})',
            line=dict(color='red', width=2, dash='dash'),
            visible=(i == 0),  # Only first currency visible initially
            legendgroup=f'group{i}',
            showlegend=True if i == 0 else False
        ), row=2, col=1)
    
    # Create dropdown menu
    dropdown_buttons = []
    
    for i, currency in enumerate(results.keys()):
        # Create visibility array for this currency
        visibility = [True]  # Bar chart always visible
        
        # Set visibility for currency traces
        for j, _ in enumerate(results.keys()):
            if j == i:
                visibility.extend([True, True])  # Show actual and fair value for selected currency
            else:
                visibility.extend([False, False])  # Hide other currencies
        
        dropdown_buttons.append(
            dict(
                label=currency,
                method="update",
                args=[{"visible": visibility},
                      {"title.text": f"FX Fair Value Dashboard - {currency} Selected"}]
            )
        )
    
    # Update layout with dropdown
    last_updated = datetime.datetime.now().strftime('%Y-%m-%d %H:%M UTC')
    
    fig.update_layout(
        title={
            'text': f"FX Fair Value Dashboard - {list(results.keys())[0]} Selected",
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 24}
        },
        height=800,
        template="plotly_white",
        hovermode='closest',
        showlegend=True,
        updatemenus=[
            dict(
                buttons=dropdown_buttons,
                direction="down",
                showactive=True,
                x=0.02,
                xanchor="left",
                y=0.45,
                yanchor="top",
                bgcolor="white",
                bordercolor="gray",
                borderwidth=1,
                font=dict(size=12)
            )
        ],
        annotations=[
            dict(
                text="Select Currency:",
                x=0.02, y=0.48,
                xref="paper", yref="paper",
                align="left",
                showarrow=False,
                font=dict(size=12, color="black")
            )
        ]
    )
    
    # Update axis labels
    fig.update_xaxes(title_text="Residual (%)", row=1, col=1)
    fig.update_xaxes(title_text="Date", row=2, col=1)
    fig.update_yaxes(title_text="Currency", row=1, col=1)
    fig.update_yaxes(title_text="Exchange Rate", row=2, col=1)
    
    # Add "Last Updated" annotation
    fig.add_annotation(
        text=f"Last Updated: {last_updated}",
        xref="paper", yref="paper",
        x=1, y=-0.05,
        xanchor='right', yanchor='top',
        showarrow=False,
        font=dict(size=12, color="gray")
    )
    
    return fig

# Generate dashboard
print("\nCreating dashboard...")
dashboard_fig = create_dashboard()

# Save dashboard
config = {
    'displayModeBar': False,
    'responsive': True
}

output_filename = "fx_fair_value.html"
pyo.plot(dashboard_fig, filename=output_filename, auto_open=False, config=config)

print(f"Dashboard saved as '{output_filename}'")

# Save summary data
summary_data = {
    'last_updated': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    'currencies_analyzed': len(summary_df),
    'fair_value_summary': summary_df.to_dict('records')
}

import json
with open('fx_fair_value_data.json', 'w') as f:
    json.dump(summary_data, f, indent=2)

print("Summary data saved as 'fx_fair_value_data.json'")
print(f"\nDashboard will show {len(summary_df)} currencies with fair value analysis")
