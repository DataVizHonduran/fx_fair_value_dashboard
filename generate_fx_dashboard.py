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
    "USDMXN", "USDBRL",
    "USDZAR", 
    "USDINR", "USDKRW", "USDTHB", "USDSGD", "USDCNH"
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
        "DEXMXUS", "DEXBZUS", 
        "DEXSFUS",
        "DEXINUS", "DEXKOUS", "DEXTHUS", "DEXSIUS", "DEXCHUS",
    ]
    fx_labels = [
        "USDNOK", "USDSEK", 
        "USDMXN", "USDBRL", 
        "USDZAR", 
        "USDINR", "USDKRW", "USDTHB", "USDSGD", "USDCNH"
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
        current_predicted = chart_df["ypred"].iloc[-1]
        current_residual = chart_df["resids"].iloc[-1]
        
        return {
            'currency': currency,
            'current_actual': current_actual,
            'current_predicted': current_predicted,
            'current_residual': current_residual,
            'r_squared': r_squared,
            'mse': mse,
            'chart_data': chart_df,
            'model': lasso,
            'valid_ratios': valid_ratios,
            'coefficients': non_zero_coefficients,
            'selected_vars': selected_vars
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

def create_main_dashboard():
    """Create the main overview dashboard with only the bar chart"""
    
    # Main bar chart showing current fair values
    colors = ['red' if x < 0 else 'green' for x in summary_df['Residual_%']]
    
    fig = go.Figure()
    
    # Create clickable bar chart with links
    hover_text = []
    link_text = []
    
    for i, row in summary_df.iterrows():
        currency = row['Currency']
        residual = row['Residual_%']
        r_squared = row['R_Squared']
        hover_text.append(f"<b>{currency}</b><br>Residual: {residual:.1f}%<br>R²: {r_squared:.2f}<br><i>Click for detailed analysis</i>")
        link_text.append(f"{currency.lower()}_analysis.html")
    
    # Main horizontal bar chart
    fig.add_trace(go.Bar(
        y=summary_df['Currency'],
        x=summary_df['Residual_%'],
        orientation='h',
        marker_color=colors,
        name='Cheap/Rich (%)',
        text=[f"{x:.1f}%" for x in summary_df['Residual_%']],
        textposition='outside',
        hovertemplate='%{customdata}<extra></extra>',
        customdata=hover_text
    ))
    
    # Add vertical line at zero
    fig.add_vline(x=0, line_dash="dash", line_color="gray")
    
    # Add currency links as annotations
    currency_links_html = "<br>".join([
        f'<a href="{currency.lower()}_analysis.html" style="text-decoration:none; color:#1f77b4;">📊 {currency} Detailed Analysis</a>' 
        for currency in summary_df['Currency']
    ])
    
    last_updated = datetime.datetime.now().strftime('%Y-%m-%d %H:%M UTC')
    
    fig.update_layout(
        title={
            'text': "FX Fair Value Overview",
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 28}
        },
        xaxis_title="Residual (%)",
        yaxis_title="Currency Pair",
        height=600,
        template="plotly_white",
        hovermode='closest',
        showlegend=False,
        margin=dict(r=200)
    )
    
    # Add navigation links
    fig.add_annotation(
        text="<b>Individual Currency Analysis:</b><br>" + currency_links_html,
        xref="paper", yref="paper",
        x=1.02, y=1,
        xanchor='left', yanchor='top',
        showarrow=False,
        font=dict(size=11),
        align="left",
        bgcolor="rgba(255,255,255,0.8)",
        bordercolor="gray",
        borderwidth=1
    )
    
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

def create_individual_currency_page(currency, result):
    """Create individual currency analysis page"""
    
    chart_data = result['chart_data']
    
    # Add white space at right side of charts
    new_dates = pd.date_range(start=chart_data.index[-1], periods=int(len(chart_data.index) * .05), freq='D')
    empty_rows = pd.DataFrame(index=new_dates)
    chart_data = pd.concat([chart_data, empty_rows])
    
    # Create two-panel chart
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=(f'{currency} vs Fair Value Model', 'Residual (% Deviation)'),
        horizontal_spacing=0.1
    )
    
    # Left panel: Actual vs Model
    fig.add_scatter(
        x=chart_data.index, 
        y=chart_data[currency], 
        mode='lines', 
        name=currency,
        line=dict(color='blue', width=2),
        row=1, col=1
    )
    
    fig.add_scatter(
        x=chart_data.index, 
        y=chart_data["ypred"], 
        mode='lines', 
        name="Fair Value Model",
        line=dict(color='red', width=2, dash='dash'),
        row=1, col=1
    )
    
    # Right panel: Residuals
    fig.add_scatter(
        x=chart_data.index, 
        y=chart_data["resids"], 
        mode='lines', 
        name="Residual (%)",
        line=dict(color='green', width=2),
        row=1, col=2
    )
    
    # Add zero line to residuals
    fig.add_hline(y=0, line_dash="dash", line_color="gray", row=1, col=2)
    
    # Model info
    r_squared = result['r_squared']
    mse = result['mse']
    current_residual = result['current_residual']
    
    # Update layout
    last_updated = datetime.datetime.now().strftime('%Y-%m-%d %H:%M UTC')
    
    fig.update_layout(
        title={
            'text': f"{currency} Fair Value Analysis - R²: {r_squared:.3f}",
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 20}
        },
        width=1200,
        height=600,
        template="plotly_white",
        showlegend=True
    )
    
    # Add model details
    model_info = f"""
    <b>Model Performance:</b><br>
    R-squared: {r_squared:.3f}<br>
    MSE: {mse:.2f}<br>
    Current Residual: {current_residual:.1f}%<br>
    <br>
    <b>Key Predictors:</b><br>
    {', '.join(result['selected_vars'][:5])}
    """
    
    fig.add_annotation(
        text=model_info,
        xref="paper", yref="paper",
        x=0.02, y=0.98,
        xanchor='left', yanchor='top',
        showarrow=False,
        font=dict(size=10),
        align="left",
        bgcolor="rgba(255,255,255,0.8)",
        bordercolor="gray",
        borderwidth=1
    )
    
    # Back to main link
    fig.add_annotation(
        text='<a href="fx_fair_value.html" style="color:#1f77b4;">← Back to Overview</a>',
        xref="paper", yref="paper",
        x=1, y=1.02,
        xanchor='right', yanchor='bottom',
        showarrow=False,
        font=dict(size=12)
    )
    
    # Last updated
    fig.add_annotation(
        text=f"Last Updated: {last_updated}",
        xref="paper", yref="paper",
        x=1, y=-0.05,
        xanchor='right', yanchor='top',
        showarrow=False,
        font=dict(size=12, color="gray")
    )
    
    return fig

# Generate all dashboard files
print("\nCreating dashboards...")

# Main overview dashboard
main_fig = create_main_dashboard()
config = {'displayModeBar': False, 'responsive': True}

# Save main dashboard
pyo.plot(main_fig, filename="fx_fair_value.html", auto_open=False, config=config)
print("Main dashboard saved as 'fx_fair_value.html'")

# Create individual currency pages
for currency, result in results.items():
    individual_fig = create_individual_currency_page(currency, result)
    filename = f"{currency.lower()}_analysis.html"
    pyo.plot(individual_fig, filename=filename, auto_open=False, config=config)
    print(f"Individual analysis saved as '{filename}'")

# Save summary data
summary_data = {
    'last_updated': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    'currencies_analyzed': len(summary_df),
    'fair_value_summary': summary_df.to_dict('records'),
    'individual_pages': [f"{currency.lower()}_analysis.html" for currency in results.keys()]
}

import json
with open('fx_fair_value_data.json', 'w') as f:
    json.dump(summary_data, f, indent=2)

print(f"\nGenerated {len(results)} individual currency analysis pages")
print("Summary data saved as 'fx_fair_value_data.json'")
print(f"\nMain dashboard: fx_fair_value.html")
print("Individual pages:", [f"{c.lower()}_analysis.html" for c in results.keys()])
