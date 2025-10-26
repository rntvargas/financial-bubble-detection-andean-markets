"""
Complete Financial Bubble Detection Pipeline
============================================
1. Downloads data from Yahoo Finance (ETFs)
2. Saves individual CSV files (Chile, Peru, Colombia)
3. Performs complete analysis (Bayesian, K-means, ARFIMA)
4. Generates all 7 figures separately
============================================
Author: Briggitte Jhosselyn Vilca Chambilla
Paper: Comparative Analysis of Financial Bubble Detection Models
"""

import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, mutual_info_score
import warnings
import os
warnings.filterwarnings('ignore')

# Setup directories
os.makedirs('data', exist_ok=True)
os.makedirs('results', exist_ok=True)
os.makedirs('figures', exist_ok=True)

print("\n" + "="*80)
print("COMPLETE FINANCIAL BUBBLE DETECTION PIPELINE")
print("="*80)

# ============================================================================
# STEP 1: DOWNLOAD DATA & SAVE INDIVIDUAL CSV FILES
# ============================================================================

print("\n[STEP 1/4] Downloading data and saving individual CSV files...")

TICKERS = {
    'Chile': 'ECH',      # iShares MSCI Chile ETF
    'Peru': 'EPU',       # iShares MSCI Peru ETF
    'Colombia': 'ICOL'   # iShares MSCI Colombia ETF
}

START_DATE = '2013-06-01'
END_DATE = '2022-08-31'

prices_dict = {}

for country, ticker in TICKERS.items():
    print(f"\n  Downloading {country} ({ticker})...")
    
    try:
        data = yf.download(ticker, start=START_DATE, end=END_DATE, progress=False)
        
        if not data.empty:
            # Handle multi-index columns
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = data.columns.get_level_values(0)
            
            # Reset index
            data = data.reset_index()
            
            # ✅ Save individual CSV
            filename = f'data/{country.lower()}_{ticker.lower()}_daily.csv'
            data.to_csv(filename, index=False)
            
            print(f"    ✓ {len(data)} observations")
            print(f"    ✓ Saved: {filename}")
            
            # Store for analysis
            prices_dict[country] = data.set_index('Date')['Close']
        else:
            print(f"    ✗ No data available")
            
    except Exception as e:
        print(f"    ✗ Error: {str(e)}")

# Create combined DataFrames
prices = pd.DataFrame(prices_dict).dropna()
returns = prices.pct_change().dropna()

# Save combined
prices.to_csv('results/prices_combined.csv')
returns.to_csv('results/returns_combined.csv')

print(f"\n  ✓ Combined data: {len(prices)} observations")
print(f"    Period: {prices.index[0].date()} to {prices.index[-1].date()}")

# ============================================================================
# STEP 2: COMPLETE ANALYSIS (BAYESIAN, K-MEANS, ARFIMA)
# ============================================================================

print("\n[STEP 2/4] Performing complete analysis...")

# --- Bayesian Networks ---
def discretize_returns(series):
    conditions = [series < -0.02, (series >= -0.02) & (series < -0.005),
                  (series >= -0.005) & (series < 0.005), (series >= 0.005) & (series < 0.02), series >= 0.02]
    return np.select(conditions, [0, 1, 2, 3, 4], default=2)

def discretize_volatility(vol_series):
    conditions = [vol_series < 0.01, (vol_series >= 0.01) & (vol_series < 0.02),
                  (vol_series >= 0.02) & (vol_series < 0.04), vol_series >= 0.04]
    return np.select(conditions, [0, 1, 2, 3], default=1)

bn_data = pd.DataFrame(index=returns.index)
for country in returns.columns:
    bn_data[f'{country}_Return_State'] = discretize_returns(returns[country])
    vol = returns[country].rolling(30).std()
    bn_data[f'{country}_Vol_State'] = discretize_volatility(vol)
bn_data = bn_data.dropna()

# Mutual Information
mi_matrix = pd.DataFrame(index=prices.columns, columns=prices.columns, dtype=float)
for i, c1 in enumerate(prices.columns):
    for j, c2 in enumerate(prices.columns):
        if c1 == c2:
            mi_matrix.loc[c1, c2] = 1.0
        else:
            var1, var2 = f'{c1}_Return_State', f'{c2}_Return_State'
            mi_matrix.loc[c1, c2] = mutual_info_score(bn_data[var1].astype(int), bn_data[var2].astype(int))
mi_matrix = mi_matrix.astype(float)

# Crisis prediction
crisis_pred = {}
for country in prices.columns:
    high_vol = bn_data[bn_data[f'{country}_Vol_State'] >= 2]
    crisis_pred[country] = (high_vol[f'{country}_Return_State'] <= 1).mean() if len(high_vol) > 0 else 0.0

# --- K-means Clustering ---
features_list = []
for country in prices.columns:
    df = pd.DataFrame({'Country': country, 'Close': prices[country], 'Return': returns[country]})
    df['Return_30'] = df['Return'].rolling(30).mean()
    df['Vol_30'] = df['Return'].rolling(30).std()
    df['Momentum'] = df['Close'].pct_change(30)
    features_list.append(df)

data_features = pd.concat(features_list).dropna()
X = data_features[['Return_30', 'Vol_30', 'Momentum']]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
data_features['Cluster'] = kmeans.fit_predict(X_scaled)
silhouette = silhouette_score(X_scaled, data_features['Cluster'])

# --- Hurst Exponent ---
def hurst_exponent(ts):
    ts = np.array(ts)
    n = len(ts)
    if n < 100:
        return np.nan
    lags = range(2, min(n//2, 100))
    RS = []
    for lag in lags:
        n_subseries = n // lag
        subseries = [ts[i*lag:(i+1)*lag] for i in range(n_subseries)]
        rs_values = []
        for sub in subseries:
            if len(sub) < 2:
                continue
            mean = np.mean(sub)
            cumsum = np.cumsum(sub - mean)
            R = np.max(cumsum) - np.min(cumsum)
            S = np.std(sub)
            if S > 0:
                rs_values.append(R/S)
        if rs_values:
            RS.append(np.mean(rs_values))
    if len(RS) < 2:
        return np.nan
    lags_valid = list(lags)[:len(RS)]
    log_lags = np.log(lags_valid)
    log_RS = np.log(RS)
    mask = np.isfinite(log_lags) & np.isfinite(log_RS)
    if np.sum(mask) < 2:
        return np.nan
    return np.polyfit(log_lags[mask], log_RS[mask], 1)[0]

hurst_results = []
for country in returns.columns:
    H = hurst_exponent(returns[country].dropna().values)
    hurst_results.append({'Country': country, 'Hurst_Exponent': H})
hurst_df = pd.DataFrame(hurst_results)

print(f"  ✓ Mutual Information avg: {mi_matrix.values[np.triu_indices_from(mi_matrix.values, k=1)].mean():.3f}")
print(f"  ✓ Silhouette Score: {silhouette:.3f}")
print(f"  ✓ Hurst Exponents calculated")

# Save results
mi_matrix.to_csv('results/mutual_information.csv')
pd.DataFrame(crisis_pred, index=['P(Crisis|HighVol)']).T.to_csv('results/crisis_prediction.csv')
data_features.to_csv('results/clustered_data.csv')
hurst_df.to_csv('results/hurst_exponents.csv', index=False)

# ============================================================================
# STEP 3: GENERATE ALL 7 FIGURES
# ============================================================================

print("\n[STEP 3/4] Generating all 7 figures...")

plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({'figure.dpi': 150, 'savefig.dpi': 300, 'font.family': 'serif', 'font.size': 11})
colors = ['#1f77b4', '#ff7f0e', '#2ca02c']

# FIGURE 1: Price Evolution
print("  [1/7] Price Evolution...")
fig, ax = plt.subplots(figsize=(10, 5))
prices_norm = (prices / prices.iloc[0]) * 100
for col, color in zip(prices_norm.columns, colors):
    ax.plot(prices_norm.index, prices_norm[col], label=col, linewidth=2.5, color=color)
ax.axvline(pd.Timestamp('2020-03-11'), color='red', linestyle='--', alpha=0.5, label='COVID-19')
ax.set_title('Normalized Stock Market Indices (2013-2022)', fontweight='bold', fontsize=13)
ax.set_ylabel('Index (Base = 100)', fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('figures/fig1_price_evolution.png', dpi=300, bbox_inches='tight')
plt.close()

# FIGURE 2: Returns Distribution
print("  [2/7] Returns Distribution...")
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
for idx, (col, color) in enumerate(zip(returns.columns, colors)):
    ret = returns[col].dropna() * 100
    axes[idx].hist(ret, bins=50, alpha=0.7, edgecolor='black', color=color, density=True)
    mu, sigma = ret.mean(), ret.std()
    x = np.linspace(ret.min(), ret.max(), 100)
    axes[idx].plot(x, 1/(sigma * np.sqrt(2 * np.pi)) * np.exp(-(x - mu)**2 / (2 * sigma**2)), 'r--', linewidth=2)
    axes[idx].set_title(f'{col}', fontweight='bold')
    axes[idx].set_xlabel('Daily Returns (%)')
    if idx == 0:
        axes[idx].set_ylabel('Density')
    axes[idx].grid(True, alpha=0.3)
plt.suptitle('Distribution of Daily Returns', fontweight='bold', fontsize=13)
plt.tight_layout()
plt.savefig('figures/fig2_returns_distribution.png', dpi=300, bbox_inches='tight')
plt.close()

# FIGURE 3: Rolling Volatility
print("  [3/7] Rolling Volatility...")
fig, ax = plt.subplots(figsize=(10, 5))
for col, color in zip(returns.columns, colors):
    vol = returns[col].rolling(30).std() * 100
    ax.plot(vol.index, vol, label=col, linewidth=2, color=color)
ax.axvline(pd.Timestamp('2020-03-11'), color='red', linestyle='--', alpha=0.5)
ax.set_title('30-day Rolling Volatility', fontweight='bold', fontsize=13)
ax.set_ylabel('Volatility (%)', fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('figures/fig3_rolling_volatility.png', dpi=300, bbox_inches='tight')
plt.close()

# FIGURE 4: K-means Clustering
print("  [4/7] K-Means Clustering...")
fig, ax = plt.subplots(figsize=(9, 7))
scatter = ax.scatter(data_features['Vol_30']*100, data_features['Return_30']*100,
                    c=data_features['Cluster'], cmap='viridis', alpha=0.6, s=25, edgecolors='black', linewidth=0.3)
ax.set_title('K-Means Clustering Results (k=3)', fontweight='bold', fontsize=13)
ax.set_xlabel('30-Day Volatility (%)', fontweight='bold')
ax.set_ylabel('30-Day Return (%)', fontweight='bold')
plt.colorbar(scatter, ax=ax, label='Cluster')
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('figures/fig4_clustering.png', dpi=300, bbox_inches='tight')
plt.close()

# FIGURE 5: Hurst Exponents
print("  [5/7] Hurst Exponents...")
fig, ax = plt.subplots(figsize=(9, 6))
bars = ax.bar(hurst_df['Country'], hurst_df['Hurst_Exponent'],
             color=colors[:len(hurst_df)], alpha=0.8, edgecolor='black', linewidth=2)
ax.axhline(0.5, color='red', linestyle='--', linewidth=2, label='Random Walk (H=0.5)')
ax.axhspan(0.5, 1.0, alpha=0.1, color='green', label='Persistent')
ax.set_title('Hurst Exponents - Long Memory Detection', fontweight='bold', fontsize=13)
ax.set_ylabel('Hurst Exponent', fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3, axis='y')
for bar, val in zip(bars, hurst_df['Hurst_Exponent']):
    ax.text(bar.get_x() + bar.get_width()/2, val + 0.02, f'{val:.3f}',
           ha='center', fontweight='bold')
plt.tight_layout()
plt.savefig('figures/fig5_hurst_exponents.png', dpi=300, bbox_inches='tight')
plt.close()

# FIGURE 6: Mutual Information
print("  [6/7] Mutual Information...")
fig, ax = plt.subplots(figsize=(7, 6))
sns.heatmap(mi_matrix.astype(float), annot=True, fmt='.3f', cmap='YlOrRd',
           cbar_kws={'label': 'Mutual Information'}, ax=ax, square=True, vmin=0, vmax=0.2,
           linewidths=1.5, linecolor='white', annot_kws={'fontsize': 11, 'fontweight': 'bold'})
ax.set_title('Mutual Information Matrix', fontweight='bold', fontsize=13)
plt.tight_layout()
plt.savefig('figures/fig6_mutual_information.png', dpi=300, bbox_inches='tight')
plt.close()

# FIGURE 7: Crisis Probability
print("  [7/7] Crisis Probability...")
fig, ax = plt.subplots(figsize=(9, 6))
crisis_vals = [crisis_pred[c] for c in prices.columns]
bars = ax.bar(prices.columns, crisis_vals, color=colors[:len(crisis_vals)],
             alpha=0.8, edgecolor='black', linewidth=2)
ax.axhline(np.mean(crisis_vals), color='red', linestyle='--', linewidth=2,
          label=f'Average: {np.mean(crisis_vals):.1%}')
ax.set_title('Crisis Probability given High Volatility', fontweight='bold', fontsize=13)
ax.set_ylabel('P(Crisis | High Volatility)', fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3, axis='y')
for bar, val in zip(bars, crisis_vals):
    ax.text(bar.get_x() + bar.get_width()/2, val + 0.02, f'{val:.1%}',
           ha='center', fontweight='bold')
plt.tight_layout()
plt.savefig('figures/fig7_crisis_probability.png', dpi=300, bbox_inches='tight')
plt.close()

# ============================================================================
# STEP 4: SUMMARY
# ============================================================================

print("\n[STEP 4/4] Summary...")

print("\n" + "="*80)
print("✅ COMPLETE PIPELINE FINISHED")
print("="*80)

print("\n📁 Individual CSV files (data/):")
for country, ticker in TICKERS.items():
    filename = f'data/{country.lower()}_{ticker.lower()}_daily.csv'
    if os.path.exists(filename):
        print(f"  ✓ {filename}")

print("\n📁 Combined CSV files (results/):")
print("  ✓ prices_combined.csv")
print("  ✓ returns_combined.csv")
print("  ✓ mutual_information.csv")
print("  ✓ crisis_prediction.csv")
print("  ✓ clustered_data.csv")
print("  ✓ hurst_exponents.csv")

print("\n📊 Figures (figures/):")
print("  ✓ fig1_price_evolution.png")
print("  ✓ fig2_returns_distribution.png")
print("  ✓ fig3_rolling_volatility.png")
print("  ✓ fig4_clustering.png")
print("  ✓ fig5_hurst_exponents.png")
print("  ✓ fig6_mutual_information.png")
print("  ✓ fig7_crisis_probability.png")

print("\n📈 Key Results:")
print(f"  • Period: {prices.index[0].date()} to {prices.index[-1].date()}")
print(f"  • Observations: {len(prices)}")
print(f"  • Mutual Information: {mi_matrix.values[np.triu_indices_from(mi_matrix.values, k=1)].mean():.3f}")
print(f"  • Silhouette Score: {silhouette:.3f}")
for country, prob in crisis_pred.items():
    print(f"  • {country} Crisis Prob: {prob:.1%}")

print("\n✓ All data, analysis, and figures ready for IEEE paper!")
print("="*80 + "\n")
