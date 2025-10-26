# Methodology Documentation

## 1. Bayesian Networks

### Overview
Bayesian Networks are probabilistic graphical models that represent conditional dependencies between variables through directed acyclic graphs.

### Discretization Strategy

#### Returns Discretization (5 states)
- **State 0 (Large Decline)**: Returns < -2%
- **State 1 (Moderate Decline)**: Returns between -2% and -0.5%
- **State 2 (Stable)**: Returns between -0.5% and 0.5%
- **State 3 (Moderate Growth)**: Returns between 0.5% and 2%
- **State 4 (Large Growth)**: Returns > 2%

#### Volatility Discretization (4 states)  
- **State 0 (Low)**: Volatility < 1%
- **State 1 (Medium)**: Volatility between 1% and 2%
- **State 2 (High)**: Volatility between 2% and 4%
- **State 3 (Very High)**: Volatility > 4%

### Analysis Components

#### Mutual Information
Measures the amount of information obtained about one variable through another:
```
MI(X,Y) = ΣΣ P(x,y) log(P(x,y) / (P(x)P(y)))
```

#### Crisis Prediction
- **Definition**: Crisis occurs when returns are in states 0 or 1 (declines)
- **Condition**: Given high volatility (states 2 or 3)
- **Formula**: P(Crisis|High Volatility) = P(Return ≤ 1 | Volatility ≥ 2)

---

## 2. K-Means Clustering

### Feature Engineering

#### Primary Features
1. **30-day Rolling Return**: Moving average of daily returns over 30 trading days
2. **30-day Rolling Volatility**: Standard deviation of returns over 30 trading days  
3. **30-day Momentum**: Price change percentage over 30 trading days

### Algorithm Parameters
- **Number of clusters (k)**: 3
- **Initialization**: k-means++
- **Maximum iterations**: 300
- **Random state**: 42 (for reproducibility)
- **Convergence tolerance**: 1e-4

### Cluster Interpretation
- **Cluster 0**: Stable market conditions (low volatility, moderate returns)
- **Cluster 1**: Growth/expansion phase (positive returns, moderate volatility)
- **Cluster 2**: Crisis/high volatility (high volatility, often negative returns)

### Quality Metrics
- **Silhouette Score**: Measures cluster cohesion and separation
- **Within-cluster Sum of Squares**: Measures cluster compactness
- **Between-cluster Sum of Squares**: Measures cluster separation

---

## 3. ARFIMA + Hurst Exponent

### Rescaled Range (R/S) Analysis

#### Algorithm Steps
1. **Mean deviation**: X(t) - μ
2. **Cumulative sum**: Y(n) = Σ[X(t) - μ] for t=1 to n
3. **Range**: R(n) = max(Y(n)) - min(Y(n))
4. **Standard deviation**: S(n) = √(Σ[X(t) - μ]²/n)
5. **R/S ratio**: R(n)/S(n)
6. **Log-log regression**: log(R/S) vs log(n)

#### Hurst Exponent Calculation
```
H = slope of log(R/S) vs log(lag)
```

#### Interpretation
- **H > 0.5**: Persistent (long memory, trending behavior)
  - Values closer to 1 indicate stronger persistence
  - Suggests momentum and bubble-like behavior
- **H = 0.5**: Random walk (efficient market hypothesis)
  - Pure white noise, no memory
- **H < 0.5**: Anti-persistent (mean-reverting)
  - Values closer to 0 indicate stronger mean reversion

### ARFIMA Model Connection
The Hurst exponent relates to the fractional differencing parameter (d) in ARFIMA models:
- **d = H - 0.5**
- **d > 0**: Long memory (H > 0.5)
- **d = 0**: Short memory (H = 0.5)  
- **d < 0**: Intermediate memory (H < 0.5)

---

## 4. Data Preprocessing

### Data Cleaning Steps
1. **Date standardization**: Convert to pandas datetime format
2. **Missing value handling**: Forward fill for small gaps, interpolation for larger gaps
3. **Outlier detection**: Remove returns > 20% (likely data errors)
4. **Business day alignment**: Ensure all series have same trading days

### Return Calculation
- **Formula**: log(P_t / P_{t-1})
- **Reasoning**: Log returns are more suitable for statistical analysis
- **Advantages**: Time additive, approximately normal for small changes

### Volatility Estimation
- **Method**: Rolling standard deviation of returns
- **Window**: 30 trading days (~1.5 months)
- **Annualization**: Multiply by √252 for annual volatility

---

## 5. Statistical Tests

### Independence Tests
- **Chi-square test**: Test independence between discretized variables
- **Mutual information**: Non-parametric measure of dependence
- **Significance level**: α = 0.05

### Normality Tests
- **Jarque-Bera test**: Test for normal distribution of returns
- **Shapiro-Wilk test**: Alternative normality test for smaller samples
- **Q-Q plots**: Visual assessment of normality

### Stationarity Tests
- **Augmented Dickey-Fuller**: Test for unit roots
- **KPSS test**: Confirm stationarity
- **Phillips-Perron**: Robust to serial correlation

---

## 6. Model Validation

### Cross-Validation
- **Time series split**: Respect temporal order
- **Train period**: First 70% of data
- **Test period**: Last 30% of data
- **Rolling window**: Expanding window validation

### Performance Metrics
- **Clustering**: Silhouette score, Davies-Bouldin index
- **Classification**: Precision, recall, F1-score for crisis prediction
- **Forecasting**: RMSE, MAE for return predictions

### Robustness Checks
- **Parameter sensitivity**: Test different discretization thresholds
- **Window size**: Vary rolling window lengths (20, 30, 60 days)
- **Subsample analysis**: Test on different time periods

---

## 7. Computational Complexity

### Algorithm Complexities
- **Bayesian Networks**: O(n × m²) where n = observations, m = states
- **K-Means**: O(n × k × i × d) where k = clusters, i = iterations, d = dimensions  
- **Hurst Exponent**: O(n × log(n)) for R/S analysis

### Memory Requirements
- **Typical dataset**: ~2,000 observations × 3 countries × 6 features ≈ 144 KB
- **Intermediate calculations**: ~10-50 MB depending on discretization
- **Output storage**: ~5-10 MB for all results

---

## 8. Limitations and Assumptions

### Bayesian Networks
- Assumes discrete state representation is meaningful
- Requires sufficient data in each state combination
- Limited to pairwise dependencies as implemented

### K-Means Clustering  
- Assumes spherical clusters
- Sensitive to initialization and outliers
- Requires pre-specification of cluster number

### Hurst Exponent
- Assumes log-normal distribution for R/S statistic
- Sensitive to sample size and detrending method
- May be biased in presence of structural breaks

### General Limitations
- ETF data may not perfectly represent underlying markets
- Limited to daily frequency (intraday patterns not captured)
- Analysis period may not include all market regimes