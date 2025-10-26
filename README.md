# Financial Bubble Detection in Andean Markets

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)
![IEEE](https://img.shields.io/badge/Format-IEEE-red)

**Comparative Analysis of Bayesian Networks, K-Means Clustering, and ARFIMA**  
Applied to Latin American Stock Markets (Chile, Peru, Colombia)

---

## Overview

This repository contains the complete implementation of three advanced methodologies for detecting financial bubbles in Latin American markets:

1. **Bayesian Networks** - Probabilistic dependencies and crisis prediction
2. **K-Means Clustering** - Market regime identification (k=3)
3. **ARFIMA + Hurst Exponent** - Long-memory detection

**Data Period:** June 2013 - August 2022 (approximately 2,317 observations)  
**Markets:** iShares MSCI ETFs (ECH, EPU, ICOL)  
**Source:** Yahoo Finance via yfinance library

---

## Key Results

| Metric | Chile | Peru | Colombia |
|--------|-------|------|----------|
| **Hurst Exponent** | 0.629 | 0.663 | 0.580 |
| **Crisis Probability (High Volatility)** | 33% | 38% | 35% |
| **Mutual Information (avg)** | 0.108 | 0.164 | 0.142 |

**Additional Findings:**
- K-Means Silhouette Score: 0.401
- Strong cross-market integration (Mutual Information > 0.10)
- All markets exhibit persistent long-memory effects (H > 0.5)
- COVID-19 crisis period successfully identified by all three methods

---

## Repository Structure

```
financial-bubble-detection-andean-markets/
├── data/              # Raw ETF data (3 CSV files)
├── results/           # Analysis outputs (6 CSV files)
├── figures/           # Publication figures (7 PNG files, 300 DPI)
├── scripts/           # Python analysis code
├── paper/             # IEEE LaTeX paper
└── docs/              # Additional documentation
```

---

## Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Dependencies Installation

```bash
# Clone repository
git clone https://github.com/yourusername/financial-bubble-detection.git
cd financial-bubble-detection

# Install required packages
pip install -r requirements.txt
```

### Required Python Packages

- pandas >= 1.3.0
- numpy >= 1.21.0
- yfinance >= 0.2.0
- matplotlib >= 3.5.0
- seaborn >= 0.11.0
- scikit-learn >= 1.0.0
- scipy >= 1.7.0
- statsmodels >= 0.13.0
- networkx >= 2.6.0

---

## Usage

### Quick Start

Execute the complete analysis pipeline:

```bash
python scripts/complete_analysis.py
```

This single command will:

1. Download ETF data from Yahoo Finance (ECH, EPU, ICOL)
2. Perform data cleaning and preprocessing
3. Execute Bayesian Networks analysis
4. Run K-Means clustering (k=3)
5. Calculate ARFIMA and Hurst exponents
6. Generate all 7 publication-ready figures (300 DPI PNG)
7. Export results to CSV files

### Output Files

**Data Directory (data/):**
- chile_ech_daily.csv - Chile iShares MSCI ETF (ECH)
- peru_epu_daily.csv - Peru iShares MSCI ETF (EPU)
- colombia_icol_daily.csv - Colombia iShares MSCI ETF (ICOL)

**Results Directory (results/):**
- prices_combined.csv - Combined closing prices
- returns_combined.csv - Daily returns
- mutual_information.csv - Cross-market dependencies
- crisis_prediction.csv - Crisis probability estimates
- clustered_data.csv - K-means cluster assignments
- hurst_exponents.csv - Long-memory analysis results

**Figures Directory (figures/):**
- fig1_price_evolution.png - Normalized price trajectories (2013-2022)
- fig2_returns_distribution.png - Return distribution histograms
- fig3_rolling_volatility.png - 30-day rolling volatility
- fig4_clustering.png - K-means clustering visualization
- fig5_hurst_exponents.png - Hurst exponents by country
- fig6_mutual_information.png - Mutual information heatmap
- fig7_crisis_probability.png - Crisis probability conditional on high volatility

---

## Methodology Summary

### Bayesian Networks

**Discretization:**
- Returns: 5 states (large decline, moderate decline, stable, moderate growth, large growth)
- Volatility: 4 states (low, medium, high, very high)

**Analysis:**
- Mutual information between markets
- Conditional probability tables
- Chi-square independence tests
- Crisis prediction using Bayesian inference

### K-Means Clustering

**Features:**
- 30-day rolling mean return
- 30-day rolling volatility
- 30-day momentum

**Parameters:**
- Number of clusters (k) = 3
- Initialization: k-means++
- Maximum iterations: 300
- Random state: 42 (reproducibility)

**Clusters Identified:**
- Cluster 0: Stable market conditions
- Cluster 1: Growth/expansion phase
- Cluster 2: Crisis/high volatility

### ARFIMA + Hurst Exponent

**Method:** Rescaled Range (R/S) Analysis

**Interpretation:**
- H > 0.5: Persistent (long memory, trending behavior)
- H = 0.5: Random walk (efficient market)
- H < 0.5: Anti-persistent (mean-reverting)

**Findings:** All three markets exhibit H > 0.60, indicating strong long-memory effects characteristic of bubble dynamics.

---

## Authors

### First Author
**Briggitte Jhosselyn Vilca Chambilla**  
Lead Researcher & Implementation  
Universidad Nacional del Altiplano  
Puno, Peru  
Email: 71639757@est.unap.pe

### Second Author
**Renato Quispe Vargas**  
Co-Researcher & Data Analysis  
Universidad Nacional del Altiplano  
Puno, Peru  
Email: 72535253@est.unap.edu.pe

### Third Author
**Fred Torres Cruz**  
Supervisor & Validation  
Universidad Nacional del Altiplano  
Puno, Peru  
Email: ftorres@unap.edu.pe

---



## Data Sources

All financial market data is publicly available through Yahoo Finance:

- **Chile:** iShares MSCI Chile ETF (Ticker: ECH)  
  URL: https://finance.yahoo.com/quote/ECH/history

- **Peru:** iShares MSCI Peru ETF (Ticker: EPU)  
  URL: https://finance.yahoo.com/quote/EPU/history

- **Colombia:** iShares MSCI Colombia ETF (Ticker: ICOL)  
  URL: https://finance.yahoo.com/quote/ICOL/history

**Access:** Data can be downloaded programmatically using the yfinance Python library (https://pypi.org/project/yfinance/) or manually via Yahoo Finance web interface.

---

## License

This project is licensed under the MIT License.

```
MIT License

Copyright (c) 2025 Briggitte Jhosselyn Vilca Chambilla, Renato Quispe Vargas, Fred Torres Cruz

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

---

## Contact

**Primary Contact:**  
Briggitte Jhosselyn Vilca Chambilla  
Email: 71639757@est.unap.pe

**Co-Authors:**  
Renato Quispe Vargas: 72535253@est.unap.edu.pe  
Fred Torres Cruz: ftorres@unap.edu.pe

**Institution:**  
Universidad Nacional del Altiplano  
Puno, Peru  
Website: https://www.unap.edu.pe/

---

## Acknowledgments

- Data source: Yahoo Finance (yfinance library)
- Methodology references: See paper/references.bib
- Conference format: IEEE Standards
- Institution: Universidad Nacional del Altiplano, Puno, Peru

---

## Version History

**v1.0.0** (October 2025)
- Initial release
- Complete implementation of three methodologies
- All 7 figures generated
- Full IEEE paper included
- Comprehensive documentation

---

## Contributing

This is an academic research project. For questions, suggestions, or collaborations:

1. Open an issue on GitHub
2. Contact authors via email
3. Fork the repository for modifications (cite original work)

---

## Future Work

Potential extensions and improvements:

- Real-time monitoring dashboard
- Additional Latin American markets (Argentina, Brazil, Mexico)
- Alternative clustering algorithms (DBSCAN, Gaussian Mixture Models)
- Deep learning approaches (LSTM, GRU for time series prediction)
- Integration with fundamental economic indicators
- Extended time period analysis (including 2023-2025 data)

---

**Last Updated:** October 26, 2025  
**Status:** Active Research Project  
**Conference Submission:** Pending
