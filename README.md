# CSI 300 Quantitative Strategy
### Multi-Factor + Machine Learning Stock Selection on A-Shares

**Course:** Quantitative Investment & Algorithmic Trading  
**Author:** David Lupinski  
**Institution:** XIAMEN UNIVERSITY  
**Submission:** May 8, 2026

## Strategy Overview
Monthly-rebalanced long-only stock selection strategy on CSI 300 constituents.  
Factors: ROE, Earnings Yield, Momentum, Turnover Rate, MFI, Value-Momentum Composite.  
Models: Random Forest vs. XGBoost — compared against Buy & Hold and Dual MA baseline.

## Project Structure
| Folder | Content |
|---|---|
| `factors/` | Factor construction |
| `data_processing/` | Cleaning, labeling, train/test split |
| `models/` | Random Forest and XGBoost |
| `backtest/` | Strategy simulation + metrics |
| `analysis/` | Robustness tests + charts |
| `report/figures/` | Exported charts for report |

## Data
Local development: yfinance | Final backtest: JoinQuant