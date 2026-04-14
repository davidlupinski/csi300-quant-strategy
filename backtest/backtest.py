# backtest/backtest.py
# CSI 300 Quant Strategy — Backtest
# David Lupinski | FH BFI Wien | 2026

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# Transaktionskosten China A-Shares
COST_BUY  = 0.0013   # 0.03% Kommission + 0.10% Slippage
COST_SELL = 0.0023   # 0.03% Kommission + 0.10% Slippage + 0.10% Stamp Duty


# ============================================================
# BLOCK 1 — load_predictions()
# Lädt RF und XGB Predictions
# Gibt pro Modell ein Dict zurück: { date -> [ts_code, ...] }
# ============================================================

def load_predictions():
    rf_pred  = pd.read_csv('data/rf_predictions.csv')
    xgb_pred = pd.read_csv('data/xgb_predictions.csv')

    rf_pred['date']  = pd.to_datetime(rf_pred['date'])
    xgb_pred['date'] = pd.to_datetime(xgb_pred['date'])

    # Nur Aktien mit Signal 1 — das sind die Kaufkandidaten
    rf_signals  = rf_pred[rf_pred['y_pred']  == 1][['date', 'ts_code']]
    xgb_signals = xgb_pred[xgb_pred['y_pred'] == 1][['date', 'ts_code']]

    # Dict: { date -> [ts_code1, ts_code2, ...] }
    rf_dict  = rf_signals.groupby('date')['ts_code'].apply(list).to_dict()
    xgb_dict = xgb_signals.groupby('date')['ts_code'].apply(list).to_dict()

    print("=== RF Signals (Aktien mit y_pred=1 pro Monat) ===")
    for date, stocks in rf_dict.items():
        print(f"  {date.date()}: {stocks}")

    print("\n=== XGB Signals ===")
    for date, stocks in xgb_dict.items():
        print(f"  {date.date()}: {stocks}")

    return rf_dict, xgb_dict


# ============================================================
# BLOCK 2 — load_prices()
# Lädt tägliche Kurse, berechnet monatliche Returns pro Aktie
# ============================================================

def load_prices():
    prices = pd.read_csv('data/price_data.csv')
    prices['trade_date'] = pd.to_datetime(prices['trade_date'], format='%Y%m%d')
    prices = prices[['trade_date', 'ts_code', 'close']].copy()
    prices = prices.sort_values(['ts_code', 'trade_date'])

    # Letzter Handelstag pro Monat pro Aktie
    monthly = (
        prices
        .groupby(['ts_code', pd.Grouper(key='trade_date', freq='ME')])['close']
        .last()
        .reset_index()
    )
    monthly.columns = ['ts_code', 'date', 'close']

    # Monatlicher Return: (Kurs_t / Kurs_t-1) - 1
    monthly['monthly_return'] = (
        monthly
        .groupby('ts_code')['close']
        .pct_change()
    )

    monthly = monthly.dropna(subset=['monthly_return'])

    print("=== Monatliche Returns (erste 10 Zeilen) ===")
    print(monthly[['date', 'ts_code', 'monthly_return']].head(10))
    print(f"\nZeitraum: {monthly['date'].min().date()} -> {monthly['date'].max().date()}")
    print(f"Aktien: {monthly['ts_code'].nunique()} | Zeilen: {len(monthly)}")

    return monthly[['date', 'ts_code', 'monthly_return']]


# ============================================================
# BLOCK 3 — simulate_strategy()
# Monatliches Rebalancing mit Transaktionskosten
# ============================================================

def simulate_strategy(signal_dict, monthly_returns, strategy_name, initial_capital=100000):
    returns_lookup = monthly_returns.set_index(['date', 'ts_code'])['monthly_return'].to_dict()

    portfolio_value  = initial_capital
    current_holdings = []
    results          = []

    dates = sorted(signal_dict.keys())

    for date in dates:
        new_holdings   = signal_dict[date]
        stocks_to_sell = [s for s in current_holdings if s not in new_holdings]
        stocks_to_buy  = [s for s in new_holdings if s not in current_holdings]

        # Verkaufskosten
        n_held = len(current_holdings) if current_holdings else 1
        for stock in stocks_to_sell:
            position_value  = portfolio_value / n_held
            portfolio_value -= position_value * COST_SELL

        # Kaufkosten
        n_new = len(new_holdings) if new_holdings else 1
        for stock in stocks_to_buy:
            position_value  = portfolio_value / n_new
            portfolio_value -= position_value * COST_BUY

        # Returns anwenden (gleichgewichtet)
        if new_holdings:
            weight       = 1.0 / len(new_holdings)
            total_return = 0.0
            for stock in new_holdings:
                ret           = returns_lookup.get((date, stock), 0.0)
                total_return += weight * ret
            portfolio_value = portfolio_value * (1 + total_return)

        results.append({
            'date':            date,
            'portfolio_value': portfolio_value,
            'n_stocks':        len(new_holdings)
        })

        current_holdings = new_holdings

    results_df = pd.DataFrame(results)

    print(f"\n=== {strategy_name} Simulation ===")
    print(results_df[['date', 'portfolio_value', 'n_stocks']].to_string(index=False))
    print(f"\nStartkapital:  CNY {initial_capital:,.0f}")
    print(f"Endkapital:    CNY {results_df['portfolio_value'].iloc[-1]:,.0f}")
    pnl = results_df['portfolio_value'].iloc[-1] - initial_capital
    print(f"PnL:           CNY {pnl:+,.0f}")

    return results_df


# ============================================================
# BLOCK 4 — benchmark_returns()
# Buy-and-Hold: alle Aktien gleichgewichtet, kein Rebalancing
# ============================================================

def benchmark_returns(monthly_returns, initial_capital=100000):
    all_stocks = monthly_returns['ts_code'].unique().tolist()
    all_dates  = sorted(monthly_returns['date'].unique())
    # Nach all_dates definieren — NEU:
    strategy_start = pd.Timestamp('2023-07-31')  # erster Monat der ML-Strategien
    all_dates = [d for d in all_dates if d >= strategy_start]
    # Einmalige Kaufkosten
    portfolio_value = initial_capital * (1 - COST_BUY)
    weight          = 1.0 / len(all_stocks)
    results         = []

    for date in all_dates:
        month_data   = monthly_returns[monthly_returns['date'] == date]
        total_return = 0.0

        for stock in all_stocks:
            row = month_data[month_data['ts_code'] == stock]
            ret = row['monthly_return'].values[0] if len(row) > 0 else 0.0
            total_return += weight * ret

        portfolio_value = portfolio_value * (1 + total_return)
        results.append({'date': date, 'portfolio_value': portfolio_value})

    results_df = pd.DataFrame(results)

    print(f"\n=== Buy-and-Hold Benchmark ===")
    print(results_df[['date', 'portfolio_value']].to_string(index=False))
    print(f"\nStartkapital:  CNY {initial_capital:,.0f}")
    print(f"Endkapital:    CNY {results_df['portfolio_value'].iloc[-1]:,.0f}")
    pnl = results_df['portfolio_value'].iloc[-1] - initial_capital
    print(f"PnL:           CNY {pnl:+,.0f}")

    return results_df


# ============================================================
# BLOCK 5 — calculate_metrics()
# Alle Pflichtmetriken laut Requirements
# ============================================================

def calculate_metrics(results_df, benchmark_df, strategy_name, risk_free_rate=0.02):
    pv           = results_df['portfolio_value'].values
    monthly_rets = np.diff(pv) / pv[:-1]
    n_months     = len(monthly_rets)

    # 1. Cumulative + Annualized Return
    cumulative_return = pv[-1] / pv[0] - 1
    ann_return        = (1 + cumulative_return) ** (12 / n_months) - 1

    # 2. Sharpe Ratio
    rf_monthly  = risk_free_rate / 12
    excess_rets = monthly_rets - rf_monthly
    sharpe      = (np.mean(excess_rets) / np.std(excess_rets)) * np.sqrt(12) if np.std(excess_rets) > 0 else 0

    # 3. Maximum Drawdown
    cumulative   = pv / pv[0]
    peak         = np.maximum.accumulate(cumulative)
    drawdown     = (cumulative - peak) / peak
    max_drawdown = drawdown.min()

    # 4. Win Rate
    win_rate = np.sum(monthly_rets > 0) / n_months

    # 5. Profit/Loss Ratio
    wins     = monthly_rets[monthly_rets > 0]
    losses   = monthly_rets[monthly_rets < 0]
    pl_ratio = (np.mean(wins) / abs(np.mean(losses))) if len(losses) > 0 and len(wins) > 0 else np.nan

    # 6. Alpha & Beta
    bnh_pv      = benchmark_df.set_index('date')['portfolio_value']
    strat_dates = results_df['date'].values
    bnh_aligned = bnh_pv.reindex(strat_dates).ffill()
    bnh_vals    = bnh_aligned.values
    bnh_rets    = np.diff(bnh_vals) / bnh_vals[:-1]

    min_len = min(len(monthly_rets), len(bnh_rets))
    s_rets  = monthly_rets[:min_len]
    b_rets  = bnh_rets[:min_len]

    beta     = np.cov(s_rets, b_rets)[0, 1] / np.var(b_rets) if np.var(b_rets) > 0 else 0
    bnh_ann  = (1 + (bnh_vals[-1] / bnh_vals[0] - 1)) ** (12 / len(bnh_rets)) - 1
    alpha    = ann_return - risk_free_rate - beta * (bnh_ann - risk_free_rate)

    # 7. Information Ratio
    excess_vs_bnh = s_rets - b_rets
    info_ratio    = (np.mean(excess_vs_bnh) / np.std(excess_vs_bnh)) * np.sqrt(12) if np.std(excess_vs_bnh) > 0 else 0

    print(f"\n{'='*47}")
    print(f"  {strategy_name} — Performance Metrics")
    print(f"{'='*47}")
    print(f"  Cumulative Return:   {cumulative_return:+.2%}")
    print(f"  Annualized Return:   {ann_return:+.2%}  (auf 12 Monate hochgerechnet)")
    print(f"  Sharpe Ratio:        {sharpe:.4f}")
    print(f"  Max Drawdown:        {max_drawdown:.2%}")
    print(f"  Win Rate:            {win_rate:.2%}")
    print(f"  Profit/Loss Ratio:   {pl_ratio:.4f}" if not np.isnan(pl_ratio) else "  Profit/Loss Ratio:   n/a (keine Gewinnmonate)")
    print(f"  Alpha:               {alpha:+.4f}")
    print(f"  Beta:                {beta:.4f}")
    print(f"  Information Ratio:   {info_ratio:.4f}")
    print(f"{'='*47}")

    return {
        'strategy':           strategy_name,
        'cumulative_return':  cumulative_return,
        'ann_return':         ann_return,
        'sharpe':             sharpe,
        'max_drawdown':       max_drawdown,
        'win_rate':           win_rate,
        'pl_ratio':           pl_ratio,
        'alpha':              alpha,
        'beta':               beta,
        'info_ratio':         info_ratio
    }


# ============================================================
# BLOCK 6 — plot_nav_curve()
# NAV-Kurve: RF vs XGB vs Buy-and-Hold
# ============================================================

def plot_nav_curve(rf_results, xgb_results, bnh_results,
                  save_path='report/figures/nav_curve.png'):
    fig, ax = plt.subplots(figsize=(12, 6))

    # Normierung auf 1.0 am jeweiligen Startpunkt
    def normalize(df):
        pv = df.set_index('date')['portfolio_value']
        return pv / pv.iloc[0]

    nav_rf  = normalize(rf_results)
    nav_xgb = normalize(xgb_results)
    nav_bnh = normalize(bnh_results)

    ax.plot(nav_bnh.index, nav_bnh.values,
            color='gray',       linewidth=2,   linestyle='--', label='Buy-and-Hold (Benchmark)')
    ax.plot(nav_rf.index,  nav_rf.values,
            color='steelblue',  linewidth=2.5, label='Random Forest')
    ax.plot(nav_xgb.index, nav_xgb.values,
            color='darkorange', linewidth=2.5, label='XGBoost')

    ax.axhline(y=1.0, color='black', linewidth=0.8, linestyle=':')
    ax.set_title('CSI 300 Quant Strategy — NAV Curve vs. Benchmark (2023)',
                 fontsize=14, fontweight='bold')
    ax.set_xlabel('Date')
    ax.set_ylabel('Normalized NAV (Start = 1.0)')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150)
    print(f"\nNAV chart saved: {save_path}")
    plt.show()


# ============================================================
# BLOCK 7 — print_comparison_table()
# Vergleichstabelle aller drei Strategien
# ============================================================

def print_comparison_table(rf_m, xgb_m, bnh_m):
    print("\n")
    print("=" * 65)
    print(f"  {'Metric':<25} {'Random Forest':>12} {'XGBoost':>12} {'Buy&Hold':>10}")
    print("=" * 65)

    def fmt_pct(v):
        return f"{v:+.2%}" if not np.isnan(v) else "n/a"
    def fmt_num(v):
        return f"{v:.4f}" if not np.isnan(v) else "n/a"

    print(f"  {'Cumulative Return':<25} {fmt_pct(rf_m['cumulative_return']):>12} {fmt_pct(xgb_m['cumulative_return']):>12} {fmt_pct(bnh_m['cumulative_return']):>10}")
    print(f"  {'Annualized Return':<25} {fmt_pct(rf_m['ann_return']):>12} {fmt_pct(xgb_m['ann_return']):>12} {fmt_pct(bnh_m['ann_return']):>10}")
    print(f"  {'Sharpe Ratio':<25} {fmt_num(rf_m['sharpe']):>12} {fmt_num(xgb_m['sharpe']):>12} {fmt_num(bnh_m['sharpe']):>10}")
    print(f"  {'Max Drawdown':<25} {fmt_pct(rf_m['max_drawdown']):>12} {fmt_pct(xgb_m['max_drawdown']):>12} {fmt_pct(bnh_m['max_drawdown']):>10}")
    print(f"  {'Win Rate':<25} {fmt_pct(rf_m['win_rate']):>12} {fmt_pct(xgb_m['win_rate']):>12} {fmt_pct(bnh_m['win_rate']):>10}")
    print(f"  {'Profit/Loss Ratio':<25} {fmt_num(rf_m['pl_ratio']) if not np.isnan(rf_m['pl_ratio']) else 'n/a':>12} {fmt_num(xgb_m['pl_ratio']) if not np.isnan(xgb_m['pl_ratio']) else 'n/a':>12} {fmt_num(bnh_m['pl_ratio']) if not np.isnan(bnh_m['pl_ratio']) else 'n/a':>10}")
    print(f"  {'Alpha':<25} {fmt_num(rf_m['alpha']):>12} {fmt_num(xgb_m['alpha']):>12} {fmt_num(bnh_m['alpha']):>10}")
    print(f"  {'Beta':<25} {fmt_num(rf_m['beta']):>12} {fmt_num(xgb_m['beta']):>12} {fmt_num(bnh_m['beta']):>10}")
    print(f"  {'Information Ratio':<25} {fmt_num(rf_m['info_ratio']):>12} {fmt_num(xgb_m['info_ratio']):>12} {'n/a':>10}")
    print("=" * 65)


# ============================================================
# MAIN
# ============================================================

if __name__ == '__main__':
    # Daten laden
    rf_dict, xgb_dict = load_predictions()
    monthly_returns   = load_prices()

    # Strategien simulieren
    rf_results  = simulate_strategy(rf_dict,  monthly_returns, 'Random Forest')
    xgb_results = simulate_strategy(xgb_dict, monthly_returns, 'XGBoost')
    bnh_results = benchmark_returns(monthly_returns)

    # Metriken berechnen
    rf_metrics  = calculate_metrics(rf_results,  bnh_results, 'Random Forest')
    xgb_metrics = calculate_metrics(xgb_results, bnh_results, 'XGBoost')
    bnh_metrics = calculate_metrics(bnh_results, bnh_results, 'Buy-and-Hold')

    # Vergleichstabelle
    print_comparison_table(rf_metrics, xgb_metrics, bnh_metrics)

    # NAV-Kurve
    plot_nav_curve(rf_results, xgb_results, bnh_results)