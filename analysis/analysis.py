# analysis/analysis.py
# CSI 300 Quant Strategy — Robustness Analysis
# David Lupinski | FH BFI Wien | 2026

import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Projekt-Root zum Suchpfad hinzufügen
sys.path.insert(0, '.')

# Aus backtest.py importieren — kein Code-Duplikat
from backtest.backtest import (
    simulate_strategy,
    calculate_metrics,
    load_predictions,
    load_prices,
    COST_BUY,
    COST_SELL,
)


# ============================================================
# BLOCK 1 — load_data()
# Lädt alle benötigten Daten für die Robustness Tests
# ============================================================

def load_data():
    """Lädt RF Predictions, XGB Predictions und monatliche Returns."""
    print("=== Lade Daten für Robustness Tests ===")
    rf_dict, xgb_dict   = load_predictions()
    monthly_returns      = load_prices()
    print(f"\nDaten geladen:")
    print(f"  RF Signale:  {len(rf_dict)} Monate")
    print(f"  XGB Signale: {len(xgb_dict)} Monate")
    print(f"  Returns:     {monthly_returns['date'].nunique()} Monate, {monthly_returns['ts_code'].nunique()} Aktien")
    return rf_dict, xgb_dict, monthly_returns

# ============================================================
# BLOCK 2 — define_market_phases()
# Bull = Markt-Return > 0 | Bear = Markt-Return <= 0
# Proxy: gleichgewichteter Durchschnitt aller Aktien pro Monat
# ============================================================

def define_market_phases(monthly_returns):
    """
    Klassifiziert jeden Monat als Bull oder Bear.
    Returns ein Dict: { date -> 'bull' oder 'bear' }
    """
    # Durchschnittlicher Markt-Return pro Monat (alle Aktien gleichgewichtet)
    market_returns = (
        monthly_returns
        .groupby('date')['monthly_return']
        .mean()
        .reset_index()
    )
    market_returns.columns = ['date', 'market_return']

    # Bull wenn positiv, Bear wenn negativ oder null
    market_returns['phase'] = market_returns['market_return'].apply(
        lambda r: 'bull' if r > 0 else 'bear'
    )

    # Ausgabe zur Kontrolle
    print("\n=== Marktphasen pro Monat ===")
    for _, row in market_returns.iterrows():
        symbol = '↑' if row['phase'] == 'bull' else '↓'
        print(f"  {row['date'].date()}  {row['market_return']:+.2%}  {symbol} {row['phase'].upper()}")

    bull_months = (market_returns['phase'] == 'bull').sum()
    bear_months = (market_returns['phase'] == 'bear').sum()
    print(f"\n  Bull-Monate: {bull_months} | Bear-Monate: {bear_months}")

    # Als Dict zurückgeben: { date -> 'bull'/'bear' }
    phase_dict = market_returns.set_index('date')['phase'].to_dict()
    return phase_dict

# ============================================================
# BLOCK 3 — bull_bear_analysis()
# Simuliert RF und XGB separat für Bull- und Bear-Phasen
# ============================================================

def bull_bear_analysis(rf_dict, xgb_dict, monthly_returns, phase_dict):
    """
    Führt den Backtest getrennt für Bull- und Bear-Phasen durch.
    Gibt Metriken-Dict zurück für den Chart in Block 4.
    """
    results = {}

    for phase in ['bull', 'bear']:
        # Welche Monate gehören zu dieser Phase?
        phase_dates = {d for d, p in phase_dict.items() if p == phase}

        if len(phase_dates) == 0:
            print(f"\n⚠️  Keine {phase.upper()}-Monate in Testdaten — übersprungen.")
            results[phase] = None
            continue

        print(f"\n{'='*50}")
        print(f"  PHASE: {phase.upper()} ({len(phase_dates)} Monate)")
        print(f"{'='*50}")

        # Signal-Dicts auf diese Phase filtern
        rf_phase  = {d: v for d, v in rf_dict.items()  if d in phase_dates}
        xgb_phase = {d: v for d, v in xgb_dict.items() if d in phase_dates}

        # Returns auf diese Phase filtern
        returns_phase = monthly_returns[monthly_returns['date'].isin(phase_dates)].copy()

        # Benchmark für diese Phase
        from backtest.backtest import benchmark_returns, COST_BUY
        all_stocks = returns_phase['ts_code'].unique().tolist()
        phase_dates_sorted = sorted(phase_dates)

        # Einfacher Benchmark: gleichgewichteter Return über Phase-Monate
        bnh_value = 100000 * (1 - COST_BUY)
        bnh_rows  = []
        weight    = 1.0 / len(all_stocks)
        for date in phase_dates_sorted:
            month_data   = returns_phase[returns_phase['date'] == date]
            total_return = 0.0
            for stock in all_stocks:
                row = month_data[month_data['ts_code'] == stock]
                ret = row['monthly_return'].values[0] if len(row) > 0 else 0.0
                total_return += weight * ret
            bnh_value = bnh_value * (1 + total_return)
            bnh_rows.append({'date': date, 'portfolio_value': bnh_value})
        bnh_phase = pd.DataFrame(bnh_rows)

        # Strategien simulieren (nur wenn Signale vorhanden)
        phase_results = {}
        for name, sig_dict in [('Random Forest', rf_phase), ('XGBoost', xgb_phase)]:
            if len(sig_dict) == 0:
                print(f"  ⚠️  Keine Signale für {name} in {phase.upper()}-Phase.")
                phase_results[name] = None
                continue
            sim = simulate_strategy(sig_dict, returns_phase, f"{name} ({phase.upper()})")
            # Mindestens 2 Monate nötig für calculate_metrics()
            if len(sig_dict) < 2:
                print(f"  ⚠️  Nur {len(sig_dict)} Monat — zu wenig für Metriken. Übersprungen.")
                phase_results[name] = None
                continue
            metrics = calculate_metrics(sim, bnh_phase, f"{name} ({phase.upper()})")
            phase_results[name] = metrics

        results[phase] = phase_results

    return results

# ============================================================
# BLOCK 4 — plot_bull_bear()
# Balkendiagramm: Annualized Return pro Phase und Modell
# ============================================================

def plot_bull_bear(results, save_path='report/figures/bull_bear_analysis.png'):
    """
    Erstellt Balkendiagramm: RF vs XGB in Bull/Bear Phasen.
    Wenn eine Phase keine Daten hat, wird sie als 0 dargestellt
    mit einem Hinweis-Text im Chart.
    """
    phases  = ['bull', 'bear']
    models  = ['Random Forest', 'XGBoost']
    colors  = {'Random Forest': 'steelblue', 'XGBoost': 'darkorange'}

    # Daten sammeln
    plot_data = {}
    for model in models:
        plot_data[model] = []
        for phase in phases:
            if results[phase] is None or results[phase][model] is None:
                plot_data[model].append(np.nan)
            else:
                plot_data[model].append(results[phase][model]['ann_return'])

    # Chart aufbauen
    fig, ax = plt.subplots(figsize=(10, 6))

    x      = np.arange(len(phases))
    width  = 0.35

    for i, model in enumerate(models):
        values = plot_data[model]
        bars   = ax.bar(
            x + i * width,
            [v if not np.isnan(v) else 0 for v in values],
            width,
            label=model,
            color=colors[model],
            alpha=0.85,
            edgecolor='black',
            linewidth=0.5
        )
        # Wert-Label auf jeden Balken
        for j, (bar, val) in enumerate(zip(bars, values)):
            if np.isnan(val):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    0.005,
                    'No data\n(test period)',
                    ha='center', va='bottom',
                    fontsize=8, color='gray', style='italic'
                )
            else:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + (0.005 if bar.get_height() >= 0 else -0.02),
                    f'{val:+.1%}',
                    ha='center', va='bottom',
                    fontsize=9, fontweight='bold'
                )

    # Nulllinie
    ax.axhline(y=0, color='black', linewidth=0.8, linestyle='-')

    # Formatierung
    ax.set_title('Robustness Test 1 — Bull vs. Bear Market Performance\n'
                 'Annualized Return by Market Phase (2023-2024 Test Period)',
                 fontsize=13, fontweight='bold')
    ax.set_ylabel('Annualized Return')
    ax.set_xticks(x + width / 2)
    ax.set_xticklabels(['Bull Market\n(↑ Avg. Return > 0)',
                         'Bear Market\n(↓ Avg. Return ≤ 0)'], fontsize=11)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
    ax.legend(fontsize=11)
    ax.grid(True, axis='y', alpha=0.3)

    # Hinweis-Text unten im Chart
    ax.text(0.5, -0.13,
            '⚠️  Test period: 2023-2024, 300 stocks. '
            'Training: 2018-2022. Full CSI 300 dataset.',
            transform=ax.transAxes,
            ha='center', fontsize=8, color='gray', style='italic')

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\nBull/Bear chart saved: {save_path}")
    plt.show()

# ============================================================
# BLOCK 5 — cost_sensitivity_analysis()
# Backtest mit verschiedenen Kostenmultiplikatoren
# ============================================================

def cost_sensitivity_analysis(rf_dict, xgb_dict, monthly_returns):
    """
    Führt den Backtest mit verschiedenen Kostenmultiplikatoren durch.
    Zeigt ab welchem Kostenniveau die Strategie schlechter als B&H wird.
    """
    from backtest.backtest import benchmark_returns

    # Kostenmultiplikatoren testen
    multipliers = [0.0, 0.5, 1.0, 2.0, 3.0, 5.0]

    # Buy-and-Hold Benchmark (Kosten fix bei 1×)
    bnh_results = benchmark_returns(monthly_returns)
    bnh_metrics = calculate_metrics(bnh_results, bnh_results, 'Buy-and-Hold')
    bnh_ann_return = bnh_metrics['ann_return']

    print(f"\n=== Buy-and-Hold Annualized Return (Baseline): {bnh_ann_return:+.2%} ===")
    print(f"\n{'Multiplier':<12} {'RF Ann.Return':>15} {'XGB Ann.Return':>15}")
    print("-" * 44)

    rf_returns  = []
    xgb_returns = []

    for mult in multipliers:
        # Temporär Kosten überschreiben
        import backtest.backtest as bt
        bt.COST_BUY  = COST_BUY  * mult
        bt.COST_SELL = COST_SELL * mult

        # Simulieren
        rf_sim  = simulate_strategy(rf_dict,  monthly_returns, f'RF  (cost {mult}×)')
        xgb_sim = simulate_strategy(xgb_dict, monthly_returns, f'XGB (cost {mult}×)')

        # Metriken
        rf_m  = calculate_metrics(rf_sim,  bnh_results, f'RF  {mult}×')
        xgb_m = calculate_metrics(xgb_sim, bnh_results, f'XGB {mult}×')

        rf_returns.append(rf_m['ann_return'])
        xgb_returns.append(xgb_m['ann_return'])

        print(f"  {mult:<10.1f} {rf_m['ann_return']:>+14.2%} {xgb_m['ann_return']:>+14.2%}")

    # Kosten zurücksetzen auf Original
    bt.COST_BUY  = COST_BUY
    bt.COST_SELL = COST_SELL

    print(f"\n  {'B&H (Referenz)':<10} {bnh_ann_return:>+14.2%} {bnh_ann_return:>+14.2%}")

    return multipliers, rf_returns, xgb_returns, bnh_ann_return

# ============================================================
# BLOCK 6 — plot_cost_sensitivity()
# Liniendiagramm: Annualized Return vs. Kostenmultiplikator
# ============================================================

def plot_cost_sensitivity(multipliers, rf_returns, xgb_returns, bnh_return,
                          save_path='report/figures/cost_sensitivity.png'):
    """
    Liniendiagramm: wie verändert sich der Return bei steigenden Kosten?
    Zeigt Break-even Punkt gegenüber Buy-and-Hold.
    """
    fig, ax = plt.subplots(figsize=(11, 6))

    # RF und XGB Linien
    ax.plot(multipliers, [r * 100 for r in rf_returns],
            color='steelblue', linewidth=2.5, marker='o',
            markersize=7, label='Random Forest')

    ax.plot(multipliers, [r * 100 for r in xgb_returns],
            color='darkorange', linewidth=2.5, marker='s',
            markersize=7, label='XGBoost')

    # Buy-and-Hold Referenzlinie
    ax.axhline(y=bnh_return * 100, color='gray', linewidth=2,
               linestyle='--', label=f'Buy-and-Hold ({bnh_return:+.1%})')

    # Nulllinie
    ax.axhline(y=0, color='black', linewidth=0.8, linestyle=':')

    # Break-even Punkte markieren
    for returns, color, name in [
        (rf_returns,  'steelblue',  'RF'),
        (xgb_returns, 'darkorange', 'XGB')
    ]:
        for i in range(len(multipliers) - 1):
            # Kreuzt die Linie den B&H Return zwischen zwei Punkten?
            r1, r2 = returns[i], returns[i+1]
            m1, m2 = multipliers[i], multipliers[i+1]
            if (r1 - bnh_return) * (r2 - bnh_return) < 0:
                # Linearer Interpolations-Schnittpunkt
                t = (bnh_return - r1) / (r2 - r1)
                breakeven_x = m1 + t * (m2 - m1)
                ax.axvline(x=breakeven_x, color=color,
                           linewidth=1, linestyle=':', alpha=0.7)
                ax.text(breakeven_x + 0.05, bnh_return * 100 + 1,
                        f'{name} break-even\n~{breakeven_x:.1f}×',
                        color=color, fontsize=8)

    # Aktueller Kostenpunkt markieren
    ax.axvline(x=1.0, color='black', linewidth=1.2,
               linestyle='--', alpha=0.5)
    ax.text(1.05, ax.get_ylim()[0] * 0.95,
            'Current\ncosts', fontsize=8, color='black', alpha=0.7)

    # Formatierung
    ax.set_title('Robustness Test 2 — Transaction Cost Sensitivity\n'
                 'Annualized Return vs. Cost Multiplier',
                 fontsize=13, fontweight='bold')
    ax.set_xlabel('Cost Multiplier (1× = Current Costs: Buy 0.13%, Sell 0.23%)',
                  fontsize=11)
    ax.set_ylabel('Annualized Return (%)', fontsize=11)
    ax.set_xticks(multipliers)
    ax.set_xticklabels([f'{m}×' for m in multipliers])
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0f}%'))
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    # Hinweis unten
    ax.text(0.5, -0.13,
            '⚠️  Based on full dataset: 300 stocks, 2018-2024. '
            'Training: 2018-2022 | Test: 2023-2024.',
            transform=ax.transAxes,
            ha='center', fontsize=8, color='gray', style='italic')

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\nCost sensitivity chart saved: {save_path}")
    plt.show()

# ============================================================
# MAIN
# ============================================================

if __name__ == '__main__':
    # Daten laden
    rf_dict, xgb_dict, monthly_returns = load_data()

    # Test 1 — Bull/Bear Analyse
    print("\n" + "="*60)
    print("  ROBUSTNESS TEST 1 — BULL/BEAR MARKET ANALYSIS")
    print("="*60)
    phase_dict = define_market_phases(monthly_returns)
    bb_results = bull_bear_analysis(rf_dict, xgb_dict, monthly_returns, phase_dict)
    plot_bull_bear(bb_results)

    # Test 2 — Kostensensitivität
    print("\n" + "="*60)
    print("  ROBUSTNESS TEST 2 — TRANSACTION COST SENSITIVITY")
    print("="*60)
    multipliers, rf_rets, xgb_rets, bnh_ret = cost_sensitivity_analysis(
        rf_dict, xgb_dict, monthly_returns
    )
    plot_cost_sensitivity(multipliers, rf_rets, xgb_rets, bnh_ret)