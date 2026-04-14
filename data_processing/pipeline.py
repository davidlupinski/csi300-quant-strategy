# data_processing/pipeline.py

import pandas as pd
import numpy as np
import os


def build_monthly_panel(price_data: pd.DataFrame) -> pd.DataFrame:
    """
    Baut monatliches Panel aus täglichen Kursdaten.
    Berechnet Momentum und MFI für jeden Monat × jede Aktie.
    """
    all_rows = []

    for stock in price_data['ts_code'].unique():

        df = price_data[price_data['ts_code'] == stock].copy()
        df = df.sort_values('trade_date').reset_index(drop=True)
        df['trade_date'] = pd.to_datetime(df['trade_date'], format='%Y%m%d')
        df = df.set_index('trade_date')

        # Tägliche Returns
        df['daily_return'] = df['close'].pct_change()

        # Faktor 1: Momentum 20 Tage
        df['momentum'] = df['close'].pct_change(periods=20)

        # Faktor 2: MFI (monatlich berechnet aus täglichen Daten)
        df['typical_price'] = (df['high'] + df['low'] + df['close']) / 3
        df['raw_money_flow'] = df['typical_price'] * df['vol']
        df['positive_flow'] = df['raw_money_flow'].where(
            df['typical_price'] > df['typical_price'].shift(1), 0)
        df['negative_flow'] = df['raw_money_flow'].where(
            df['typical_price'] <= df['typical_price'].shift(1), 0)
        df['mfi'] = 100 - (100 / (1 + 
            df['positive_flow'].rolling(14).sum() / 
            (df['negative_flow'].rolling(14).sum() + 1e-10)))

        # Forward Return für Label
        df['forward_return_21d'] = df['close'].pct_change(periods=21).shift(-21)

        # Auf Monatsende resampeln
        df_monthly = df.resample('BME').last()
        df_monthly['ts_code'] = stock
        all_rows.append(df_monthly)

    panel = pd.concat(all_rows).reset_index()
    panel = panel.rename(columns={'trade_date': 'date'})

    print(f"✅ Panel: {len(panel)} rows, "
          f"{panel['ts_code'].nunique()} stocks, "
          f"{panel['date'].nunique()} months")
    return panel


def merge_factors(panel: pd.DataFrame) -> pd.DataFrame:
    """
    Merged PE, Turnover (monatlich) und ROE (Snapshot) ins Panel.
    Berechnet Earnings Yield und Z-Score Normalisierung.
    Berechnet Composite Score.
    """
    # Fundamentals monatlich laden (PE, Turnover)
    fundamentals = pd.read_csv('data/fundamentals_monthly.csv')
    fundamentals['trade_date'] = pd.to_datetime(
        fundamentals['trade_date'], format='%Y%m%d')
    fundamentals = fundamentals.rename(columns={'trade_date': 'date'})

    # ROE laden (Snapshot — akzeptabel da Quartalsbericht)
    # NEU
    roe = pd.read_csv('data/roe_historical.csv')[['ts_code', 'roe']]

    # Merge
    panel = panel.merge(
        fundamentals[['ts_code', 'date', 'pe_ttm', 'turnover_rate']],
        on=['ts_code', 'date'], how='left')
    panel = panel.merge(roe, on='ts_code', how='left')

    # Earnings Yield = 1/PE
    panel['pe_ttm'] = panel['pe_ttm'].replace(0, np.nan)
    panel.loc[panel['pe_ttm'] < 0, 'pe_ttm'] = np.nan
    panel['earnings_yield'] = 1 / panel['pe_ttm']

    # NaNs füllen
    for col in ['momentum', 'mfi', 'turnover_rate', 'roe', 'earnings_yield']:
        panel[col] = panel[col].fillna(0)

    print(f"✅ Faktoren gemergt")
    return panel


def create_labels(panel: pd.DataFrame) -> pd.DataFrame:
    """
    Erstellt binäre Labels cross-sectional pro Monat.
    Label 1 = forward_return über Median, Label 0 = darunter.
    """
    panel = panel.copy()
    panel['label'] = None

    for date, group in panel.groupby('date'):
        median = group['forward_return_21d'].median()
        panel.loc[group.index, 'label'] = (
            group['forward_return_21d'] > median).astype(int)

    panel['label'] = panel['label'].astype(int)

    print(f"✅ Labels erstellt:")
    print(f"   Label 1: {panel['label'].sum()} rows")
    print(f"   Label 0: {(panel['label'] == 0).sum()} rows")
    return panel


def clean_data(panel: pd.DataFrame) -> pd.DataFrame:
    """
    Entfernt Zeilen ohne Forward Return (letzter Monat).
    """
    before = len(panel)
    panel = panel.dropna(subset=['forward_return_21d'])
    print(f"✅ Data Cleaning: {before} → {len(panel)} rows "
          f"({before - len(panel)} gedroppt)")
    return panel


def add_zscores(panel: pd.DataFrame) -> pd.DataFrame:
    """
    Z-Score Normalisierung cross-sectional pro Monat.
    Notwendig weil alle Faktoren verschiedene Skalen haben:
    ROE (~36), Momentum (~0.03), MFI (~50) etc.
    Z = (x - mean) / std → alle Faktoren auf gleiche Skala.
    Berechnet auch Composite Score mit gleichen Gewichten wie factors.py.
    """
    feature_cols = ['momentum', 'mfi', 'turnover_rate', 'roe', 'earnings_yield']

    for col in feature_cols:
        panel[f'z_{col}'] = None

    for date, group in panel.groupby('date'):
        for col in feature_cols:
            mean = group[col].mean()
            std  = group[col].std()
            if std > 0:
                panel.loc[group.index, f'z_{col}'] = (group[col] - mean) / std
            else:
                panel.loc[group.index, f'z_{col}'] = 0

    # Z-Score Columns zu float konvertieren
    for col in feature_cols:
        panel[f'z_{col}'] = panel[f'z_{col}'].astype(float)

    # Composite Score (identisch zu factors.py)
    panel['composite_score'] = (
        0.25 * panel['z_roe'] +
        0.25 * panel['z_earnings_yield'] +
        0.25 * panel['z_momentum'] +
        0.15 * panel['z_mfi'] -
        0.10 * panel['z_turnover_rate']
    )

    print(f"✅ Z-Scores und Composite Score berechnet")
    return panel


def time_based_split(panel: pd.DataFrame, split_date: str = '2023-07-01'):
    """
    Zeitbasierter Train/Test Split — kein Look-ahead Bias.
    Training: alles vor split_date
    Test: alles ab split_date
    Features: Z-Score normalisierte Faktoren + Composite Score
    """
    train = panel[panel['date'] < split_date]
    test  = panel[panel['date'] >= split_date]

    feature_cols = ['z_momentum', 'z_mfi', 'z_turnover_rate',
                    'z_roe', 'z_earnings_yield', 'composite_score']

    X_train = train[feature_cols]
    y_train = train['label']
    X_test  = test[feature_cols]
    y_test  = test['label']

    print(f"✅ Train/Test Split (cutoff: {split_date}):")
    print(f"   Training: {len(X_train)} rows | "
          f"{train['date'].min().date()} → {train['date'].max().date()}")
    print(f"   Test:     {len(X_test)} rows  | "
          f"{test['date'].min().date()} → {test['date'].max().date()}")

    return X_train, X_test, y_train, y_test, train, test

def save_pipeline_outputs(train: pd.DataFrame, test: pd.DataFrame,
                           output_dir: str = 'data'):
    os.makedirs(output_dir, exist_ok=True)
    train.to_csv(f'{output_dir}/train_data.csv', index=False)
    test.to_csv(f'{output_dir}/test_data.csv', index=False)
    print(f"✅ Gespeichert: train_data.csv ({len(train)} rows), "
          f"test_data.csv ({len(test)} rows)")
    
# Test
if __name__ == "__main__":
    prices = pd.read_csv('data/price_data.csv')
    panel = build_monthly_panel(prices)
    panel = merge_factors(panel)
    panel = create_labels(panel)
    panel = clean_data(panel)
    panel = add_zscores(panel)
    X_train, X_test, y_train, y_test, train, test = time_based_split(panel)
    print(panel[['ts_code', 'z_momentum', 'z_roe', 'z_earnings_yield',
                 'z_mfi', 'composite_score', 'label']].head(10))
    save_pipeline_outputs(train, test)