# data_processing/pipeline.py

import pandas as pd
import numpy as np
import os


def build_monthly_panel(price_data: pd.DataFrame) -> pd.DataFrame:

    all_rows = []

    for stock in price_data['ts_code'].unique():

        df = price_data[price_data['ts_code'] == stock].copy()
        df = df.sort_values('trade_date').reset_index(drop=True)

        df['trade_date'] = pd.to_datetime(df['trade_date'], format='%Y%m%d')
        df = df.set_index('trade_date')

        df['daily_return'] = df['close'].pct_change()
        df['momentum'] = df['close'].pct_change(periods=20)
        df['forward_return_21d'] = df['close'].pct_change(periods=21).shift(-21)

        df_monthly = df.resample('BME').last()
        df_monthly['ts_code'] = stock
        all_rows.append(df_monthly)

    panel = pd.concat(all_rows).reset_index()
    panel = panel.rename(columns={'trade_date': 'date'})

    print(f"✅ Panel: {len(panel)} rows, "
          f"{panel['ts_code'].nunique()} stocks, "
          f"{panel['date'].nunique()} months")
    return panel


def create_labels(panel: pd.DataFrame) -> pd.DataFrame:
    # Label 1 = forward_return über Median dieses Monats
    # Label 0 = forward_return unter Median

    def label_month(group):
        median = group['forward_return_21d'].median()
        group = group.copy()
        group['label'] = (group['forward_return_21d'] > median).astype(int)
        return group

    # date als normale Spalte behalten, nicht als Index verwenden
    panel = panel.copy()
    panel['label'] = None

    for date, group in panel.groupby('date'):
        median = group['forward_return_21d'].median()
        panel.loc[group.index, 'label'] = (group['forward_return_21d'] > median).astype(int)

    panel['label'] = panel['label'].astype(int)

    print(f"✅ Labels erstellt:")
    print(f"   Label 1: {panel['label'].sum()} rows")
    print(f"   Label 0: {(panel['label'] == 0).sum()} rows")

    return panel


def clean_data(panel: pd.DataFrame) -> pd.DataFrame:
    # Zeilen ohne Label droppen (letzter Monat — keine Zukunft vorhanden)
    before = len(panel)
    panel = panel.dropna(subset=['forward_return_21d'])
    dropped = before - len(panel)

    # Feature NaNs mit 0 füllen (neutrales Signal)
    panel['momentum'] = panel['momentum'].fillna(0)

    print(f"✅ Data Cleaning:")
    print(f"   Rows vorher: {before}")
    print(f"   Rows gedroppt: {dropped}")
    print(f"   Rows nachher: {len(panel)}")
    print(f"   Verbleibende NaNs: {panel[['momentum', 'forward_return_21d', 'label']].isna().sum().sum()}")

    return panel


# Test
if __name__ == "__main__":
    prices = pd.read_csv('data/price_data.csv')
    panel = build_monthly_panel(prices)
    panel = create_labels(panel)
    panel = clean_data(panel)
    print(panel[['ts_code', 'date', 'forward_return_21d', 'label']].head(12))