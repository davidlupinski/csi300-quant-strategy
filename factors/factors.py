# ============================================================
# factors.py
# CSI 300 Multi-Factor Construction
# Author: David Lupinski | Xiamen University
# ============================================================

# --- Block 1: Imports & Setup ---
import tushare as ts
import pandas as pd
import numpy as np
from dotenv import load_dotenv
import os

# Token sicher aus .env laden
load_dotenv()
token = os.getenv('TUSHARE_TOKEN')
ts.set_token(token)
pro = ts.pro_api()

print("✅ Tushare verbunden!")

# --- Block 2: Load CSI 300 Constituents ---
def get_csi300_stocks(trade_date='20240102'):
    """
    Returns all CSI 300 stocks for a given trading date.
    trade_date: trading day in format YYYYMMDD
    """
    df = pro.index_weight(
        index_code='399300.SZ',
        trade_date=trade_date
    )
    # Extract stock codes as a list
    stocks = df['con_code'].tolist()
    print(f"✅ {len(stocks)} CSI 300 stocks loaded for {trade_date}")
    return stocks

# Test
stocks = get_csi300_stocks('20240102')
print(stocks[:5])

# --- Block 3: Load Daily Price Data ---
def get_price_data(stocks, start_date='20180101', end_date='20241231'):
    all_data = []
    
    for i, stock in enumerate(stocks):
        success = False
        retries = 3
        while not success and retries > 0:
            try:
                df = pro.daily(
                    ts_code=stock,
                    start_date=start_date,
                    end_date=end_date
                )
                all_data.append(df)
                success = True
                if (i + 1) % 10 == 0:
                    print(f"  Progress: {i+1}/{len(stocks)} stocks loaded")
            except Exception as e:
                retries -= 1
                print(f"  ⚠️ Retry {stock} ({retries} left): {e}")
                import time
                time.sleep(5)  # 5 Sekunden warten vor Retry
    
    price_data = pd.concat(all_data, ignore_index=True)
    print(f"✅ Price data loaded: {len(price_data)} rows")
    return price_data

# --- Block 4: Calculate Momentum Factor ---
def calculate_momentum(price_data, window=20):
    """
    Momentum = return over the last N trading days.
    Higher momentum → stock is trending upward → positive signal.
    window: number of trading days to look back (default: 20 = ~1 month)
    """
    momentum_list = []

    # Loop through each stock separately
    for stock in price_data['ts_code'].unique():

        # Filter data for this stock only
        df = price_data[price_data['ts_code'] == stock].copy()

        # Sort by date (oldest first)
        df = df.sort_values('trade_date')

        # Calculate momentum: (current price / price N days ago) - 1
        df['momentum'] = df['close'].pct_change(periods=window)

        # Keep only the most recent value per stock
        latest = df.iloc[-1][['ts_code', 'trade_date', 'momentum']]
        momentum_list.append(latest)

    # Combine all stocks into one DataFrame
    momentum_df = pd.DataFrame(momentum_list)
    momentum_df = momentum_df.dropna()  # Remove stocks with missing values

    print(f"✅ Momentum calculated for {len(momentum_df)} stocks")
    return momentum_df

# Test
#momentum = calculate_momentum(prices)
#print(momentum.head(10))

# --- Block 5: Load Fundamental Data (PE, PB, Turnover) ---
def get_fundamental_data(stocks, trade_date='20240103'):
    """
    Loads fundamental factors for all stocks one by one.
    - pe_ttm: Price-to-Earnings ratio
    - pb: Price-to-Book ratio  
    - turnover_rate: daily turnover rate
    """
    all_data = []

    for stock in stocks:  # Test with first 10 stocks
        df = pro.daily_basic(
            ts_code=stock,      # One stock at a time
            trade_date=trade_date,
            fields='ts_code,trade_date,pe_ttm,pb,turnover_rate'
        )
        if len(df) > 0:         # Only append if data exists
            all_data.append(df)

    # Combine all stocks
    fundamentals = pd.concat(all_data, ignore_index=True)
    print(f"✅ Fundamental data loaded: {len(fundamentals)} stocks")
    print(fundamentals.head(10))
    return fundamentals

# Test
#fundamentals = get_fundamental_data(stocks)

# --- Block 6: Load ROE (Profitability Factor) ---
def get_roe_data(stocks, period='20231231'):
    """
    Loads ROE (Return on Equity) for all stocks.
    ROE = Net Income / Shareholders Equity
    Higher ROE → more profitable → positive signal.
    period: latest financial reporting period (YYYYMMDD)
    """
    all_data = []

    for stock in stocks:  # Test with first 10 stocks
        df = pro.fina_indicator(
            ts_code=stock,
            period=period,
            fields='ts_code,ann_date,roe'
        )
        if len(df) > 0:
            # Keep only most recent entry
            latest = df.iloc[0:1]
            all_data.append(latest)

    # Combine all stocks
    roe_df = pd.concat(all_data, ignore_index=True)
    roe_df = roe_df.dropna(subset=['roe'])

    print(f"✅ ROE loaded for {len(roe_df)} stocks")
    print(roe_df.head(10))
    return roe_df

# Test
#roe_data = get_roe_data(stocks)
# --- Block 6b: Load Monthly Fundamentals (PE, Turnover) ---
def get_monthly_fundamentals(stocks, start_date='20180101', end_date='20241231'):
    FUND_FILE = 'data/fundamentals_monthly.csv'
    
    # Bereits geladene Daten prüfen
    if os.path.exists(FUND_FILE):
        existing = pd.read_csv(FUND_FILE)
        done_dates = existing['trade_date'].unique().tolist()
        print(f"📂 Bereits geladen: {len(done_dates)} Datumswerte — weiter ab da")
    else:
        existing = pd.DataFrame()
        done_dates = []
        os.makedirs('data', exist_ok=True)

    dates = pd.date_range(start=start_date, end=end_date, freq='BME')
    dates = [d.strftime('%Y%m%d') for d in dates]

    all_data = []
    import time

    for di, date in enumerate(dates):
        if int(date) in done_dates or date in done_dates:
            continue  # bereits geladen → überspringen

        for stock in stocks:
            success = False
            retries = 3
            while not success and retries > 0:
                try:
                    df = pro.daily_basic(
                        ts_code=stock,
                        trade_date=date,
                        fields='ts_code,trade_date,pe_ttm,turnover_rate'
                    )
                    if len(df) > 0:
                        all_data.append(df)
                    success = True
                except Exception as e:
                    retries -= 1
                    print(f"  ⚠️ Retry {stock} {date} ({retries} left): {e}")
                    time.sleep(5)

        # Nach jedem Monat speichern
        if all_data:
            temp = pd.concat([existing] + all_data, ignore_index=True)
            temp.to_csv(FUND_FILE, index=False)

        if (di + 1) % 10 == 0:
            print(f"  Progress: {di+1}/{len(dates)} months loaded")

    if all_data:
        fundamentals_monthly = pd.concat([existing] + all_data, ignore_index=True)
    else:
        fundamentals_monthly = existing

    print(f"✅ Monthly fundamentals: {len(fundamentals_monthly)} rows")
    return fundamentals_monthly

# --- Block 6c: Load Historical ROE (für pipeline.py) ---
def get_roe_historical(stocks, period='20171231'):
    """
    Lädt ROE aus dem Jahresabschluss Q4 2022.
    Wird in pipeline.py verwendet damit kein Look-ahead Bias entsteht:
    Wir trainieren auf 2023 Daten → brauchen ROE von VOR 2023.
    """
    all_data = []

    for stock in stocks:
        df = pro.fina_indicator(
            ts_code=stock,
            period=period,
            fields='ts_code,ann_date,roe'
        )
        if len(df) > 0:
            latest = df.iloc[0:1]
            all_data.append(latest)

    roe_historical = pd.concat(all_data, ignore_index=True)
    roe_historical = roe_historical.dropna(subset=['roe'])

    print(f"✅ Historical ROE loaded: {len(roe_historical)} stocks")
    return roe_historical
# --- Block 7: Calculate MFI (Money Flow Index) ---
def calculate_mfi(price_data, window=14):
    """
    Calculates Money Flow Index for all stocks.
    MFI combines price and volume to measure buying/selling pressure.
    window: lookback period in trading days (default: 14)
    
    Formula:
    1. Typical Price = (High + Low + Close) / 3
    2. Raw Money Flow = Typical Price x Volume
    3. MFI = 100 - (100 / (1 + Positive Flow / Negative Flow))
    """
    mfi_list = []

    for stock in price_data['ts_code'].unique():

        # Filter data for this stock only
        df = price_data[price_data['ts_code'] == stock].copy()

        # Sort by date oldest first
        df = df.sort_values('trade_date').reset_index(drop=True)

        # Step 1: Typical Price
        df['typical_price'] = (df['high'] + df['low'] + df['close']) / 3

        # Step 2: Raw Money Flow
        df['raw_money_flow'] = df['typical_price'] * df['vol']

        # Step 3: Positive and Negative Money Flow
        # Positive = typical price higher than previous day
        df['positive_flow'] = 0.0
        df['negative_flow'] = 0.0

        for i in range(1, len(df)):
            if df['typical_price'].iloc[i] > df['typical_price'].iloc[i-1]:
                df.loc[i, 'positive_flow'] = df['raw_money_flow'].iloc[i]
            else:
                df.loc[i, 'negative_flow'] = df['raw_money_flow'].iloc[i]

        # Step 4: Rolling sum over window period
        df['pos_flow_sum'] = df['positive_flow'].rolling(window=window).sum()
        df['neg_flow_sum'] = df['negative_flow'].rolling(window=window).sum()

        # Step 5: MFI calculation
        # Add small number to avoid division by zero
        df['mfi'] = 100 - (100 / (1 + df['pos_flow_sum'] / 
                                      (df['neg_flow_sum'] + 1e-10)))

        # Keep only the most recent value
        latest = df.iloc[-1][['ts_code', 'trade_date', 'mfi']]
        mfi_list.append(latest)

    # Combine all stocks
    mfi_df = pd.DataFrame(mfi_list).reset_index(drop=True)
    mfi_df = mfi_df.dropna()

    print(f"✅ MFI calculated for {len(mfi_df)} stocks")
    print(mfi_df.head(10))
    return mfi_df

# Test
#mfi_data = calculate_mfi(prices)

# --- Block 8: Combine All Factors + Composite Score ---
def calculate_composite_score(fundamentals, roe_data, 
                               momentum_data, mfi_data):
    """
    Merges all factors into one DataFrame and calculates composite score.
    
    Step 1: Merge all factor DataFrames on ts_code
    Step 2: Calculate Earnings Yield (1/PE)
    Step 3: Z-score normalize all factors cross-sectionally
    Step 4: Calculate composite score with weights
    
    Composite = 0.25 * Z(ROE) 
              + 0.25 * Z(EarningsYield) 
              + 0.25 * Z(Momentum) 
              + 0.15 * Z(MFI) 
              - 0.10 * Z(Turnover)
    """

    # Step 1: Merge all factors on stock code
    df = fundamentals[['ts_code', 'pe_ttm', 'turnover_rate']].copy()
    df = df.merge(roe_data[['ts_code', 'roe']], on='ts_code', how='inner')
    df = df.merge(momentum_data[['ts_code', 'momentum']], on='ts_code', how='inner')
    df = df.merge(mfi_data[['ts_code', 'mfi']], on='ts_code', how='inner')

    print(f"✅ Merged: {len(df)} stocks with all factors")

    # Step 2: Earnings Yield = 1/PE (higher = cheaper = better)
    # Replace negative or zero PE with NaN (negative PE = loss-making company)
    df['pe_ttm'] = df['pe_ttm'].replace(0, np.nan)
    df.loc[df['pe_ttm'] < 0, 'pe_ttm'] = np.nan
    df['earnings_yield'] = 1 / df['pe_ttm']

    # Step 3: Z-score normalization for each factor
    # Z = (value - mean) / std
    def zscore(series):
        return (series - series.mean()) / series.std()

    df['z_roe']            = zscore(df['roe'])
    df['z_earnings_yield'] = zscore(df['earnings_yield'])
    df['z_momentum']       = zscore(df['momentum'])
    df['z_mfi']            = zscore(df['mfi'])
    df['z_turnover']       = zscore(df['turnover_rate'])

    # Step 4: Composite Score
    # Turnover is negative signal → subtract it
    df['composite_score'] = (
        0.25 * df['z_roe'] +
        0.25 * df['z_earnings_yield'] +
        0.25 * df['z_momentum'] +
        0.15 * df['z_mfi'] -
        0.10 * df['z_turnover']
    )

    # Step 5: Rank stocks by composite score (highest = best)
    df = df.sort_values('composite_score', ascending=False)
    df['rank'] = range(1, len(df) + 1)

    print("\n🏆 Top 5 stocks by Composite Score:")
    print(df[['ts_code', 'roe', 'earnings_yield', 
              'momentum', 'mfi', 'composite_score', 'rank']].head())
    
    print("\n❌ Bottom 5 stocks:")
    print(df[['ts_code', 'roe', 'earnings_yield',
              'momentum', 'mfi', 'composite_score', 'rank']].tail())
    
    return df

# Test
#factor_df = calculate_composite_score(
#    fundamentals, roe_data, momentum, mfi_data
#)

# --- Block 9: Save Final Factor Data ---
def save_factor_data(factor_df, filename='data/factors_latest.csv'):
    """
    Saves the complete factor DataFrame to CSV.
    This avoids re-running all API calls every time.
    """
    # Create data folder if it doesn't exist
    os.makedirs('data', exist_ok=True)

    # Save to CSV
    factor_df.to_csv(filename, index=False)
    print(f"✅ Factor data saved to {filename}")
    print(f"   Stocks: {len(factor_df)}")
    print(f"   Columns: {factor_df.columns.tolist()}")

# --- MAIN RUN ---
# This block only runs when you execute factors.py directly
if __name__ == "__main__":
    print("=" * 50)
    print("CSI 300 FACTOR CONSTRUCTION")
    print("=" * 50)

    # Step 1: Get stock universe
    stocks = get_csi300_stocks('20240102')

    # Step 2: Load price data (from disk if available)
    DATA_FILE = 'data/price_data.csv'
    if os.path.exists(DATA_FILE):
        print("📂 Loading price data from local file...")
        prices = pd.read_csv(DATA_FILE)
    else:
        print("🌐 Fetching price data from Tushare...")
        os.makedirs('data', exist_ok=True)
        prices = get_price_data(stocks)
        prices.to_csv(DATA_FILE, index=False)

    # Step 2b: Load monthly fundamentals (PE, Turnover)
    FUND_FILE = 'data/fundamentals_monthly.csv'
    if os.path.exists(FUND_FILE):
        existing = pd.read_csv(FUND_FILE)
        print(f"📂 Fundamentals teilweise geladen: {len(existing)} Zeilen")
        print(f"   Letztes Datum: {existing['trade_date'].max()}")
        # Prüfen ob vollständig — sollte ~25000 Zeilen haben
        if len(existing) < 20000:
            print("⚠️  Nicht vollständig — lade weiter...")
            fundamentals_monthly = get_monthly_fundamentals(stocks)
        else:
            fundamentals_monthly = existing
            print("✅ Vollständig geladen")
    else:
        print("🌐 Fetching monthly fundamentals from Tushare...")
        fundamentals_monthly = get_monthly_fundamentals(stocks)
        fundamentals_monthly.to_csv(FUND_FILE, index=False)
        print(f"✅ Saved to {FUND_FILE}")

    # Step 2c: Load historical ROE (Q4 2022 — kein Look-ahead Bias)
    ROE_FILE = 'data/roe_historical.csv'
    if os.path.exists(ROE_FILE):
        print("📂 Loading historical ROE from local file...")
        roe_historical = pd.read_csv(ROE_FILE)   # ← Fix: Variable setzen
    else:
        print("🌐 Fetching historical ROE from Tushare...")
        roe_historical = get_roe_historical(stocks)
        roe_historical.to_csv(ROE_FILE, index=False)
        print(f"✅ Saved to {ROE_FILE}")

    # Step 3: Calculate all factors
    momentum    = calculate_momentum(prices)
    mfi         = calculate_mfi(prices)
    fundamentals = get_fundamental_data(stocks)
    roe_data    = get_roe_data(stocks)

    # Step 4: Combine into composite score
    factor_df = calculate_composite_score(
        fundamentals, roe_data, momentum, mfi
    )

    # Step 5: Save results
    save_factor_data(factor_df)

    print("\n" + "=" * 50)
    print("✅ FACTOR CONSTRUCTION COMPLETE!")
    print("=" * 50)
    print(f"\n🏆 Top 3 stocks to BUY:")
    print(factor_df[['ts_code', 'composite_score', 'rank']].head(3))
    print(f"\n❌ Bottom 3 stocks to AVOID:")
    print(factor_df[['ts_code', 'composite_score', 'rank']].tail(3))
