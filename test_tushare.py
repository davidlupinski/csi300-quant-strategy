import tushare as ts

ts.set_token('DEIN_TOKEN_HIER')  # Token aus .env laden
pro = ts.pro_api()

# Test 1: CSI 300 Constituents
df_index = pro.index_weight(
    index_code='399300.SZ', 
    trade_date='20240102'
)
print("✅ CSI 300 Constituents:")
print(df_index.head())
print(f"Anzahl Aktien: {len(df_index)}")
print()

# Test 2: Fundamentaldaten (ROA, PE, Turnover)
df_basic = pro.daily_basic(
    ts_code='000001.SZ',
    start_date='20240101',
    end_date='20240131'
)
print("✅ Fundamentaldaten Ping An Bank:")
print(df_basic.head())