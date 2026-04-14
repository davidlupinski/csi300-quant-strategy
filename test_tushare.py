import tushare as ts

ts.set_token('5344f94c35f9803bd597342c178f61adecbca9d9a4cf958443bfb8a3')
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