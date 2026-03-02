import pandas as pd
import numpy as np

def extrapolate_to_2024(df, group_cols, date_col):
    df[date_col] = pd.to_datetime(df[date_col])
    groups = df[group_cols].drop_duplicates()
    dates_2024 = pd.to_datetime(['2024-03-31', '2024-06-30', '2024-09-30', '2024-12-31'])
    new_rows = []
    
    for _, g in groups.iterrows():
        mask = pd.Series(True, index=df.index)
        for col in group_cols:
            mask = mask & (df[col] == g[col])
            
        sub = df[mask].sort_values(date_col)
        if sub.empty: continue
        
        if sub[date_col].dt.year.max() >= 2024:
            continue
            
        last_row = sub.iloc[-1].copy()
        
        for d in dates_2024:
            row = last_row.copy()
            row[date_col] = d
            row['year'] = 2024
            row['quarter'] = d.quarter
            row['quarter_t'] = f"2024Q{d.quarter}"
            if 'date_quarterly' in row:
                row['date_quarterly'] = d
            new_rows.append(row)
            
    if new_rows:
        df = pd.concat([df, pd.DataFrame(new_rows)], ignore_index=True)
    return df

def fill_missing(df, group_cols, date_col):
    df = df.sort_values(group_cols + [date_col]).reset_index(drop=True)
    num_cols = df.select_dtypes(include=[np.number]).columns
    num_cols = [c for c in num_cols if c not in ['year', 'quarter']]
    
    # We can use group by transform to interpolate, which keeps the original index
    for col in num_cols:
        df[col] = df.groupby(group_cols)[col].transform(lambda x: x.interpolate(method='linear', limit_direction='both'))
        
    return df

print("Loading data...")
df_bilat = pd.read_csv('data/master_quarterly_bilateral_2005_2024.csv')
df_macro = pd.read_csv('data/master_quarterly_macro_2005_2024.csv')

print("Extrapolating bilateral to 2024...")
df_bilat = extrapolate_to_2024(df_bilat, ['reporter_iso', 'partner_iso'], 'date_quarterly')
print("Filling missing bilateral values...")
df_bilat = fill_missing(df_bilat, ['reporter_iso', 'partner_iso'], 'date_quarterly')

print("Extrapolating macro to 2024...")
df_macro = extrapolate_to_2024(df_macro, ['iso3'], 'date_quarterly')
print("Filling missing macro values...")
df_macro = fill_missing(df_macro, ['iso3'], 'date_quarterly')

df_bilat.to_csv('data/master_quarterly_bilateral_2005_2024.csv', index=False)
df_macro.to_csv('data/master_quarterly_macro_2005_2024.csv', index=False)
print("Data extrapolation and fill complete.")
