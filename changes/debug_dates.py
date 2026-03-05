import pandas as pd

# Load data
csv_path = r"C:\Users\muthumaniraj\OneDrive\Documents\me research\PV_Research_chat_recheck\PV_ALL_DATA_MASTER.csv"
print(f"Reading {csv_path}...")

try:
    df = pd.read_csv(csv_path)
    print("CSV loaded successfully.")
except Exception as e:
    print(f"Failed to load CSV: {e}")
    exit()

#  Check Date column
print("Checking 'Date' column...")
# Try parsing with the expected format
try:
    # First, let's just see unique lengths or patterns
    print("Sample dates head:", df['Date'].head().tolist())
    print("Sample dates tail:", df['Date'].tail().tolist())
    
    # Attempt conversion and find failures
    df['Date_Parsed'] = pd.to_datetime(df['Date'], format='%Y-%m-%d', errors='coerce')
    
    # Find rows where Date_Parsed is NaT but Date was not NaN
    failed_rows = df[df['Date_Parsed'].isna() & df['Date'].notna()]
    
    if not failed_rows.empty:
        print(f"Found {len(failed_rows)} rows with unparseable dates.")
        print("Examples of bad dates:")
        print(failed_rows['Date'].head(10))
        print("Indices:", failed_rows.index.tolist()[:10])
    else:
        print("All dates parsed successfully with '%Y-%m-%d'.")

except Exception as e:
    print(f"Error during analysis: {e}")
