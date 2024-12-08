import yfinance as yf
import pandas as pd
from boltons.iterutils import first
import pandas as pd
import statsmodels.api as sm



def getActivelyTradedstocks():
    file_path = "../data/oneTimeData/ActiveStocks29_12_23.csv"
    data = pd.read_csv(file_path, skiprows=1)
    fifth_column = data.iloc[:, 0]
    fifth_column.to_csv("../data/StockPERMNOS.csv", index=False)




def load_CRSP_data():
    file_path = "../data/CRSP_STOCK_INFO.csv"
    df = pd.read_csv(file_path)
    df["m_cap"] = abs(df["PRC"])*df["SHROUT"]*1000
    return df

def load_COMPUSTAT_data():
    file_path = "../data/COMPUSTAT_STOCK_INFO.csv"
    df = pd.read_csv(file_path)
    df['LINKENDDT'] = df['LINKENDDT'].replace('E', 99991231).astype(
        int)  # Treat 'E' as a far-future date
    return df

def merge_COMPUSTAT_CRSP():
    CRSP_data = load_CRSP_data()
    COMPUSTAT_data = load_COMPUSTAT_data()

    # Create a dictionary to group COMPUSTAT data by LPERMNO for faster filtering
    compustat_dict = {
        permno: group for permno, group in COMPUSTAT_data.groupby('LPERMNO')
    }

    # Group CRSP_data by PERMNO for chunked processing
    CRSP_groups = CRSP_data.groupby('PERMNO')

    output_chunks = []
    for permno, crsp_chunk in CRSP_groups:
        # Retrieve the corresponding COMPUSTAT subset for this PERMNO
        compustat_subset = compustat_dict.get(permno, pd.DataFrame())

        # Skip processing if no matching COMPUSTAT data
        if compustat_subset.empty:
            continue

        # Merge the CRSP chunk with the relevant COMPUSTAT subset
        merged = crsp_chunk.merge(compustat_subset, left_on='PERMNO', right_on='LPERMNO', how='left')

        # Filter rows where the daily date falls within the valid link date range
        valid_links = (merged['date'] >= merged['LINKDT']) & (merged['date'] <= merged['LINKENDDT'])
        merged = merged[valid_links]

        # Filter for the most recent datadate before or equal to the daily date
        merged = merged.sort_values(by=['PERMNO', 'date', 'datadate'])
        merged = merged[merged['datadate'] <= merged['date']]
        merged = merged.loc[merged.groupby(['PERMNO', 'date'])['datadate'].idxmax()]

        # Select and rename columns
        filtered = merged[['date', 'PERMNO', 'TICKER', 'm_cap', 'ceq', 'RET']].rename(
            columns={'ceq': 'book_value'})

        # Filter out rows with NaN book_value
        filtered = filtered.dropna(subset=['book_value'])

        # Add the processed chunk to the output list
        output_chunks.append(filtered)

    # Combine all processed chunks into a single DataFrame
    output = pd.concat(output_chunks).reset_index(drop=True)
    return output

def load_FF_data():
    file_path = '../data/FF_DAILY_FACTORS.csv'
    # Load the CSV, skipping rows and naming columns
    df = pd.read_csv(file_path, skiprows=3, nrows=25817)
    df.rename(columns={df.columns[0]: "date"}, inplace=True)
    return df[["date", "Mkt-RF", "RF"]]


def merge_FF_COMPUSTAT_CRSP():
    CRSP_COMP_merged = merge_COMPUSTAT_CRSP()
    FF_data = load_FF_data()
    # Merge the two DataFrames on the 'date' column
    merged_data = pd.merge(CRSP_COMP_merged, FF_data, on="date", how="left")
    return merged_data

'''
df = merge_FF_COMPUSTAT_CRSP()
df.loc[:, 'RET'] = pd.to_numeric(df['RET'], errors='coerce')
df = df.dropna(subset=['RET'])
df['RET'] = df['RET'].astype(float)
df['XR'] = df['RET'] - df['RF']
df["book_value"] = df["book_value"]*1000000
df.to_csv("../data/Neural_Network_Data.csv", index=False)
print(df.head())'''