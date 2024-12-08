import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm


def load_FAMA_FRENCH_Factor_data():
    file_path = "data/FF_DAILY_FACTORS.csv"
    # Load the CSV, skipping rows and naming columns
    data = pd.read_csv(file_path, skiprows=3, nrows = 25817)
    data.rename(columns={data.columns[0]: "date"}, inplace=True)
    return data

def extract_fama_french_data_frames():
    #Load data from file
    file_path = "data/CRSP_STOCK_INFO.csv"
    df = pd.read_csv(file_path)
    #Group by permno
    grouped = df.groupby('PERMNO')

    # Extract company data one by one and store them to the list of company dataframes
    company_dataframes = []
    for permno, group in grouped:
        company_df = group.reset_index(drop=True)  # Reset index for clarity
        company_dataframes.append(company_df)  # Save the company's DataFrame to a list
    return company_dataframes
    


def famaFrenchDataMerge(stock_data, split_date):
    FFdata = load_FAMA_FRENCH_Factor_data()
    mergedData = pd.merge(stock_data, FFdata, how='inner', left_on='date', right_on='date')
    mergedData.loc[:, 'RET'] = pd.to_numeric(mergedData['RET'], errors='coerce')
    mergedData = mergedData.dropna(subset=['RET'])
    mergedData['RET'] = mergedData['RET'].astype(float)
    mergedData['XR'] = mergedData['RET'] - mergedData['RF']
    full_final_df = mergedData[['PERMNO','TICKER','date', 'Mkt-RF', 'SMB', 'HML', 'XR']]
    # Filter rows based on the numeric or string date
    final_df_train = full_final_df[full_final_df['date'] <= split_date]
    final_df_test = full_final_df[full_final_df['date'] > split_date]
    return final_df_train, final_df_test


def ff_train(data):
    X = data[['Mkt-RF', 'SMB', 'HML']]
    X = sm.add_constant(X)
    y = data['XR']
    ff_model = sm.OLS(y, X).fit()
    return(ff_model)

def ff_test(data, model):
    X = data[['Mkt-RF', 'SMB', 'HML']]
    X = sm.add_constant(X)
    prediction = model.predict(X)
    return prediction

'''stockdata = extract_fama_french_data_frames()
print(stockdata[0].head())'''