import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, PowerTransformer
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from neuralNetworkClass import neuralNetwork
from fama_french_analysis import *
import statsmodels.api as sm


# Load dataset
df = pd.read_csv('data/Neural_Network_Data.csv')

#Final feature engineering
df["BtoM"] = df["book_value"] / df["m_cap"] #Extracting BtoM
df['log_m_cap'] = np.log(df['m_cap']) #Transforming M_cap using log, to make it more normally distributed. Market Cap typically spans many orders of magnitude.


df = df[df['BtoM'] > 0] #Drop negative BtoM ratios, there are a couple, due to some data with negative Book Values, 
df['log_BtoM'] = np.log(df['BtoM']) #Log transforming BtoM also, as it is again very skewed

#Splitting on our date for training and test data
train_data = df[df['date'] <= 20191231]
test_data = df[df['date'] > 20191231]


# Define features and target
features = ['log_m_cap', 'Mkt-RF', 'log_BtoM']
target = 'XR'

X_train = train_data[features].values
y_train = train_data[target].values
X_test = test_data[features].values
y_test = test_data[target].values

# Initialize the model
model = neuralNetwork(input_dim=X_train.shape[1])

# Scaling training features to between 0 and 1
X_train_scaled, X_test_scaled = model.preprocess_data(X_train, X_test)

#Scaling targets
#Yeo_Johnsen tranformation to reduce spread:
yeo_johnson_transformer = PowerTransformer(method='yeo-johnson')
y_train_transformed = yeo_johnson_transformer.fit_transform(y_train.reshape(-1, 1)).flatten()
y_test_transformed = yeo_johnson_transformer.transform(y_test.reshape(-1, 1)).flatten()

#MinMax Scaling to between 0 and 1, for easier analysis by NN
scaler = MinMaxScaler()
y_train_scaled = scaler.fit_transform(y_train_transformed.reshape(-1, 1)).flatten()
y_test_scaled = scaler.transform(y_test_transformed.reshape(-1, 1)).flatten()


# Training the model
history = model.train(X_train_scaled, y_train_scaled, validation_split=0.2, epochs=50, batch_size=64)

# Predict on test data, then unscaling and untransforming it to obtain "proper" predictions
predictions = model.predict(X_test_scaled)
predictions_unscaled = scaler.inverse_transform(predictions.reshape(-1, 1)).flatten()
predictions_original = yeo_johnson_transformer.inverse_transform(predictions_unscaled.reshape(-1, 1)).flatten()

#Add predictions to test_Frame
test_data['NN_Predictions'] = predictions_original
ff_predictions = pd.read_csv('data/predictions/fama_french_predictions.csv')


merged_df = pd.merge(
    test_data[['date', 'PERMNO', 'TICKER', 'XR', 'NN_Predictions']],  
    ff_predictions[['date', 'PERMNO', 'FF_prediction']],                
    on=['date', 'PERMNO'],                                    
    how='inner'                                             
)


test_data.to_csv('predictions_merged_solstormtest.csv', index=False)





