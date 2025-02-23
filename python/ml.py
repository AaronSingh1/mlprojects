import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import  datareader as data
from keras.models import load_model
import streamlit as st

start ='2010-01-01'
end='2019-12-31'

#st.title('Stock trend prediction')

#user_input=st.text_input('Enter Stock','AAPL')
df=data.DataReader("C:\\Users\\aaron\\OneDrive\\Desktop\\New folder\\python\\TSLA.csv",start,end)
df.head()

"""#describing data
st.subheader('data from 2010 -2019')
st.writr(df.describe())

st.subheader('Closing Price vs Time chart')
fig =plt.figure(figsize=(12,6))
plt.plot(df.Close)
st.pyplot(fig)

st.subheader('Closing Price vs Time chart with 100 MA')
ma100=df.Close.rolling(100).mean()
fig =plt.figure(figsize=(12,6))
plt.plot(ma100)
plt.plot(df.Close)
st.pyplot(fig)

st.subheader('Closing Price vs Time chart with 100 MA and 200 MA')
ma100=df.Close.rolling(100).mean()
ma200=df.Close.rolling(200).mean()
fig =plt.figure(figsize=(12,6))
plt.plot(ma100, 'r')
plt.plot(ma200, 'g')
plt.plot(df.Close, 'b')
st.pyplot(fig)

#splitting data into training and testing

data_training=pd.DataFrame(df['Close'][0:int(len(df)*0.70)])
data_training=pd.DataFrame(df['Close'][int(len(df)*0.70): int(len(df))])

from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler(feature_range=(0,1))

data_training_array=scaler.fit_transform(data_training)
#spikiting Data into x_train and y
x_traoin = []
y_train = []

for i in range(100,data_training_array.shape[0])

    x_train.append(data_training_array[i-100; i])
    y_train.append(data_training_array[i, 0])

x_train, y_train= np.array(x_train),np.array(y_train)

#load my model
model=load_model('keras_model.h5')

#Testing part

past_100_days = data_training.tail(100)
final_df=past_100days.append(data_testing, ignore _index= True)
input_data= scaler.fit_transform(final_df)

x_test = []
y_test = []

for i in range (100,input_data.shape[0]);
    x_test.append(input_data[i-100: i])
    y_test.apped(input_data[i , 0])

x_test, y_test = np.array(x_test), np.array(y_test)
y_predicted = model.predict(x+test)
scaler = scaler.scale_

scale_factor= 1/scaler[0]
y_predicted = y_predicted* scale_factory* scale_factor 
y_test = y_test*scale_factor

#final graph
st.subheader ('Predictins vs Original' )
fig2 = plt.figure(figsize=(12.6))
plt.plot(y_test, 'b' ,Label= 'Original Price')
plt.plot(y_predicted, 'r' ,Label = 'Predicted Price ')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
plt.show()"""