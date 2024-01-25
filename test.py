import numpy as np
import pandas as pd
import seaborn as sns
import os
import matplotlib.pyplot as plt
import pprint
import sklearn
# %matplotlib inline
# /Users/daisyrheadharwadkar/Downloads/silon.csv
path = "/Users/daisyrheadharwadkar/Downloads/silon.csv"
df= pd.read_csv(path)

dataset=df
dataset["Month"]=pd.to_datetime(df["Date and hour"]).dt.month
dataset["Year"]=pd.to_datetime(df["Date and hour"]).dt.year
dataset["Date"]=pd.to_datetime(df["Date and hour"]).dt.date
dataset["Time"]=pd.to_datetime(df["Date and hour"]).dt.time
dataset["Day"]=pd.to_datetime(df["Date and hour"]).dt.day_name()
dataset=df.set_index("Date and hour")
dataset.index=pd.to_datetime(dataset.index)
# print(dataset.head(5))
###
###
###
TestData = dataset.tail(100)

Training_Set = dataset.iloc[:,0:1]

Training_Set = Training_Set[:-60]

# print(TestData)
from sklearn.preprocessing import MinMaxScaler

Training_Set = Training_Set
sc = MinMaxScaler(feature_range=(0, 1))
Train = sc.fit_transform(Training_Set)
# print(Train)
###
###
###
###
X_Train = []
Y_Train = []

# Range should be fromm 60 Values to END 
for i in range(60, Train.shape[0]):
    
    # X_Train 0-59 
    X_Train.append(Train[i-60:i])
    
    # Y Would be 60 th Value based on past 60 Values 
    Y_Train.append(Train[i])

# Convert into Numpy Array
X_Train = np.array(X_Train)
Y_Train = np.array(Y_Train)

# print(X_Train.shape)
# print(Y_Train.shape)
# print(len(X_Train))
X_Train = np.reshape(X_Train, newshape=(X_Train.shape[0], X_Train.shape[1], 1))
# print(len(X_Train[0]))

from keras.models import Sequential
from keras.layers import Dense,LSTM
from tensorflow.keras.layers import Dense, Dropout

input_shape = (X_Train.shape[1], 1)
# print(input_shape)
# print(X_Train[1])
# print(Y_Train[1])
regressor = Sequential()

# # Adding the first LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 60, return_sequences = True, input_shape = (X_Train.shape[1], 1)))
regressor.add(Dropout(0.2))

# # Adding a second LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 60, return_sequences = True))
regressor.add(Dropout(0.2))

# Adding a third LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 60, return_sequences = True))
regressor.add(Dropout(0.2))

# Adding a fourth LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 60))
regressor.add(Dropout(0.2))

# Adding the output layer
regressor.add(Dense(units = 1))

# Compiling the RNN
regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')


regressor.fit(X_Train, Y_Train, epochs = 10, batch_size = 32)

Df_Total = pd.concat((dataset[["Avg Avg Wind Speed @ 80m [m/s]"]], TestData[["Avg Avg Wind Speed @ 80m [m/s]"]]), axis=0)

print(Df_Total)
inputs = Df_Total[len(Df_Total) - len(TestData) - 60:].values
print(inputs.shape)
# We need to Reshape
inputs = inputs.reshape(-1,1)

# Normalize the Dataset
inputs = sc.transform(inputs)

X_test = []
for i in range(60, 160):
    X_test.append(inputs[i-60:i])
    
# Convert into Numpy Array
X_test = np.array(X_test)

# Reshape before Passing to Network
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

# Pass to Model 
predicted_speed= regressor.predict(X_test)

# Do inverse Transformation to get Values 
predicted_speed = sc.inverse_transform(predicted_speed)

True_Speed = TestData["Avg Avg Wind Speed @ 80m [m/s]"].to_list()
Predicted_Speed  = predicted_speed
dates = TestData.index.to_list()

Machine_Df = pd.DataFrame(data={
    "Date":dates,
    "TrueSpeed": True_Speed,
    "PredictedSpeed":[x[0] for x in Predicted_Speed ]
})

print(Machine_Df)

True_Speed= TestData["Avg Avg Wind Speed @ 80m [m/s]"].to_list()
Predicted_Speed  = [x[0] for x in Predicted_Speed ]
dates = TestData.index.to_list()


fig = plt.figure()

ax1= fig.add_subplot(111)

x = dates
y = True_Speed

y1 = Predicted_Speed

plt.plot(x,y, color="green")
plt.plot(x,y1, color="red")
# beautify the x-labels
plt.gcf().autofmt_xdate()
# plt.xlabel('Dates')
# plt.ylabel("Speed ")
# plt.title("Machine Learned the Pattern Predicting Future Values ")
# plt.legend()
plt.show()