import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

data = pd.read_csv('student-mat.csv')
data['G1'] = (data['G1'] - data[['G1', 'G2', 'G3']].min().min()) / (data[['G1', 'G2', 'G3']].max().max() - data[['G1', 'G2', 'G3']].min().max())
data['G2'] = (data['G2'] - data[['G1', 'G2', 'G3']].min().min()) / (data[['G1', 'G2', 'G3']].max().max() - data[['G1', 'G2', 'G3']].min().max())
data['G3'] = (data['G3'] - data[['G1', 'G2', 'G3']].min().min()) / (data[['G1', 'G2', 'G3']].max().max() - data[['G1', 'G2', 'G3']].min().max())
data_copy = data.copy()
data['school'] = pd.Categorical(data['school']).codes
data['sex'] = pd.Categorical(data['sex']).codes
data['age'] = data['age'] - data['age'].min()
data['address'] = pd.Categorical(data['address']).codes
data['famsize'] = pd.Categorical(data['famsize']).codes
data['Pstatus'] = pd.Categorical(data['Pstatus']).codes
data['Medu'] = data['Medu'] - data['Medu'].min()
data['Fedu'] = data['Fedu'] - data['Fedu'].min()
data['Mjob'] = pd.Categorical(data['Mjob']).codes
data['Fjob'] = pd.Categorical(data['Fjob']).codes
data['reason'] = pd.Categorical(data['reason']).codes
data['guardian'] = pd.Categorical(data['guardian']).codes
data['traveltime'] = data['traveltime'] - data['traveltime'].min()
data['studytime'] = data['studytime'] - data['studytime'].min()
data['schoolsup'] = pd.Categorical(data['schoolsup']).codes
data['famsup'] = pd.Categorical(data['famsup']).codes
data['paid'] = pd.Categorical(data['paid']).codes
data['activities'] = pd.Categorical(data['activities']).codes
data['nursery'] = pd.Categorical(data['nursery']).codes
data['higher'] = pd.Categorical(data['higher']).codes
data['internet'] = pd.Categorical(data['internet']).codes
data['romantic'] = pd.Categorical(data['romantic']).codes
data['famrel'] = data['famrel'] - data['famrel'].min()
data['freetime'] = data['freetime'] - data['freetime'].min()
data['goout'] = data['goout'] - data['goout'].min()
data['Dalc'] = data['Dalc'] - data['Dalc'].min()
data['Walc'] = data['Walc'] - data['Walc'].min()
data['health'] = data['health'] - data['health'].min()

columns = ['sex', 'school', 'address', 'famsize', 'Pstatus', 'Medu', 'Fedu', 'Mjob', 'Fjob', 'reason', 'guardian', 'schoolsup', 'famsup', 'paid', 'activities', 'nursery', 'higher', 'internet', 'romantic']

ct = make_column_transformer((OneHotEncoder(), columns), remainder = 'passthrough')
data_copy = ct.fit_transform(data_copy)

x = data.iloc[:, :-3]
x = np.asarray(x).astype('float32')
y = data.iloc[:, -3:]
y = np.asarray(y).astype('float32')
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=0)

ann = tf.keras.models.Sequential()
ann.add(tf.keras.layers.Dense(units=16, activation='relu'))
ann.add(tf.keras.layers.Dense(units=16, activation='relu'))
ann.add(tf.keras.layers.Dense(units=3, activation='softmax'))
ann.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
history = ann.fit(x_train, y_train, batch_size=32, epochs=1000)

y_pred = ann.predict(x_test)
mse = tf.keras.losses.BinaryCrossentropy()
mse(y_test, y_pred).numpy()

plt.plot(history.history['loss'], label='loss')
plt.ylim([0.5, 0.8])
plt.xlabel('Epoch')
plt.ylabel('Error')
plt.legend()
plt.grid(True)
plt.show()

x = data_copy[:, :-3]
x = np.asarray(x).astype('float32')
y = data_copy[:, -3:]
y = np.asarray(y).astype('float32')
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=0)

ann = tf.keras.models.Sequential()
ann.add(tf.keras.layers.Dense(units=16, activation='softmax'))
ann.add(tf.keras.layers.Dense(units=16, activation='softmax'))
ann.add(tf.keras.layers.Dense(units=3, activation='softmax'))
ann.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
history = ann.fit(x_train, y_train, batch_size=32, epochs=1000)

y_pred = ann.predict(x_test)
mse = tf.keras.losses.BinaryCrossentropy()
mse(y_test, y_pred).numpy()

plt.plot(history.history['loss'], label='loss')
plt.ylim([0.5, 0.8])
plt.xlabel('Epoch')
plt.ylabel('Error')
plt.legend()
plt.grid(True)
plt.show()

rf = RandomForestRegressor(n_estimators=10, random_state=0)
rf.fit(x_train, y_train)
y_pred = rf.predict(x_test)

r2 = r2_score(y_test, y_pred)
print(r2)
