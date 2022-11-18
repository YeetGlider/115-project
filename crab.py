import pandas as pd
import plotly.express as px

df = pd.read_csv("escape_velocity.csv")

Velocity_list = df["Velocity"].tolist()
Escaped_list = df["Escaped"].tolist()

fig = px.scatter(x=Velocity_list, y=Escaped_list)
fig.show()

import numpy as np
Velocity_array = np.array(Velocity_list)
Escaped_array = np.array(Escaped_list)

#Slope and intercept using pre-built function of Numpy
m, c = np.polyfit(Velocity_array, Escaped_array, 1)

y = []
for x in Velocity_array:
  y_value = m*x + c
  y.append(y_value)

#plotting the graph
fig = px.scatter(x=Velocity_array, y=Escaped_array)
fig.update_layout(shapes=[
    dict(
      type= 'line',
      y0= min(y), y1= max(y),
      x0= min(Velocity_array), x1= max(Velocity_array)
    )
])
fig.show()

import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

X = np.reshape(Velocity_list, (len(Velocity_list), 1))
Y = np.reshape(Escaped_list, (len(Escaped_list), 1))

lr = LogisticRegression()
lr.fit(X, Y)

plt.figure()
plt.scatter(X.ravel(), Y, color='black', zorder=20)

def model(x):
  return 1 / (1 + np.exp(-x))

#Using the line formula 
X_test = np.linspace(0, 5000, 10000)
Escaped_chances = model(X_test * lr.coef_ + lr.intercept_).ravel()

plt.plot(X_test, melting_chances, color='red', linewidth=3)
plt.axhline(y=0, color='k', linestyle='-')
plt.axhline(y=1, color='k', linestyle='-')
plt.axhline(y=0.5, color='b', linestyle='--')

#do hit and trial by changing the vlaue of X_test here.
plt.axvline(x=X_test[6843], color='b', linestyle='--')

plt.ylabel('y')
plt.xlabel('X')
plt.xlim(3400, 3450)
plt.show()
print(X_test[6843])