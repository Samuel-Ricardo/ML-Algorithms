# CODE IMPLEMETNATION

# %%

from sklearn.linear_model import LinearRegression

import matplotlib.pyplot as plt
import numpy as np 

# %%

x = np.array([121, 125,131, 141, 152, 161]).reshape(-1,1)
y = np.array([300, 350, 425, 405, 496, 517])

plt.scatter(x, y)
plt.xlabel("Area")
plt.ylabel("Price")

plt.show()


# %%

lr = LinearRegression()
lr.fit(x, y)


#%%

w = lr.coef_
b = lr.intercept_

print('Slope: ', w)
print('Intercept: ', b)


#%%

plt.scatter(x, y)

plt.xlabel("Area")
plt.ylabel("Price")

plt.plot([x[0], x[-1]], [x[0]*w + b, x[-1]*w  + b])

plt.show()


# %%

testX = np.array([[130]])
lr.predict(testX)


# %%

