#%%

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression


#%%

X = [[2200,15],[2750,20],[5000,40],[4000,20],[3300,20],[2000,10],[2500,12],[12000,80],
[2880,10],[2300,15],[1500,10],[3000,8],[2000,14],[2000,10],[2150,8],[3400,20],
[5000,20],[4000,10],[3300,15],[2000,12],[2500,14],[10000,100],[3150,10],
[2950,15],[1500,5],[3000,18],[8000,12],[2220,14],[6000,100],[3050,10]]

y = [1,1,0,0,1,1,1,1,0,1,1,0,1,1,0,1,0,0,0,1,1,1,0,1,0,1,0,1,1,0]


#%%

ss = StandardScaler()

X_train = ss.fit_transform(X)

print(X_train)


#%%

lr = LogisticRegression()
lr.fit(X_train, y)


#%%

testX = [[2000, 8]]
X_test = ss.transform(testX)

print("Value to be predicted: ", X_test)

label = lr.predict(X_test)
print("predicted label = ", label)

prob = lr.predict_proba(X_test)
print("probability = ", prob)

#%%













