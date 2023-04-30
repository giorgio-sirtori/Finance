from sklearn.linear_model import LinearRegression
import joblib
import numpy as np

# create some example data
X = np.array([1, 2, 3, 4, 5]).reshape(-1, 1)
y = np.array([2, 4, 6, 8, 10])

# train the model
model = LinearRegression()
model.fit(X, y)

# save the model to a file
joblib.dump(model, 'model.pkl')



