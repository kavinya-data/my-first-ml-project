import numpy as np
from sklearn.linear_model import LinearRegression

# Data (X = Time, Y = Marks)
X = np.array ([[1], [2], [4], [5], [6]])
Y = np.array ([[30], [45], [60], [75], [80]])

# Build and Train the Model
model = LinearRegression()
model.fit(X, Y)

# Prediction
time_to_predict = np.array ([[3]])
predicted_marks = model.predict(time_to_predict)
print(f"Score for someone who worked for 3 hours: {predicted_marks[0][0]:.2f}")
