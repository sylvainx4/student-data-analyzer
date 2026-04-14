import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

data = pd.read_csv("data/student_data.csv")
print("Dataset loaded:", data.shape)

X = data[["G1", "G2"]]
y = data["G3"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

predictions = model.predict(X_test)

mse = mean_squared_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print("\nModel Evaluation:")
print(f"Mean Squared Error : {mse:.2f}")
print(f"R2 Score           : {r2:.2f}")
print("\nSample Predictions vs Actual:")
results = pd.DataFrame({"Actual G3": y_test.values, "Predicted G3": predictions.round(1)})
print(results.head(10).to_string(index=False))
