import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load dataset
data = pd.read_csv(r'C:\Users\impashree\Desktop\pro\Air (2).csv')

# Data preprocessing: Convert air quality to binary labels ('good' or 'bad')
data['air_quality'] = ['good' if air_quality < 500 else 'bad' for air_quality in data['field3']]

# Visualize the distribution of the target variable
sns.countplot(x='air_quality', data=data, hue='air_quality', palette='pastel', dodge=False, legend=False)
plt.title('Air Quality Distribution')
plt.show()

# Line graph for Temperature (field1), Humidity (field2), and Gas Level (field3)
plt.figure(figsize=(10, 6))
plt.plot(data['field1'], label='Temperature', color='blue')
plt.plot(data['field2'], label='Humidity', color='green')
plt.plot(data['field3'], label='Gas Level', color='red')
plt.title('Temperature, Humidity, and Gas Level Over Time')
plt.xlabel('Index')
plt.ylabel('Values')
plt.legend()
plt.show()

# Count good and bad air quality
good_air_count = sum(data['air_quality'] == 'good')
bad_air_count = sum(data['air_quality'] == 'bad')
print(f"Good Air Quality Count: {good_air_count}%")
print(f"Bad Air Quality Count: {bad_air_count}%")

# Prepare features and targets
X = data[['field1', 'field2', 'field3']].fillna(0)
label_encoder = LabelEncoder()
y_classification = label_encoder.fit_transform(data['air_quality'])

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y_classification, test_size=0.2, random_state=42)

# Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train Logistic Regression
logistic_model = LogisticRegression(max_iter=1000)
logistic_model.fit(X_train_scaled, y_train)
y_pred_logistic = logistic_model.predict(X_test_scaled)

# Train Random Forest
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train_scaled, y_train)
y_pred_rf = rf_model.predict(X_test_scaled)

# Evaluate models
accuracy_logistic = accuracy_score(y_test, y_pred_logistic)
accuracy_rf = accuracy_score(y_test, y_pred_rf)

# Print evaluations
print(f"Logistic Regression Accuracy: {accuracy_logistic:.2f}")
print(f"Random Forest Accuracy: {accuracy_rf:.2f}")

# Visualize model performance
models = ['Logistic Regression', 'Random Forest']
accuracies = [accuracy_logistic, accuracy_rf]

plt.figure(figsize=(8, 5))
plt.bar(models, accuracies, color=['lightgreen', 'royalblue'])
plt.title('Comparison of Model Performance')
plt.ylabel('Accuracy')
plt.ylim(0, 1)
for i, v in enumerate(accuracies):
    plt.text(i, v + 0.01, f"{v:.2f}", ha='center')

plt.show()

