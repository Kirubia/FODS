import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
data = {
    'age': [25, 30, 35, 40, 45, 28, 32, 38, 42, 48],
    'income': [50000, 60000, 70000, 80000, 90000, 55000, 65000, 75000, 85000, 95000],
    'browsing_duration': [15, 20, 25, 30, 35, 18, 22, 28, 33, 40],
    'device_type': ['Mobile', 'Desktop', 'Mobile', 'Desktop', 'Mobile', 'Desktop', 'Mobile', 'Desktop', 'Mobile', 'Desktop'],
    'purchase': [1, 0, 1, 0, 1, 0, 1, 0, 1, 0]  
}
df = pd.DataFrame(data)
label_encoder = LabelEncoder()
df['device_type'] = label_encoder.fit_transform(df['device_type'])
X = df[['age', 'income', 'browsing_duration', 'device_type']]
y = df['purchase']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

decision_tree_model = DecisionTreeClassifier(random_state=42)

decision_tree_model.fit(X_train, y_train)

y_pred = decision_tree_model.predict(X_test)

from sklearn.metrics import accuracy_score, classification_report
accuracy = accuracy_score(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")
print("Classification Report:\n", classification_rep)

new_customer_data = pd.DataFrame({
    'age': [30],
    'income': [60000],
    'browsing_duration': [25],
    'device_type': ['Mobile']
})

new_customer_data['device_type'] = label_encoder.transform(new_customer_data['device_type'])

prediction = decision_tree_model.predict(new_customer_data)

print("\nPrediction for the new customer:")
print("Will make a purchase" if prediction[0] == 1 else "Will not make a purchase")
