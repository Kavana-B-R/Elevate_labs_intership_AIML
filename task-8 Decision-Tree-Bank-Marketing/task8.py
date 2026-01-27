import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report

df = pd.read_csv("bank.csv", sep=";")

print("Dataset loaded successfully!\n")
print(df.head())

print("\nDataset Info:")
print(df.info())

print("\nMissing Values Check:")
print(df.isnull().sum())

df.replace("unknown", pd.NA, inplace=True)

# Fill missing values with mode (most frequent)
df.fillna(df.mode().iloc[0], inplace=True)

print("\nHandled 'unknown' values")

df_encoded = pd.get_dummies(df, drop_first=True)

print("\nCategorical features encoded")

X = df_encoded.drop("y_yes", axis=1)
y = df_encoded["y_yes"]

print("\nFeature matrix shape:", X.shape)
print("Target shape:", y.shape)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("\nTrain size:", X_train.shape)
print("Test size :", X_test.shape)

model = DecisionTreeClassifier(
    max_depth=4,
    criterion="gini",
    random_state=42
)

model.fit(X_train, y_train)

print("\nDecision Tree model trained successfully!")

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("\nTest Accuracy:", accuracy)
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

train_accuracy = model.score(X_train, y_train)
test_accuracy = model.score(X_test, y_test)

print("Train Accuracy:", train_accuracy)
print("Test Accuracy :", test_accuracy)

plt.figure(figsize=(22, 12))
plot_tree(
    model,
    feature_names=X.columns,
    class_names=["No", "Yes"],
    filled=True
)
plt.title("Decision Tree – Bank Marketing Subscription Prediction")
plt.show()

print("\nTask 8 Completed Successfully!")
