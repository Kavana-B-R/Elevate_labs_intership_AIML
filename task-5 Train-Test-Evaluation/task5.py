import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix

df = pd.read_csv("heart.csv")

print("Dataset loaded successfully!\n")
print(df.head())

X = df.drop("target", axis=1)
y = df["target"]

print("\nFeatures shape:", X.shape)
print("Target shape:", y.shape)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("\nTraining data size:", X_train.shape)
print("Testing data size:", X_test.shape)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

print("\nLogistic Regression model trained successfully!")

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)

print("\nModel Evaluation Metrics:")
print("Accuracy :", accuracy)
print("Precision:", precision)
print("Recall   :", recall)

cm = confusion_matrix(y_test, y_pred)

print("\nConfusion Matrix:")
print(cm)

print("\nConfusion Matrix Explanation:")
print("True Negatives :", cm[0][0])
print("False Positives:", cm[0][1])
print("False Negatives:", cm[1][0])
print("True Positives :", cm[1][1])

print("\nInterpretation:")
print("- Accuracy shows overall correctness of the model.")
print("- Precision shows how reliable positive predictions are.")
print("- Recall shows how well the model detects heart disease cases.")
print("- Confusion matrix explains prediction distribution.")

print("\nTask 5 Completed Successfully!")
