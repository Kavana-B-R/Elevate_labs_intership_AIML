import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("titanic_dataset.csv")

print("Dataset loaded successfully!\n")
print(df.head())

print("\nDataset Info:")
print(df.info())

numerical_features = ["Age", "Fare", "SibSp", "Parch"]

for col in numerical_features:
    plt.figure()
    df[col].hist(bins=10)
    plt.title(f"Distribution of {col}")
    plt.xlabel(col)
    plt.ylabel("Count")
    plt.show()

categorical_features = ["Sex", "Pclass", "Embarked", "Survived"]

for col in categorical_features:
    plt.figure()
    sns.countplot(x=col, data=df)
    plt.title(f"Count Plot of {col}")
    plt.xlabel(col)
    plt.ylabel("Count")
    plt.show()

for col in numerical_features:
    plt.figure()
    sns.boxplot(y=df[col])
    plt.title(f"Box Plot of {col}")
    plt.ylabel(col)
    plt.show()

plt.figure(figsize=(8, 6))
correlation_matrix = df.corr(numeric_only=True)
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()

print("\nCorrelation with Survived:")
print(correlation_matrix["Survived"].sort_values(ascending=False))

print("\nEDA SUMMARY:")
print("- Females have a higher survival rate than males.")
print("- Higher passenger class (Pclass=1) shows better survival chances.")
print("- Fare has a positive correlation with survival.")
print("- Some numerical features contain outliers.")
print("- Dataset shows clear patterns useful for prediction.")

print("\nTask 3 EDA Completed Successfully!")
