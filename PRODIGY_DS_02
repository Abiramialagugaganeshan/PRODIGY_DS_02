import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

url = "/content/drive/MyDrive/test.csv"
titanic_df = pd.read_csv(url)

print(titanic_df.head())

print(titanic_df.isnull().sum())
titanic_df['Age'].fillna(titanic_df['Age'].median(), inplace=True)
sns.histplot(titanic_df['Age'], bins=30, kde=True)
plt.title('Distribution of Age')
plt.show()

sns.countplot(x='Sex', data=titanic_df)
plt.title('Passenger Gender Distribution')
plt.show()
sns.countplot(x='Sex', hue='Cabin', data=titanic_df)
plt.title('Survival by Passenger Class')
plt.show()
sns.boxplot(x='Pclass', y='Age', data=titanic_df)
plt.title('Survival by Age')
plt.show()
correlation_matrix = titanic_df.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Matrix')
plt.show()
