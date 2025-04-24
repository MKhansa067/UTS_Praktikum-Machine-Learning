# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn import tree

# %%
citrus = pd.read_csv('citrus.csv')
citrus.describe().T  # Melihat statistik data (mean, std, min, max, dll)

# %%
print(citrus.head(10))  # Menampilkan 10 data teratas

# %%
sns.pairplot(citrus, hue='name', palette='Set1')
plt.show()

# %%
X = citrus.drop('name', axis=1)
y = citrus['name']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=10)
print(len(X_train))  # Jumlah data training
print(len(X_test))   # Jumlah data testing

# %%
model = DecisionTreeClassifier(criterion='entropy')
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print(X_test)  # Menampilkan data testing

# %%
print(classification_report(y_test, y_pred))

# %%
cm = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots(figsize=(7,7))
sns.set(font_scale=1.4)
sns.heatmap(cm, ax=ax, annot=True, annot_kws={"size": 16})
plt.xlabel('Predictions', fontsize=18)
plt.ylabel('Actuals', fontsize=18)
plt.title('Confusion Matrix', fontsize=18)
plt.show()

# %%
features = ['diameter', 'weight', 'red', 'green', 'blue']
fig, ax = plt.subplots(figsize=(25,20))
tree.plot_tree(model, feature_names=features, class_names=model.classes_, filled=True)
plt.show()

# %%
citrus_test_data = {
    'diameter': 5.5,
    'weight': 110.0,
    'red': 160,
    'green': 80,
    'blue': 10
}
prediction_input_df = pd.DataFrame([citrus_test_data])
prediction = model.predict(prediction_input_df[features])
print(prediction)
