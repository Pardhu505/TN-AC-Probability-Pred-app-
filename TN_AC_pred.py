import pandas as pd
import numpy as np
import pip
pip.main(['install', 'scikit-learn'])

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import pip
pip.main(['install', 'xgboost'])
import xgboost as xgb
import pip
pip.main(['install', 'imbalanced-learn'])
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import GridSearchCV


df = pd.read_excel(r'D:\Tamil Nadu Works\Tamil Nadu __ Master Tracker.xlsx', sheet_name='Last 3 AE results')
le_assembly = LabelEncoder()
le_party = LabelEncoder()
df['Assembly Name'] = le_assembly.fit_transform(df['Assembly Name'])
df['Party'] = le_party.fit_transform(df['Party'])
assembly_names = le_assembly.classes_
party_names = le_party.classes_
X = df[['Year', 'Assembly Name', 'Party']]
y = df['Position'] == 1  # party won 
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'class_weight': ['balanced']  # class imbalance
}
clf = RandomForestClassifier(random_state=42)
grid_search = GridSearchCV(clf, param_grid, scoring='f1', cv=5)
grid_search.fit(X_train, y_train)
best_clf = grid_search.best_estimator_
best_clf.fit(X_train, y_train)

y_pred = best_clf.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
from sklearn.metrics import f1_score

f1 = f1_score(y_test, y_pred)
from sklearn.metrics import precision_score

precision = precision_score(y_test, y_pred)

print(f"Accuracy of the model: {accuracy:.2f}")
print(f"F1 Score of the model: {f1:.2f}")
print(f"Precision Score of the model: {precision:.2f}")

# functions
def predict_winner(assembly_name, party_name, year):
    assembly_name_index = le_assembly.transform([assembly_name])[0]
    party_name_index = le_party.transform([party_name])[0]
    
    
    input_data = pd.DataFrame({
        'Year': [year],
        'Assembly Name': [assembly_name_index],
        'Party': [party_name_index]
    })
    
    prediction = best_clf.predict_proba(input_data)
    return prediction[0][1]  

assembly_name_input = input("Enter the Assembly Name: ")
party_name_input = input("Enter the Party Name: ")
year_input = int(input("Enter the Year: "))

probabilities = []
for party in party_names:
    prob = predict_winner(assembly_name_input, party, year_input)
    probabilities.append(prob)

selected_party_prob = predict_winner(assembly_name_input, party_name_input, year_input)
print(f"Probability of {party_name_input} winning: {selected_party_prob:.2f}")