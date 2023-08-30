import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.feature_selection import RFECV
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import Lasso
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score, classification_report
import tensorflow as tf
from keras.layers import Input, Dense
from keras.models import Model
from sklearn.ensemble import RandomForestClassifier


# Define the column names
column_names = ["id", "clump_thickness", "uniformity_of_cell_size", "uniformity_of_cell_shape", "marginal_adhesion",
                "single_epithelial_cell_size", "bare_nuclei", "bland_chromatin", "normal_nucleoli", "mitoses", "class"]

# Load the CSV file into a pandas DataFrame
path = "C:/Users/tgoul/OneDrive/Υπολογιστής/Thesis/ThesisFiles/python/data/breast-cancer-wisconsin.csv"
data = pd.read_csv(path, names=column_names)

# preprocess
# Display the first 5 rows of the DataFrame to verify that the data was loaded correctly
print(data.head())

# Check for missing data in the DataFrame
print(data.isnull().sum())


# Check for non-numerical columns and replace non-numeric values with NaN
non_numeric = []
for col in data.columns:
    if not pd.api.types.is_numeric_dtype(data[col]):
        non_numeric.append(col)
        non_numeric_rows = data.loc[~data[col].apply(
            lambda x: str(x).isdigit()), col].index.tolist()
        for row in non_numeric_rows:
            print(
                f"Non-numeric value found: {data.iloc[row, data.columns.get_loc(col)]} in column {col} and row {row}")
            data.iloc[row, data.columns.get_loc(col)] = np.nan

print("Data before replacing NaN values with mean:\n", data.head())

# Replace NaN values with the mean of their respective columns
for col in non_numeric:
    data[col] = pd.to_numeric(data[col], errors='coerce')
    mean = data[col].mean()
    num_missing = data[col].isna().sum()
    data[col].fillna(mean, inplace=True)
    print(
        f"Column {col}: {num_missing} missing values replaced with mean {mean:.2f}.")

print("Data after replacing NaN values with mean:\n", data.head())
# Remove unnecessary columns
data = data.drop(columns=['id'])
print(data.columns)

# Encode categorical features
categorical_columns = []
for col in data.columns:
    if data[col].dtype == 'object':
        categorical_columns.append(col)
if len(categorical_columns) > 0:
    print("Categorical columns:", categorical_columns)
    data = pd.get_dummies(data, columns=categorical_columns)
else:
    print("No categorical columns found.")

# Split the data into training and testing sets
# Specify the target column
target_column = 'class'

# Get the input features and target values
X = data.drop(columns=[target_column])
y = data[target_column]

# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42)

# Print the shape of the resulting datasets
print("X_train shape:", X_train.shape)
print("y_train shape:", y_train.shape)
print("X_test shape:", X_test.shape)
print("y_test shape:", y_test.shape)

# Scale numerical features
# Initialize the scaler
scaler = MinMaxScaler()
print("Data before scaling:\n", X_train.head())

# Specify the columns to be scaled
columns_to_scale = ['clump_thickness', 'uniformity_of_cell_size', 'uniformity_of_cell_shape', 'marginal_adhesion',
                    'single_epithelial_cell_size', 'bare_nuclei', 'bland_chromatin', 'normal_nucleoli', 'mitoses']
# Scale the columns in the training set
X_train[columns_to_scale] = scaler.fit_transform(X_train[columns_to_scale])

# Scale the columns in the test set
X_test[columns_to_scale] = scaler.transform(X_test[columns_to_scale])

# Print the first 10 rows of the data after scaling
print("Data after scaling:\n", X_train[:10])

# Implement traditional feature selection methods
# filter methods
# Chi-squared feature selection
# Select the top 5 features using chi-squared test
selector = SelectKBest(score_func=chi2, k=5)
X_new = selector.fit_transform(X_train, y_train)

# Print the names of the top 5 features
mask = selector.get_support()  # list of booleans
new_features = []  # The list of your K best features
for bool, feature in zip(mask, X_train.columns):
    if bool:
        new_features.append(feature)

print("Top 5 features selected by chi-squared test:", new_features)

# Wrapper method
# Create a logistic regression classifier
clf = LogisticRegression()
# Perform recursive feature elimination with cross-validation
selector = RFECV(estimator=clf, step=1, cv=5, scoring='accuracy')
selector.fit(X_train, y_train)

# Print the optimal number of features and their names
print("Optimal number of features:", selector.n_features_)
print("Selected features:", X_train.columns[selector.support_])

# Embedded method
# Create a Lasso classifier
clf = Lasso(alpha=0.1)
# Fit the Lasso classifier to the data
clf.fit(X_train, y_train)
# Print the selected features and their coefficients
selected_features = X_train.columns[clf.coef_ != 0]
print("Selected features:", selected_features)
print("Coefficients:", clf.coef_[clf.coef_ != 0])

# Train machine learning models on the selected features

# Specify the selected features
selected_features = ['clump_thickness', 'uniformity_of_cell_size','uniformity_of_cell_shape', 'marginal_adhesion', 'single_epithelial_cell_size']

# Train a logistic regression model on the selected features
model = LogisticRegression()
model.fit(X_train[selected_features], y_train)

# Evaluate the model on the test set
y_pred = model.predict(X_test[selected_features])
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, pos_label=2)
recall = recall_score(y_test, y_pred, pos_label=2)
f1 = f1_score(y_test, y_pred, pos_label=2)
conf_matrix = confusion_matrix(y_test, y_pred)

print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 score:", f1)
print("Confusion matrix:\n", conf_matrix)


report = classification_report(y_test, y_pred, digits = 10)
print(report)

#randomForestClassifier
rf_model = RandomForestClassifier()


rf_model.fit(X_train[selected_features], y_train)


y_pred_rf = rf_model.predict(X_test[selected_features])
accuracy_rf = accuracy_score(y_test, y_pred_rf)
precision_rf = precision_score(y_test, y_pred_rf, pos_label=2)
recall_rf = recall_score(y_test, y_pred_rf, pos_label=2)
f1_rf = f1_score(y_test, y_pred_rf, pos_label=2)
conf_matrix_rf = confusion_matrix(y_test, y_pred_rf)
print("Random Forest Classifier - Accuracy:", accuracy_rf)
print("Random Forest Classifier - Precision:", precision_rf)
print("Random Forest Classifier - Recall:", recall_rf)
print("Random Forest Classifier - F1 score:", f1_rf)
print("Random Forest Classifier - Confusion matrix:\n", conf_matrix_rf)
