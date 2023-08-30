import pandas as pd             # for data manipulation and analysis
import numpy as np              # for numerical operations
from sklearn.model_selection import train_test_split   # for splitting data into training and testing sets
from sklearn.preprocessing import MinMaxScaler        # for feature scaling
# for creating neural network layers
from keras.layers import Input, Dense, LeakyReLU,Conv2D,Flatten
from keras.models import Model                         # for creating and training neural network models
from sklearn.linear_model import LogisticRegression   # for creating and training logistic regression models
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                            confusion_matrix, classification_report, roc_curve, roc_auc_score)  # for model evaluation
from io import StringIO         # for creating in-memory file objects
from sklearn.ensemble import RandomForestClassifier
import tensorflow as tf
from keras.utils import plot_model

# Define column names
column_names = ["id", "clump_thickness", "uniformity_of_cell_size",   "uniformity_of_cell_shape",
"marginal_adhesion", "single_epithelial_cell_size", "bare_nuclei", "bland_chromatin",
"normal_nucleoli", "mitoses", "class"]

# Load the CSV file into a pandas DataFrame
path = "C:/Users/tgoul/OneDrive/Υπολογιστής/Thesis/ThesisFiles/python/data/breast-cancer-wisconsin.csv"
data = pd.read_csv(path, names=column_names)

# Display the first 5 rows
print(data.head())
print(data.head(),"\n")

# Check for missing data in the DataFrame
print(data.isnull().sum())
print(data.isna().sum())

# preprocess

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

#Split the data into training and testing sets
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

# Initialize the scaler
scaler = MinMaxScaler()
print("Data before scaling:\n", X_train.head())

# Specify the columns to be scaled
columns_to_scale = ['clump_thickness', 'uniformity_of_cell_size', 'uniformity_of_cell_shape', 'marginal_adhesion',                   'single_epithelial_cell_size', 'bare_nuclei', 'bland_chromatin', 'normal_nucleoli', 'mitoses']

# Scale the columns in the training set
X_train[columns_to_scale] = scaler.fit_transform(X_train[columns_to_scale])

# Scale the columns in the test set
X_test[columns_to_scale] = scaler.transform(X_test[columns_to_scale])

# Print the first 10 rows of the data after scaling
print("Data after scaling:\n", X_train[:10])



#Feature selection 
#autoencoder

# define the shape of the input data
input_shape = (X_train.shape[1],)

# define the architecture of the autoencoder
input_layer = Input(shape=input_shape)
encoded = Dense(64, activation='relu')(input_layer)
encoded = Dense(32, activation='relu')(encoded)
encoded = Dense(16, activation='relu')(encoded)

decoded = Dense(32, activation='relu')(encoded)
decoded = Dense(64, activation='relu')(decoded)
decoded = Dense(X_train.shape[1], activation='sigmoid')(decoded)

autoencoder = Model(input_layer, decoded)

# compile the autoencoder
autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

# train the autoencoder    
autoencoder.fit(X_train, X_train, epochs=50, batch_size=32, validation_split=0.2)


# use the encoder to transform the input data into a lower-dimensional representation
encoder = Model(input_layer, encoded)
X_train_encoded = encoder.predict(X_train)
X_test_encoded = encoder.predict(X_test)

# train the logistic regression model on the encoded data

lr = LogisticRegression()
lr.fit(X_train_encoded, y_train)

# evaluate the model on the test set
y_pred = lr.predict(X_test_encoded)
report = classification_report(y_test, y_pred)
print(report)

# Print a summary of the model
autoencoder.summary()

# Plot the model
plot_model(autoencoder, to_file='model.png',show_shapes=True, show_layer_names=True)
# # Generate the visualization as an image file
# plot_model(autoencoder, to_file='neural_network.png', show_shapes=True, show_layer_names=True)

# # Feature selection using CNN

# # Reshape the input data for CNN
# X_train_cnn = X_train.values.reshape(-1, 3, 3, 1)
# X_test_cnn = X_test.values.reshape(-1, 3, 3, 1)

# # Define the shape of the input data
# input_shape = X_train_cnn[0].shape

# # Define the architecture of the CNN
# input_layer = Input(shape=input_shape)
# conv1 = Conv2D(32, kernel_size=(3, 3), activation='relu')(input_layer)
# conv2 = Conv2D(64, kernel_size=(3, 3), activation='relu')(conv1)
# flatten = Flatten()(conv2)
# dense1 = Dense(64, activation='relu')(flatten)
# dense2 = Dense(64, activation='relu')(dense1)
# output_layer = Dense(10, activation='softmax')(dense2)
# encoded = Dense(16, activation='relu')(flatten)


# # Build the CNN model
# cnn_model = Model(input_layer, encoded)

# # Compile the CNN model
# cnn_model.compile(optimizer='adam', loss='binary_crossentropy')

# # Train the CNN model
# cnn_model.fit(X_train_cnn, X_train_cnn, epochs=50,
#             batch_size=32, validation_split=0.2)

# # Use the CNN model to extract features from the input data
# X_train_encoded = cnn_model.predict(X_train_cnn)
# X_test_encoded = cnn_model.predict(X_test_cnn)

# # Train the logistic regression model on the encoded data
# lr = LogisticRegression()
# lr.fit(X_train_encoded, y_train)

# # Evaluate the model on the test set
# y_pred = lr.predict(X_test_encoded)
# report = classification_report(y_test, y_pred)
# print(report)

# # Convert the report to a DataFrame
# df = pd.read_csv(StringIO(report), skiprows=1, sep=' {2,}', engine='python')
# df.columns = ['class', 'precision', 'recall', 'f1-score', 'support']

# # Save the DataFrame to an Excel file
# df.to_excel(
#     'C:/Users/tgoul/OneDrive/Υπολογιστής/Thesis/ThesisFiles/python/FSdLResults.xlsx')

# # convert the report to a DataFrame
# df = pd.read_csv(StringIO(report), skiprows=1, sep=' {2,}', engine='python')
# df.columns = ['class', 'precision', 'recall', 'f1-score', 'support']



# # save the DataFrame to an Excel file
# df.to_excel('C:/Users/tgoul/OneDrive/Υπολογιστής/Thesis/ThesisFiles/python/FSdLResults.xlsx')
