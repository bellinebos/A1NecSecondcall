import pandas as pd
from sklearn.model_selection import train_test_split

data_original = pd.read_csv('data.csv')

# Show mean, median, min and max values
print(data_original.describe().loc[['mean', '50%', 'min', 'max']])

#one-hot encoding of variable Status
data_original = pd.get_dummies(data_original, columns=['Status'], prefix='Status')

# Convert 'True'/'False' to '1'/'0' for the newly created columns
data_original['Status_Developed'] = data_original['Status_Developed'].astype(int)
data_original['Status_Developing'] = data_original['Status_Developing'].astype(int)

# Remove the 'percentage expenditure' column from the dataset as 1303 entries exceed 100%
data_new = data_original.drop(columns=['percentage expenditure'])

# Remove the 'measles' column from the dataset as 525 instances reporting over 1000 cases per 1000 people
data_new = data_new.drop(columns=['Measles '])

# Remove the 'bmi' column from the dataset as values are far outside realistic bounds
data_new = data_new.drop(columns=[' BMI '])

# Remove the 'under-five deaths' column from the dataset as 16 instances where values exceeded 1000 per 1000 persons
data_new = data_new.drop(columns=['under-five deaths '])

# Remove the 'year' column from the dataset as year of observation doesnt add relevance in predicting life expanctancy
data_new = data_new.drop(columns=['Year'])

# Remove the 'thinness 5-9' column from the dataset as it has overlap with thinness 1-19
data_new = data_new.drop(columns=[' thinness 5-9 years'])

# Remove the 'Population' column from the dataset as it has 22.2% missing values
data_new = data_new.drop(columns=['Population'])

# Remove the 'Hepatitis B' column from the dataset as it has 18.2% missing values
data_new = data_new.drop(columns=['Hepatitis B'])

# Remove  the 'Country' column from the dataset as it is categorical and too many values for one-hot encoding
data_new = data_new.drop(columns=['Country'])

# Remove missing values
data_cleaned = data_new.dropna()

# Save preprocessed dataset to a new CSV file
data_cleaned.to_csv('processeddataset.csv', index=False)

# splitting data
# Step 1: Separate input features (X) and output feature (y)
X = data_cleaned.drop(columns=["Life expectancy "])  # Input features
y = data_cleaned["Life expectancy "]  # Output feature

# Step 2: Split the data into 80% training/validation and 20% test
X_train_val, X_test, y_train_val, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, shuffle=True
)

# Step 3: Check the resulting shapes
print("Training+Validation set (X):", X_train_val.shape)
print("Test set (X):", X_test.shape)
print("Training+Validation set (y):", y_train_val.shape)
print("Test set (y):", y_test.shape)

# Combine input features and output labels for training/validation and test sets
train_data = pd.concat([X_train_val, y_train_val], axis=1)  # Concatenate X and y for training
test_data = pd.concat([X_test, y_test], axis=1)  # Concatenate X and y for testing

# Save the data to CSV files
train_data.to_csv('traindata.csv', index=False)  # Save the training data
test_data.to_csv('testdata.csv', index=False)  # Save the test data
