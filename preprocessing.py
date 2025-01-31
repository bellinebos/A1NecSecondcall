import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

data_original = pd.read_csv('data.csv')

# Show mean, median, min and max values
print(data_original.describe().loc[['mean', '50%', 'min', 'max']])

# Filter only numeric columns
numeric_df = data_original.select_dtypes(include=['number'])

# Plot the boxplots
plt.subplots(figsize=(20, 8))

count = 1
for i in numeric_df.columns:
    plt.subplot(2, 10, count)  # Adjust the layout (2 rows, 10 columns)
    sns.boxplot(data=numeric_df[i])
    plt.title(i, fontsize=12)
    count += 1

plt.suptitle('Boxplots of all numeric variables of the dataset', fontsize=16)
plt.subplots_adjust(top=0.85)  # Adjust the title space to avoid overlap
plt.tight_layout()  # Ensure that subplots do not overlap
plt.show()

# Remove the 'percentage expenditure' column from the dataset as 1303 entries exceed 100%
data_new = data_original.drop(columns=['percentage expenditure'])

# Remove the 'measles' column from the dataset as 525 instances reporting over 1000 cases per 1000 people
data_new = data_new.drop(columns=['Measles '])

# Remove the 'bmi' column from the dataset as values are far outside realistic bounds
data_new = data_new.drop(columns=[' BMI '])

# Remove the 'under-five deaths' column from the dataset as 16 instances where values exceeded 1000 per 1000 persons
data_new = data_new.drop(columns=['under-five deaths '])

# Remove the 'thinness 5-9' column from the dataset as it has overlap with thinness 1-19
data_new = data_new.drop(columns=[' thinness 5-9 years'])

# Remove the 'infant deaths' column from the dataset as it is a subset of the broader measure under-five deaths
data_new = data_new.drop(columns=['infant deaths'])

# Remove the 'Population' column from the dataset as it has 22.2% missing values
data_new = data_new.drop(columns=['Population'])

# Remove the 'Hepatitis B' column from the dataset as it has 18.2% missing values
data_new = data_new.drop(columns=['Hepatitis B'])

# Remove  the 'Country' column from the dataset as it is categorical and too many values for one-hot encoding
data_new = data_new.drop(columns=['Country'])

# Remove missing values
data_cleaned = data_new.dropna()


