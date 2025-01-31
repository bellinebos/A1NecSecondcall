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

# Remove the 'percentage expenditure' column from the dataset
data_new = data_original.drop(columns=['percentage expenditure'])

# Drop 'measles' from the dataset
data_new = data_new.drop(columns=['Measles '])

# Remove the 'infant deaths' column
data_new = data_new.drop(columns=['infant deaths'])

# Remove the 'Population' column
data_new = data_new.drop(columns=['Population'])

# Remove the 'Hepatitis B' column
data_new = data_new.drop(columns=['Hepatitis B'])

# Drop the 'bmi' column
data_new = data_new.drop(columns=[' BMI '])

# Drop the 'under-five deaths' column
data_new = data_new.drop(columns=['under-five deaths '])

# Drop the 'thinness 5-9' column
data_new = data_new.drop(columns=[' thinness 5-9 years'])

# Drop the 'Country' column
data_new = data_new.drop(columns=['Country'])

# Remove missing values
data_cleaned = data_new.dropna()


