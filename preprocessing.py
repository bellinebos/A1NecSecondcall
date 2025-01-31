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

delete variableS: percentage expenditure, measles, bmi, under five deaths, Thinness 5-9 Years, population, hepatitis b, infant deaths, countries
