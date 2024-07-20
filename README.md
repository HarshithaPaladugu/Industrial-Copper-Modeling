# Industrial-Copper-Modeling
A machine learning regression model can address the  issues in  copper industry with less complex data related to sales and pricing.

### 1. **Reading the Dataset**
```python
import pandas as pd
import numpy as np
df_4=pd.read_excel("C:\\Users\\HP\\GUVI_PROJ\\Copper_Set_one.xlsx")
```
- The dataset is read from an Excel file into a DataFrame `df_4`.

### 2. **Cleaning 'material_ref' Column**
```python
df_4['material_ref'] = df_4['material_ref'].apply(lambda x: np.nan if str(x).startswith('00000') else x)
df_4.to_excel("C:\\Users\\HP\\GUVI_PROJ\\Copper_Set_one_modified.xlsx", index=False)
```
- Values in the 'material_ref' column that start with '00000' are replaced with NaN. The modified DataFrame is saved to a new Excel file.

### 3. **Finding Number of Unique Values in Each Column**
```python
print(len(df_4['item_date'].unique()))
print(len(df_4['customer'].unique()))
...
print(len(df_4['status'].unique()))
```
- The number of unique values for each specified column is printed.

### 4. **Finding Number of Missing Values in Each Column**
```python
missing_values = df_4.isnull().sum()
print(missing_values)
```
- The number of missing values in each column is printed.

### 5. **Handling Missing Values**
```python
for column in df_4.columns:
    if df_4[column].isnull().sum() > 0:
        if df_4[column].dtype in ['int64', 'float64']:
            if df_4[column].skew() < -1 or df_4[column].skew() > 1:
                df_4[column].fillna(df_4[column].median(), inplace=True)
            else:
                df_4[column].fillna(df_4[column].mean(), inplace=True)
        else:
            df_4[column].fillna(df_4[column].mode()[0], inplace=True)
df_4.to_excel("C:\\Users\\HP\\GUVI_PROJ\\Copper_Set_one_modified.xlsx", index=False)
```
- Missing values in numeric columns are filled with the median if the data is skewed or the mean if not. Missing values in non-numeric columns are filled with the mode. The modified DataFrame is saved.

### 6. **Checking for Remaining Missing Values**
```python
missing_values = df_4.isnull().sum()
print(missing_values)
```
- The code checks if there are any remaining missing values after handling them.

### 7. **Data Formatting: Converting Columns to Numeric**
```python
df_4['quantity tons']=pd.to_numeric(df_4['quantity tons'],errors='coerce')
df_4['country']=pd.to_numeric(df_4['country'],errors='coerce')
...
df_4.to_excel("C:\\Users\\HP\\GUVI_PROJ\\Copper_Set_one_modified.xlsx",index=False)
```
- Columns are converted to numeric data types, and the DataFrame is saved again.

### 8. **Data Formatting: Converting Date Columns**
```python
df_4['item_date']=pd.to_datetime(df_4['item_date'],format='%Y%m%d', errors='coerce').dt.date
df_4['delivery date']=pd.to_datetime(df_4['delivery date'],format='%Y%m%d', errors='coerce').dt.date
df_4.to_excel("C:\\Users\\HP\\GUVI_PROJ\\Copper_Set_one_modified.xlsx",index=False)
```
- Date columns are converted to datetime format.

### 9. **Splitting Date Columns**
```python
df_4['item_date'] = pd.to_datetime(df_4['item_date'], format='%Y%m%d')
df_4['delivery date'] = pd.to_datetime(df_4['delivery date'], format='%Y%m%d')
df_4['item_year'] = df_4['item_date'].dt.year
df_4['item_month'] = df_4['item_date'].dt.month
df_4['item_day'] = df_4['item_date'].dt.day
df_4['delivery_year'] = df_4['delivery date'].dt.year
df_4['delivery_month'] = df_4['delivery date'].dt.month
df_4['delivery_day'] = df_4['delivery date'].dt.day
df_4.drop(columns=['item_date', 'delivery date'], inplace=True)
df_4.to_excel("C:\\Users\\HP\\GUVI_PROJ\\Copper_Set_one_modified.xlsx", index=False)
print(df_4.head())
```
- Date columns are split into separate year, month, and day columns. The original date columns are dropped.

### 10. **Visualizing Skewness and Outliers**
```python
import seaborn as sns
import matplotlib.pyplot as plt

sns.distplot(df_4['quantity tons'])
plt.show()
sns.violinplot(df_4['quantity tons'])
plt.show()
...
sns.violinplot(df_4['delivery_day'])
plt.show()
```
- The distributions and outliers in the data are visualized using `distplot` and `violinplot`.

### 11. **Handling Skewness in the Dataset**
```python
from scipy.stats import boxcox

def handle_skewness(df, numerical_cols):
    for col in numerical_cols:
        if col != 'selling_price':  # Avoid transforming the target variable directly
            skewness = df[col].skew()
            if skewness > 1 or skewness < -1:
                if df[col].min() > 0:
                    df[col], _ = boxcox(df[col] + 1e-9)
                else:
                    df[col] = np.log1p(df[col] - df[col].min() + 1)
    return df

df = handle_skewness(df, numerical_cols)

output_path = "C:\\Users\\HP\\GUVI_PROJ\\Copper_Set_one_modified_outliers_handled_skew_handled.xlsx"
df.to_excel(output_path, index=False)
```
- Skewness in the data is handled using Box-Cox or log transformation, depending on the distribution of the values.

### 12. **Identifying Categorical Columns**
```python
categorical_cols = df.select_dtypes(include=['object']).columns
print("Categorical Columns:", categorical_cols)
```
- The categorical columns in the DataFrame are identified and printed.

### 13. **Function to Analyze Relationship and Apply Suitable Encoding for Classification**
The function `encode_column_classification` analyzes the relationship between a categorical column and the target variable for classification tasks and applies suitable encoding based on the analysis.

#### Function Definition:
```python
def encode_column_classification(df, column, target, top_n=10):
    # Create contingency table
    contingency_table = pd.crosstab(df[column], df[target])
    # Perform Chi-Square test
    chi2, p, _, _ = chi2_contingency(contingency_table)
    print(f"Chi-Square test between {column} and {target}: p-value = {p}")
    
    # If high cardinality, use Ordinal Encoding
    if df[column].nunique() > top_n:
        print(f"High cardinality detected in {column}. Applying Ordinal Encoding.")
        encoder = OrdinalEncoder()
        df[column] = encoder.fit_transform(df[[column]])
    else:
        # Choose encoding method based on p-value
        if p < 0.05:  # Significant relationship
            # Apply ordinal encoding
            encoder = OrdinalEncoder()
            df[column] = encoder.fit_transform(df[[column]])
        else:
            # Apply one-hot encoding with top N categories
            df_top_n = df[column].value_counts().index[:top_n]
            df[column] = df[column].apply(lambda x: x if x in df_top_n else 'Other')
            df = pd.get_dummies(df, columns=[column], drop_first=True)
    return df
```

#### Execution:
```python
# Apply encoding based on target variable for classification
for col in categorical_cols:
    df = encode_column_classification(df, col, target_variable_classification)

# Save the modified dataset to a new file
output_path_classification =  "C:\\Users\\HP\\GUVI_PROJ\\Copper_Set_one_modified_outliers_handled_skew_handled_classif_encoded.xlsx"
df.to_excel(output_path_classification, index=False)

# For better demonstration, print a few rows of the transformed dataframe
print(df.head())
```
- For each categorical column, the function performs a Chi-Square test to determine the relationship with the target variable (`status`).
- If the column has high cardinality, it applies Ordinal Encoding.
- Based on the p-value, it either applies Ordinal Encoding (if the relationship is significant) or One-Hot Encoding (if the relationship is not significant).
- The modified DataFrame is saved to a new Excel file, and a few rows are printed for verification.

### 14. **Encoding Categorical Variables for Regression Modeling**
The next part of the code deals with encoding categorical variables for a regression task.

#### Load the Dataset:
```python
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder
from scipy.stats import spearmanr

df = pd.read_excel("C:\\Users\\HP\\GUVI_PROJ\\Copper_Set_one_modified_outliers_handled_skew_handled.xlsx")

categorical_cols = df.select_dtypes(include=['object']).columns
print("Categorical Columns:", categorical_cols)
```
- The dataset is loaded, and categorical columns are identified and printed.

### 15. **Reducing High Cardinality**
A helper function `reduce_cardinality` is defined to handle high cardinality in categorical columns.
```python
def reduce_cardinality(df, column, threshold=0.01):
    freq = df[column].value_counts(normalize=True)
    rare_categories = freq[freq < threshold].index
    df[column] = df[column].apply(lambda x: 'Other' if x in rare_categories else x)
    return df
```
- This function replaces rare categories in a column with 'Other' if their frequency is below a specified threshold.

### 16. **Analyzing Relationship and Applying Suitable Encoding for Regression**
A function `encode_column` is defined to analyze the relationship between a categorical column and the target variable for regression tasks and apply suitable encoding.
```python
def encode_column(df, column, target_reg):
    if column in categorical_cols:  # Only encode categorical columns
        if target_reg in df.columns:
            correlation, _ = spearmanr(df[column].astype(str), df[target_reg])
            print(f"Spearman correlation between {column} and {target_reg}: {correlation}")

        df = reduce_cardinality(df, column)

        if correlation > 0.5:  # Adjust the threshold as needed
            encoder = OrdinalEncoder()
            df[column] = encoder.fit_transform(df[[column]])
        else:
            df = pd.get_dummies(df, columns=[column], drop_first=True)

    return df
```
- The function calculates the Spearman correlation between the column and the target variable (`selling_price`).
- High cardinality in the column is reduced.
- Depending on the correlation value, it applies either Ordinal Encoding or One-Hot Encoding.

This function would be applied to each categorical column similarly to the classification encoding function.
Both classification and regression parts handle categorical data differently, tailored to the nature of the target variable and relationships within the dataset.
Certainly! Let's walk through the workflow and execution of the additional code you've provided.



#### 17. **Analyzing and Encoding Categorical Columns for Regression**
The goal of this section is to encode categorical columns in a dataset for regression analysis.

##### Load Dataset and Identify Categorical Columns:
```python
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
from scipy.stats import spearmanr
import category_encoders as ce

# Load the dataset
df = pd.read_excel("C:\\Users\\HP\\GUVI_PROJ\\Copper_Set_one_modified_outliers_handled_skew_handled.xlsx")

# Identify categorical columns
categorical_cols = df.select_dtypes(include=['object']).columns.difference(['id'])
print("Categorical Columns:", categorical_cols)

# Define target variable for regression
target_variable = 'selling_price'
```
- The dataset is loaded from an Excel file.
- Categorical columns are identified, excluding the 'id' column.
- The target variable for regression (`selling_price`) is defined.

##### Ensure 'id' Column is Included in Non-Categorical Columns:
```python
non_categorical_cols = df.select_dtypes(exclude=['object']).columns
if 'id' in df.columns:
    non_categorical_cols = non_categorical_cols.union(['id'])
```
- Non-categorical columns are identified, ensuring the 'id' column is included.

##### Function to Reduce High Cardinality:
```python
def reduce_cardinality(df, column, threshold=0.01):
    freq = df[column].value_counts(normalize=True)
    rare_categories = freq[freq < threshold].index
    df[column] = df[column].apply(lambda x: 'Other' if x in rare_categories else x)
    return df
```
- This function reduces high cardinality by replacing rare categories with 'Other'.

##### Function to Analyze Relationship and Apply Suitable Encoding:
```python
def encode_column(df, column, target_variable):
    df = reduce_cardinality(df, column)
    correlation, _ = spearmanr(df[column].astype(str), df[target_variable])
    print(f"Spearman correlation between {column} and {target_variable}: {correlation}")

    if correlation > 0.5:
        try:
            encoder = LabelEncoder()
            df[column] = encoder.fit_transform(df[column])
        except Exception as e:
            print(f"Label encoding failed for {column}: {e}")
            encoder = OrdinalEncoder()
            df[column] = encoder.fit_transform(df[[column]])
    else:
        df = pd.get_dummies(df, columns=[column], drop_first=True)
    
    return df
```
- This function first reduces high cardinality.
- It then calculates the Spearman correlation between the column and the target variable.
- Based on the correlation, it applies either Label Encoding/Ordinal Encoding (if high correlation) or One-Hot Encoding (if low correlation).

##### Apply Encoding to Each Categorical Column:
```python
for col in categorical_cols:
    df = encode_column(df, col, target_variable)

# Ensure non-categorical columns are included in the final dataframe
final_df = pd.concat([df[non_categorical_cols], df.drop(columns=non_categorical_cols)], axis=1)
```
- The function is applied to each categorical column.
- Non-categorical columns are ensured to be included in the final DataFrame.

##### Save Modified Dataset and Print:
```python
output_path = "C:\\Users\\HP\\GUVI_PROJ\\Copper_Set_one_modified_outliers_handled_skew_handled_reg_encod.xlsx"
final_df.to_excel(output_path, index=False)

print(final_df.head())
```
- The modified DataFrame is saved to a new Excel file.
- A few rows of the transformed DataFrame are printed for verification.

#### 18. **Filtering Data Points for Classification**
The goal of this section is to filter the dataset to include only specific rows for classification tasks.

##### Load Dataset and Inspect 'status' Column:
```python
import pandas as pd

file_path = "C:\\Users\\HP\\GUVI_PROJ\\Copper_Set_one_modified_outliers_handled_skew_handled.xlsx"
df = pd.read_excel(file_path)

print("Unique values in 'status' column before filtering:")
print(df['status'].unique())
```
- The dataset is loaded, and unique values in the 'status' column are printed before filtering.

##### Clean 'status' Column and Filter:
```python
df['status'] = df['status'].str.strip().str.lower()

print("Unique values in 'status' column after stripping and lowering case:")
print(df['status'].unique())

df_filtered = df[df['status'].isin(['won', 'lost'])]

if df_filtered.empty:
    print("The filtered dataframe is empty. Please check the 'status' values.")
else:
    output_file = "C:\\Users\\HP\\GUVI_PROJ\\Copper_Set_one_modified_outliers_handled_skew_handled_filtered.xlsx"
    df_filtered.to_excel(output_file, index=False)

    print("Rows with 'status' other than 'won' or 'lost' have been removed.")
    print(df_filtered.head())
```
- Whitespace is stripped, and text is converted to lowercase to handle inconsistencies in the 'status' column.
- The DataFrame is filtered to include only rows where 'status' is 'won' or 'lost'.
- If the filtered DataFrame is not empty, it is saved to a new Excel file, and a few rows are printed for verification.

#### 19. **Loading and Identifying Categorical Columns**
The goal is to encode categorical columns in the dataset for classification tasks.

##### Load Dataset and Identify Categorical Columns:
```python
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
from scipy.stats import chi2_contingency

# Load the dataset
df = pd.read_excel("C:\\Users\\HP\\GUVI_PROJ\\Copper_Set_one_modified_outliers_handled_skew_handled_filtered.xlsx")

# Identify categorical columns
categorical_cols = df.select_dtypes(include=['object']).columns
print("Categorical Columns:", categorical_cols)

# Define target variable for classification
target_variable_classification = 'status'
```
- The dataset is loaded from an Excel file.
- Categorical columns are identified.
- The target variable for classification (`status`) is defined.

#### 20. **Function to Analyze and Encode Categorical Columns for Classification**
This function applies suitable encoding to categorical columns based on their relationship with the target variable.

##### Function Definition:
```python
def encode_column_classification(df, column, target, top_n=10):
    # Create contingency table
    contingency_table = pd.crosstab(df[column], df[target])
    # Perform Chi-Square test
    chi2, p, _, _ = chi2_contingency(contingency_table)
    print(f"Chi-Square test between {column} and {target}: p-value = {p}")
    
    # Apply label encoding if possible
    try:
        encoder = LabelEncoder()
        df[column] = encoder.fit_transform(df[column])
        print(f"Applied Label Encoding to {column}")
        return df
    except Exception as e:
        print(f"Label encoding failed for {column}: {e}")
    
    # If high cardinality, use Ordinal Encoding
    if df[column].nunique() > top_n:
        print(f"High cardinality detected in {column}. Applying Ordinal Encoding.")
        encoder = OrdinalEncoder()
        df[column] = encoder.fit_transform(df[[column]])
    else:
        # Choose encoding method based on p-value
        if p < 0.05:  # Significant relationship
            print(f"Significant relationship detected between {column} and {target}. Applying Ordinal Encoding.")
            encoder = OrdinalEncoder()
            df[column] = encoder.fit_transform(df[[column]])
        else:
            print(f"No significant relationship detected between {column} and {target}. Applying One-Hot Encoding.")
            # Apply one-hot encoding with top N categories
            df_top_n = df[column].value_counts().index[:top_n]
            df[column] = df[column].apply(lambda x: x if x in df_top_n else 'Other')
            df = pd.get_dummies(df, columns=[column], drop_first=True)
    return df
```
- The function performs a Chi-Square test to determine the relationship between the categorical column and the target variable.
- It attempts to apply Label Encoding. If it fails, it applies Ordinal Encoding if the column has high cardinality or a significant relationship with the target variable.
- Otherwise, it applies One-Hot Encoding with the top N categories.

##### Apply Encoding to Each Categorical Column:
```python
for col in categorical_cols:
    df = encode_column_classification(df, col, target_variable_classification)

# Save the modified dataset to a new file
output_path_classification = "C:\\Users\\HP\\GUVI_PROJ\\Copper_Set_one_modified_outliers_handled_skew_handled_classif_encoded_new.xlsx"
df.to_excel(output_path_classification, index=False)

# For better demonstration, print a few rows of the transformed dataframe
print(df.head())
```
- The encoding function is applied to each categorical column.
- The modified DataFrame is saved to a new Excel file.
- A few rows of the transformed DataFrame are printed for verification.

#### 21. **Handling Missing Values**
The goal is to handle any missing values in the classification modeling dataset.

##### Load Dataset and Check for Missing Values:
```python
import pandas as pd

df = pd.read_excel("C:\\Users\\HP\\GUVI_PROJ\\Copper_Set_one_modified_outliers_handled_skew_handled_classif_encoded_new.xlsx")
missing_values = df.isnull().sum()
print(missing_values)
```
- The dataset is loaded from the new Excel file.
- Missing values in the dataset are counted and printed.

##### Handle Missing Values:
```python
# Load the Excel file into a DataFrame
df_4 = pd.read_excel("C:\\Users\\HP\\GUVI_PROJ\\Copper_Set_one_modified_outliers_handled_skew_handled_classif_encoded_new.xlsx")

# Handling null values
for column in df_4.columns:
    if df_4[column].isnull().sum() > 0:
        if df_4[column].dtype in ['int64', 'float64']:
            # For numeric columns, fill with mean or median
            if df_4[column].skew() < -1 or df_4[column].skew() > 1:
                df_4[column].fillna(df_4[column].median(), inplace=True)
            else:
                df_4[column].fillna(df_4[column].mean(), inplace=True)
        else:
            # For non-numeric columns, fill with mode
            df_4[column].fillna(df_4[column].mode()[0], inplace=True)

df_4.to_excel("C:\\Users\\HP\\GUVI_PROJ\\Copper_Set_one_modified_outliers_handled_skew_handled_classif_encoded_new_1.xlsx", index=False)
```
- The dataset is loaded again.
- Missing values in numeric columns are filled with the mean or median, depending on the skewness.
- Missing values in non-numeric columns are filled with the mode.
- The updated DataFrame is saved to a new Excel file.

##### Verify Missing Values:
```python
df = pd.read_excel("C:\\Users\\HP\\GUVI_PROJ\\Copper_Set_one_modified_outliers_handled_skew_handled_classif_encoded_new_1.xlsx")
missing_values = df.isnull().sum()
print(missing_values)
```
- The updated dataset is loaded, and missing values are checked again to ensure none are left.

#### 22. **Adding New Features**
The goal is to add the columns `item_date`, `delivery_date`, and `days_to_delivery` to the dataset.

##### Load Dataset and Check Required Columns:
```python
import pandas as pd

file_path = "C:\\Users\\HP\\GUVI_PROJ\\Copper_Set_one_modified_outliers_handled_skew_handled_classif_encoded_new_1.xlsx"
df = pd.read_excel(file_path)

required_columns = ['item_year', 'item_month', 'item_day', 'delivery_year', 'delivery_month', 'delivery_day', 
                    'quantity tons', 'thickness', 'width', 'selling_price']
missing_columns = [col for col in required_columns if col not in df.columns]

if missing_columns:
    raise ValueError(f"Missing required columns: {missing_columns}")
```
- The dataset is loaded from the new Excel file.
- Required columns are checked to ensure they exist in the DataFrame.

##### Convert Date Columns to Integers and Create New Features:
```python
# Convert year, month, and day columns to integers
df['item_year'] = df['item_year'].astype(int)
df['item_month'] = df['item_month'].astype(int)
df['item_day'] = df['item_day'].astype(int)
df['delivery_year'] = df['delivery_year'].astype(int)
df['delivery_month'] = df['delivery_month'].astype(int)
df['delivery_day'] = df['delivery_day'].astype(int)

# Create item_date and delivery_date
df['item_date'] = pd.to_datetime(df[['item_year', 'item_month', 'item_day']].astype(str).agg('-'.join, axis=1))
df['delivery_date'] = pd.to_datetime(df[['delivery_year', 'delivery_month', 'delivery_day']].astype(str).agg('-'.join, axis=1))

# Calculate the difference in days between item_date and delivery_date
df['days_to_delivery'] = (df['delivery_date'] - df['item_date']).dt.days

output_file = "C:\\Users\\HP\\GUVI_PROJ\\Copper_Set_one_modified_outliers_handled_skew_handled_classif_encoded_new_1.xlsx"
df.to_excel(output_file, index=False)

print("Updated DataFrame with new features saved to the Excel file.")
print(df.head())
```
- Year, month, and day columns are converted to integers.
- New features `item_date` and `delivery_date` are created by combining the respective year, month, and day columns.
- `days_to_delivery` is calculated as the difference between `delivery_date` and `item_date`.
- The updated DataFrame with new features is saved to a new Excel file.
