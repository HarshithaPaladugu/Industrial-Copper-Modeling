# Industrial-Copper-Modeling
A machine learning regression model can address the  issues in  copper industry with less complex data related to sales and pricing.
Sure, let's break down the workflow and execution of the provided code step-by-step:

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

