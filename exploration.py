import pandas as pd

df = pd.read_csv('project_data.csv')

# first lets see how big this dataset is
print("Total Rows:", df.shape[0])
print("Total Columns:", df.shape[1])
print("Feature Variables (without Class):", df.shape[1] - 1)

print()

# checking the class distribution
total_rows = len(df)
labeled = df['Class'].notna().sum()
counts = df['Class'].value_counts(dropna=False)

print("Class = No:", counts['No'])
print("Class = Yes:", counts['Yes'])
print("Class missing:", df['Class'].isnull().sum())
print()
print("Total labeled rows:", labeled)
print("% No:", round(counts['No'] / labeled * 100, 1))
print("% Yes:", round(counts['Yes'] / labeled * 100, 1))
print("% missing class out of all rows:", round(df['Class'].isnull().sum() / total_rows * 100, 1))

print()

# now lets look at missing values overall
total_cells = df.shape[0] * df.shape[1]
total_missing = df.isnull().sum().sum()

print("Total cells:", total_cells)
print("Total missing cells:", total_missing)
print("Missing percentage:", round(total_missing / total_cells * 100, 2))
print("Columns with missing values:", (df.isnull().sum() > 0).sum())
print("Rows with at least one missing value:", df.isnull().any(axis=1).sum())

print()

# missing values per column sorted
missing_per_col = df.isnull().sum()
missing_per_col = missing_per_col[missing_per_col > 0].sort_values(ascending=False)

print("Missing values per column:")
for col, count in missing_per_col.items():
    pct = round(count / total_rows * 100, 1)
    print(f"  {col}: {count} ({pct}%)")

print()

# checking data types
print("Data types:")
print(df.dtypes.value_counts())
print()
print("float64 columns:", len(df.select_dtypes(include='float64').columns))
print("int64 columns:", len(df.select_dtypes(include='int64').columns))
print("int64 column names:", df.select_dtypes(include='int64').columns.tolist())

print()

# i noticed that many columns have exactly 1038 missing values
# lets check how many columns share this exact number
missing_counts = df.isnull().sum()
cols_1038 = missing_counts[missing_counts == 1038].index.tolist()

print("Columns with exactly 1038 missing:", len(cols_1038))
print("Column names:", cols_1038)
print()

# now checking if these 1038 missing values are in the same rows
first_col = cols_1038[0]
all_same = all(df[col].isnull().equals(df[first_col].isnull()) for col in cols_1038)
print("Are all 1038 missing rows the same across all these columns?", all_same)

print()

# lets investigate what these 1038 rows actually are
vacant = df[df[first_col].isnull()]
occupied = df[df[first_col].notna()]

print("Total rows:", len(df))
print("Rows with 1038 pattern:", len(vacant))
print("Class labels in those rows:", vacant['Class'].value_counts(dropna=False).to_dict())
print()
print("Average filled columns in 1038 rows:", round(vacant.notnull().sum(axis=1).mean(), 1), "out of 145")
print("Average filled columns in normal rows:", round(occupied.notnull().sum(axis=1).mean(), 1), "out of 145")

print()

# breaking down the 1206 missing class rows
both = df[df['Class'].isnull() & df[first_col].isnull()]
class_only = df[df['Class'].isnull() & df[first_col].notna()]

print("Total missing Class:", df['Class'].isnull().sum())
print("Missing Class AND demographics (vacant units):", len(both))
print("Missing Class but have household data:", len(class_only))