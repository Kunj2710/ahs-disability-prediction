import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder

df = pd.read_csv('project_data.csv')
print("original shape:", df.shape)

# drop CONTROL - just a unique ID, no predictive value
df = df.drop(columns=['CONTROL'])
print("after dropping CONTROL:", df.shape)

# drop rows where Class is missing
df = df.dropna(subset=['Class'])
print("after dropping missing class rows:", df.shape)
print("class distribution:")
print(df['Class'].value_counts())

# separate features and target
X = df.drop(columns=['Class'])
y = df['Class']

print()
print("features shape:", X.shape)

# define column types based on codebook knowledge

numeric_cols = [
    'TOTROOMS', 'FINROOMS', 'UFINROOMS', 'STORIES', 'UNITFLOORS',
    'BEDROOMS', 'BATHROOMS', 'YRBUILT',
    'NUMELDERS', 'NUMADULTS', 'NUMPEOPLE', 'NUMVETS', 'NUMYNGKIDS',
    'NUMOLDKIDS', 'HHYNGKIDS', 'HHOLDKIDS', 'NUMNONREL', 'NUMSUBFAM',
    'NUMSECFAM', 'HHADLTKIDS', 'HHMOVE', 'HHGRAD', 'PERPOVLVL',
    'HINCP', 'FINCP', 'TOTHCAMT', 'ELECAMT', 'GASAMT', 'OILAMT',
    'OTHERAMT', 'TRASHAMT', 'WATERAMT', 'UTILAMT', 'INSURAMT',
    'POVLVLINC', 'CELLPHONE', 'NUMASTHMAST'
]

unique_counts = X.nunique()
binary_cols = unique_counts[unique_counts <= 2].index.tolist()
categorical_cols = [col for col in X.columns
                    if col not in numeric_cols and col not in binary_cols]

print("numeric columns:", len(numeric_cols))
print("binary categorical columns:", len(binary_cols))
print("multi-category categorical columns:", len(categorical_cols))

# impute missing values before outlier detection
# numeric: median (more robust than mean when outliers exist)
# categorical: mode (most frequent value)
print()
print("missing values before imputation:", X.isnull().sum().sum())

num_imputer = SimpleImputer(strategy='median')
cat_imputer = SimpleImputer(strategy='most_frequent')

X[numeric_cols] = num_imputer.fit_transform(X[numeric_cols])
X[categorical_cols] = cat_imputer.fit_transform(X[categorical_cols])
X[binary_cols] = cat_imputer.fit_transform(X[binary_cols])

print("missing values after imputation:", X.isnull().sum().sum())

# outlier handling using IQR method (from lecture 1)
# a value is an outlier if below Q1 - 6*IQR or above Q3 + 6*IQR
# we use factor 6 (conservative)
# instead of removing rows we CAP the outlier values at the boundary
# this way we dont lose any rows - we have 39 numeric columns
# and removing rows would eliminate too much data
print()
print("handling outliers by capping at IQR boundaries...")

outlier_factor = 6
total_capped = 0

for col in numeric_cols:
    q1 = X[col].quantile(0.25)
    q3 = X[col].quantile(0.75)
    iqr = q3 - q1
    low  = q1 - outlier_factor * iqr
    high = q3 + outlier_factor * iqr
    before = ((X[col] < low) | (X[col] > high)).sum()
    X[col] = X[col].clip(lower=low, upper=high)
    total_capped += before

print("total outlier values capped:", total_capped)
print("rows kept (no rows removed):", len(X))

# encode target variable
# Yes -> 1 (has disability), No -> 0
y = y.map({'Yes': 1, 'No': 0})
print()
print("target after encoding:")
print(y.value_counts())
print("class imbalance ratio:", round(y.value_counts()[0] / y.value_counts()[1], 2), ":1")

# encode categorical variables
# binary: label encode to 0 and 1
# multi-category: one-hot encode (dummy variables like lecture 1 slide)
le = LabelEncoder()
for col in binary_cols:
    X[col] = le.fit_transform(X[col].astype(str))

X = pd.get_dummies(X, columns=categorical_cols)
print()
print("shape after one-hot encoding:", X.shape)

# z-score standardization for numeric columns
# formula: v' = (v - mean) / std  (lecture 1, slide 29)
# needed because numeric variables have very different ranges
# income goes up to millions, NUMELDERS only goes to 4
scaler = StandardScaler()
X[numeric_cols] = scaler.fit_transform(X[numeric_cols])

print()
print("final dataset shape:", X.shape)
print("final class distribution:")
print(y.value_counts())
print()
print("preprocessing complete - ready for model building")