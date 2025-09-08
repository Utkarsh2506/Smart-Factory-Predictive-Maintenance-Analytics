import pandas as pd
from sqlalchemy import create_engine
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, ConfusionMatrixDisplay
from sqlalchemy import text

warnings.filterwarnings('ignore')

# --- Database Connection ---
db_user = 'root'
db_password = 'Bundela0421'
db_host = 'localhost'
db_name = 'factory_db'

engine = create_engine(f'mysql+mysqlconnector://{db_user}:{db_password}@{db_host}/{db_name}')
sql_query = "SELECT * FROM machine_data"
df = pd.read_sql(sql_query, con=engine)

print("Data loaded successfully. Here are the first 5 rows:")
print(df.head())
print("\nDataFrame Information:")
df.info()

# -------------------- EDA ------------------------

sns.set_style('whitegrid')

plt.figure(figsize=(15, 10))
numerical_cols = [
    'Air_Temperature_[K]',
    'Process_Temperature_[K]',
    'Rotational_Speed_[rpm]',
    'Torque_[Nm]',
    'Tool_Wear_[min]'
]

for i, col in enumerate(numerical_cols):
    plt.subplot(2, 3, i + 1)
    sns.histplot(df[col], kde=True)
    plt.title(f'Distribution of {col}')
plt.tight_layout()
plt.show()

try:
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    sns.countplot(x='Type', data=df, order=df['Type'].value_counts().index)
    plt.title('Machine Count by Type')

    plt.subplot(1, 2, 2)
    sns.countplot(x='Machine_Failure', data=df)
    plt.title('Overall Failure Count (1 = Failure)')
    plt.show()
except ValueError as e:
    print(f"\nERROR: The plot failed. {e}")
    print("ACTION: Look at the printed column list above and correct the column name in the code.")

plt.figure(figsize=(15, 10))
for i, col in enumerate(numerical_cols):
    plt.subplot(2, 3, i + 1)
    sns.boxplot(x='Machine_Failure', y=col, data=df)
    plt.title(f'{col} by Machine Failure')
plt.tight_layout()
plt.show()

corr_df = df[numerical_cols + ['Machine_Failure']]
correlation_matrix = corr_df.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm', square=True)
plt.title('Correlation Matrix of Numerical Features')
plt.show()

# -------------------- Feature Engineering & Preprocessing ------------------------

df['Power'] = ((df['Torque_[Nm]'] * df['Rotational_Speed_[rpm]']) / 9550).round(2)
df['TempDiff'] = (df['Process_Temperature_[K]'] - df['Air_Temperature_[K]'])

print('Dataframe with new features')
print(df[['Rotational_Speed_[rpm]', 'Torque_[Nm]', 'Power', 'Air_Temperature_[K]', 'Process_Temperature_[K]', 'TempDiff']].head())

# Clean column names for model compatibility
df.columns = (
    df.columns
    .str.replace('[', '', regex=False)
    .str.replace(']', '', regex=False)
    .str.replace('<', '', regex=False)
    .str.replace('>', '', regex=False)
    .str.replace(' ', '_')
)

le = LabelEncoder()
df['Type_le'] = le.fit_transform(df['Type'])

X = df.drop([
    'Machine_Failure',
    'Product_ID',
    'Type',
    'TWF',
    'HDF',
    'PWF',
    'OSF',
    'RNF'
], axis=1)
y = df['Machine_Failure']

# Split data into training and testing sets (80/20 split)
# stratify=y ensures proportional representation of the target in train/test sets.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print("\nData preparation complete.")
print(f"Shape of training features (X_train): {X_train.shape}")
print(f"Shape of testing features (X_test): {X_test.shape}")

# -------------------- Model Training & Evaluation ------------------------

scale = y_train.value_counts()[0] / y_train.value_counts()[1]

model = XGBClassifier(objective='binary:logistic', scale_pos_weight=scale, use_label_encoder=False, eval_metric='logloss')
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("\n--- Classification Report ---")
print(classification_report(y_test, y_pred))

print("\n--- Confusion Matrix ---")
ConfusionMatrixDisplay.from_estimator(model, X_test, y_test, cmap='Blues')
plt.title('Confusion Matrix')
plt.show()

# -------------------- Generate & Save Predictions ------------------------

failure_probabilities = model.predict_proba(X)[:, 1]

results_df = pd.DataFrame({
    'ProductID': df['Product_ID'],
    'actual_failure': df['Machine_Failure'],
    'failure_probability': failure_probabilities
})

print("\n--- Prediction Results for Dashboard ---")
print(results_df.head())

with engine.connect() as connection:
    connection.execute(text("DROP TABLE IF EXISTS Failure_Predictions"))

results_df.to_sql('Failure_Predictions', con=engine, if_exists='append', index=False)

print("\nPrediction results have been saved to the 'Failure_Predictions' table in MySQL.")