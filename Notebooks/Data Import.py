import pandas as pd
from sqlalchemy import create_engine

# Replace 'your_password' with your actual MySQL root password.
db_user = 'root'
db_password = 'your_password'
db_host = 'localhost'
db_name = 'factory_db'

# Replace 'Filepath_to_your/Factory Dataset.csv' with your actual file path to the dataset.
df = pd.read_csv('Filepath_to_your/Factory Dataset.csv')
df = df.drop('UDI', axis=1)

engine = create_engine(f'mysql+mysqlconnector://{db_user}:{db_password}@{db_host}/{db_name}')

df.to_sql('machine_data', con=engine, if_exists='append', index=False)

print("Data has been successfully loaded into the 'machine_data' table in MySQL.")