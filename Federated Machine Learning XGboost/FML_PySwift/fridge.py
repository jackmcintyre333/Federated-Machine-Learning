import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from pyswift_federated import PySwiftFederatedFramework
# Load your IoT fridge dataset
csv_path = r"Train_Test_IoT_Fridge.csv"
df = pd.read_csv(csv_path)
df = df.drop(columns=["type"])
# Make sure to replace 'target' below with the actual name of your label column
label_column = "label"

for col in df.columns:
        if df[col].dtype == 'object':
            labelEncoder = LabelEncoder()
            df[col] = labelEncoder.fit_transform(df[col])
# Split into training and testing sets
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df[label_column])

# Initialize the framework
framework = PySwiftFederatedFramework(df=train_df, label_col=label_column, num_clients=10)

# Train models
models = framework.run_federated_training()

# Evaluate the aggregated FedAvg model
framework.evaluate_fedavg_model(test_df)