# Developing a Neural Network Classification Model
## NAME : PRAJAN P
## REG NO: 212223240121
## AIM

To develop a neural network classification model for the given dataset.

## Problem Statement

An automobile company has plans to enter new markets with their existing products. After intensive market research, they’ve decided that the behavior of the new market is similar to their existing market.

In their existing market, the sales team has classified all customers into 4 segments (A, B, C, D ). Then, they performed segmented outreach and communication for a different segment of customers. This strategy has work exceptionally well for them. They plan to use the same strategy for the new markets.

You are required to help the manager to predict the right group of the new customers.

## Neural Network Model

<img width="603" height="792" alt="image" src="https://github.com/user-attachments/assets/fe48fe6a-4d9a-40fd-b64e-af194362bc8d" />

## DESIGN STEPS

### STEP 1: Data Collection and Understanding
Collect customer data from the existing market and identify the features that influence customer segmentation. Define the target variable as the customer segment (A, B, C, or D).

### STEP 2: Data Preprocessing
Remove irrelevant attributes, handle missing values, and encode categorical variables into numerical form. Split the dataset into training and testing sets.

### STEP 3: Model Design and Training
Design a neural network classification model with suitable input, hidden, and output layers. Train the model using the training data to learn patterns for customer segmentation.

### STEP 4: Model Evaluation and Prediction
Evaluate the trained model using test data and use it to predict the customer segment for new customers in the target market.


## PROGRAM

### Name: PRAJAN P
### Register Number: 212223240121

```python
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from torch.utils.data import TensorDataset, DataLoader
import seaborn as sns
import matplotlib.pyplot as plt

print("Name: PRAJAN P")
print("Register No: 212223240121")

data = pd.read_csv("customers.csv")

data = data.drop(columns=["ID"])

data.fillna({"Work_Experience": 0, "Family_Size": data["Family_Size"].median()}, inplace=True)

categorical_columns = ["Gender", "Ever_Married", "Graduated", "Profession", "Spending_Score", "Var_1"]
for col in categorical_columns:
    data[col] = LabelEncoder().fit_transform(data[col])

label_encoder = LabelEncoder()
data["Segmentation"] = label_encoder.fit_transform(data["Segmentation"])

X = data.drop(columns=["Segmentation"])
y = data["Segmentation"].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

X_train = torch.tensor(X_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long)
y_test = torch.tensor(y_test, dtype=torch.long)

train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=16, shuffle=True)
test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=16)

class PeopleClassifier(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, 32)
        self.fc2 = nn.Linear(32, 16)
        self.fc3 = nn.Linear(16, 4)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

input_size = X_train.shape[1]
model = PeopleClassifier(input_size)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

epochs = 50

for epoch in range(epochs):
    model.train()
    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()
    if (epoch+1) % 10 == 0:
        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")

model.eval()
predictions, actuals = [], []

with torch.no_grad():
    for X_batch, y_batch in test_loader:
        outputs = model(X_batch)
        _, predicted = torch.max(outputs, 1)
        predictions.extend(predicted.numpy())
        actuals.extend(y_batch.numpy())

accuracy = accuracy_score(actuals, predictions)
conf_matrix = confusion_matrix(actuals, predictions)
class_report = classification_report(actuals, predictions, target_names=label_encoder.classes_)

print("\nName: PRAJAN P")
print("Register No: 212223240121")
print(f"Test Accuracy: {accuracy*100:.2f}%")
print("Confusion Matrix:\n", conf_matrix)
print("Classification Report:\n", class_report)

sns.heatmap(conf_matrix, annot=True, cmap='Blues', xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_, fmt='g')
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.title("Confusion Matrix")
plt.show()

sample_input = X_test[12].unsqueeze(0)

with torch.no_grad():
    output = model(sample_input)
    predicted_class_index = torch.argmax(output[0]).item()
    predicted_class_label = label_encoder.inverse_transform([predicted_class_index])[0]

print("\nName: PRAJAN P")
print("Register No: 212223240121")
print(f"Predicted class for sample input: {predicted_class_label}")
print(f"Actual class for sample input: {label_encoder.inverse_transform([y_test[12].item()])[0]}")

```

## Dataset Information

<img width="736" height="378" alt="image" src="https://github.com/user-attachments/assets/32df1ba9-23a3-4c05-b90b-16d00eef7706" />

## OUTPUT

### Confusion Matrix

<img width="539" height="454" alt="image" src="https://github.com/user-attachments/assets/25f7268a-ad50-4f5d-ade4-248e8a1ef766" />


### Classification Report

<img width="603" height="581" alt="image" src="https://github.com/user-attachments/assets/1f1c3ae5-0dcc-4229-a42e-369bb9127250" />

### New Sample Data Prediction

<img width="423" height="110" alt="image" src="https://github.com/user-attachments/assets/bc103d6d-e75f-4d68-b397-3e4cfb867f65" />

## RESULT

Thus neural network classification model is developded for the given dataset. 
