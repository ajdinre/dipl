# train.py

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from model import Model1, DOModel, BNModel
from graphdataset import GraphDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load your dataset
TRAFFIC_FILE = "./data2/hops_12_nmap1_20000.csv"
dataset = GraphDataset(traffic_file=TRAFFIC_FILE)

# Split the dataset into training and testing sets
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

# Define the data loaders
batch_size = 16
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Instantiate the model
input_size = len(dataset.features_df_columns) - 1  # Exclude the label column
hidden_size = 64
output_size = 1  # Binary classification
#model = Model1(input_size, hidden_size, output_size).to(device)
model = BNModel(input_size, hidden_size, output_size).to(device)

# Define the loss function and optimizer
criterion = torch.nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Number of training epochs
num_epochs = 10

# Training loop
for epoch in range(num_epochs):
    running_loss = 0.0
    for i, data in enumerate(train_loader, 0):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()

        outputs = model(inputs.float())
        #print(f"{outputs=}")
        #print(f"{labels.float().unsqueeze(1)=}")
        loss = criterion(outputs, labels.float())
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f'Epoch: {epoch + 1}, Loss: {running_loss / len(train_loader)}')

print('Finished Training')
