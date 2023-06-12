import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, fbeta_score


from models import ReLUFeedForwardNN
from graphdataset import GraphDataset

def load_data(traffic_file, split_ratio=0.8, batch_size=16):
    dataset = GraphDataset(traffic_file=traffic_file)
    train_size = int(split_ratio * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader, len(dataset.features_df_columns)

def create_model(model_type, input_size, hidden_size=64, output_size=1, device="cpu"):
    model = model_type(input_size, hidden_size, output_size).to(device)
    return model

def get_loss_and_optim(model, learning_rate=0.01):
    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    return criterion, optimizer

def calculate_accuracy(outputs, labels):
    with torch.no_grad():
        pred = torch.round(torch.sigmoid(outputs))  # sigmoid and round to get binary predictions
    correct = (pred == labels).float()  # compare predictions to true label
    acc = correct.sum() / len(correct)
    return acc

def calculate_f_scores(outputs, labels):
    with torch.no_grad():
        pred = torch.round(torch.sigmoid(outputs))  # sigmoid and round to get binary predictions
    pred = pred.cpu().numpy()
    labels = labels.cpu().numpy()
    
    f1 = f1_score(labels, pred, average='micro')  # calculate f1 score
    f2 = fbeta_score(labels, pred, beta=2, average='micro')  # calculate f2 score

    return f1, f2


def train_model(model, criterion, optimizer, train_loader, device="cpu", num_epochs=10):
    for epoch in range(num_epochs):
        running_loss = 0.0
        running_acc = 0.0
        running_f1 = 0.0
        running_f2 = 0.0

        for i, data in enumerate(train_loader, 0):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs.float())
            loss = criterion(outputs, labels.float())
            loss.backward()
            optimizer.step()
            acc = calculate_accuracy(outputs, labels)
            f1, f2 = calculate_f_scores(outputs, labels)

            running_loss += loss.item()
            running_acc += acc.item()
            running_f1 += f1
            running_f2 += f2

        avg_loss = running_loss / len(train_loader)
        avg_acc = running_acc / len(train_loader)
        avg_f1 = running_f1 / len(train_loader)
        avg_f2 = running_f2 / len(train_loader)

        print(f"Epoch: {epoch + 1}, Loss: {avg_loss:.5f}, Accuracy: {avg_acc:.5f}, F1: {avg_f1:.5f}, F2: {avg_f2:.5f}")

    print("Finished Training")



def main():
    model_type = ReLUFeedForwardNN

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    traffic_file = "./data2/hops_12_nmap1_20000.csv"
    train_loader, test_loader, input_size = load_data(traffic_file)
    input_size -= 1  # Exclude the label column
    model = create_model(model_type, input_size, device=device)
    criterion, optimizer = get_loss_and_optim(model)
    train_model(model, criterion, optimizer, train_loader, device=device)

if __name__ == "__main__":
    main()
