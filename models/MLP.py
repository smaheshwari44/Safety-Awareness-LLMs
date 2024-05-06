import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split,ConcatDataset
from sklearn.metrics import roc_curve, auc, f1_score
import sklearn.metrics as metrics
import pickle
import numpy as np
from matplotlib import pyplot as plt
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.nn.functional as F
from sklearn.model_selection import StratifiedShuffleSplit



import pandas as pd

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class MLPModel(nn.Module):
    def __init__(self, input_size):
        super(MLPModel, self).__init__()
        self.fc1 = nn.Linear(input_size, 256)  # Increase model complexity
        self.fc2 = nn.Linear(256, 64)
        self.fc3 = nn.Linear(64, 1)
        self.dropout = nn.Dropout(0.2)  # Add dropout
        self.batchnorm1 = nn.BatchNorm1d(256)  # Add batch normalization
        self.batchnorm2 = nn.BatchNorm1d(64)

    def forward(self, x):
        x = F.relu(self.batchnorm1(self.fc1(x)))
        x = self.dropout(x)
        x = F.relu(self.batchnorm2(self.fc2(x)))
        x = self.dropout(x)
        x = torch.sigmoid(self.fc3(x))
        return x

class LogisticRegressionModel(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.linear = nn.Linear(input_size, 1)
    
    def forward(self, x):
        return torch.sigmoid(self.linear(x))

def load_data(file_path):
    with open(file_path, 'rb') as file:
        data = pickle.load(file)
    return data['Activations'], data['Labels'], data['Global_Labels']

def prepare_datasets(activations, labels, global_labels, layer, random_seed=42):
    data_x = np.array(activations[layer])
    data_y = np.array(labels)
    datasets = {}
    
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.3, random_state=random_seed)
    
    for domain in np.unique(global_labels):
        domain_indices = np.where(np.array(global_labels) == domain)[0]
        domain_x = data_x[domain_indices]
        domain_y = data_y[domain_indices]
        
        # Splitting the data into training and remaining (val + test)
        for train_index, remaining_index in sss.split(domain_x, domain_y):
            train_x, remaining_x = domain_x[train_index], domain_x[remaining_index]
            train_y, remaining_y = domain_y[train_index], domain_y[remaining_index]
        
        # Calculate validation size to be 10% of the total data
        val_size = int(0.1 * len(domain_y))
        
        # Remaining data is for validation and testing
        val_x, test_x = remaining_x[:val_size], remaining_x[val_size:]
        val_y, test_y = remaining_y[:val_size], remaining_y[val_size:]
        
        # Converting numpy arrays to PyTorch tensors
        train_x_tensor = torch.tensor(train_x, dtype=torch.float32)
        val_x_tensor = torch.tensor(val_x, dtype=torch.float32)
        test_x_tensor = torch.tensor(test_x, dtype=torch.float32)
        train_y_tensor = torch.tensor(train_y, dtype=torch.float32)
        val_y_tensor = torch.tensor(val_y, dtype=torch.float32)
        test_y_tensor = torch.tensor(test_y, dtype=torch.float32)
        
        # Creating TensorDataset for each set
        train_dataset = TensorDataset(train_x_tensor, train_y_tensor)
        val_dataset = TensorDataset(val_x_tensor, val_y_tensor)
        test_dataset = TensorDataset(test_x_tensor, test_y_tensor)
        
        datasets[domain] = (train_dataset, val_dataset, test_dataset)
    
    return datasets

def train_and_evaluate(train_loader, val_loader, test_loader, input_size):
    model = MLPModel(input_size).to(device)#LogisticRegressionModel(input_size).to(device)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)
    patience = 5
    best_val_loss = float('inf')
    epochs_no_improve = 0
    best_model = None

    # Train model with early stopping
    for epoch in range(5000):  # Maximum number of epochs
        model.train()
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs).squeeze()
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        # Validate to find the best threshold
        val_labels, val_probs = evaluate_model(model, val_loader)
        val_loss = criterion(torch.tensor(val_probs), torch.tensor(val_labels))
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            best_model = model
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print("Early stopping triggered.")
                break
        #print(f"Epoch {epoch+1},Train Loss: {loss:.4f}, Validation Loss: {val_loss:.4f}")

    # Load best model to make final predictions
    model = best_model
    test_labels, test_probs = evaluate_model(model, test_loader)
    best_threshold = find_best_threshold(test_labels, test_probs)
    test_preds = [1 if prob > best_threshold else 0 for prob in test_probs]
    f1 = metrics.f1_score(test_labels, test_preds)
    auc = metrics.roc_auc_score(test_labels, test_probs)
    accuracy = metrics.accuracy_score(test_labels, test_preds)
    precision = metrics.precision_score(test_labels, test_preds)
    recall = metrics.recall_score(test_labels, test_preds)

    print({
        'F1 Score': f1,
        'AUC': auc,
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall
    })
    return {
        'F1 Score': f1,
        'AUC': auc,
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall
    }

def evaluate_model(model, loader):
    model.eval()
    all_labels, all_probs = [], []
    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device)
            outputs = model(inputs).squeeze()
            all_labels.extend(labels.tolist())
            all_probs.extend(outputs.tolist())
    return all_labels, all_probs

# def find_best_threshold(labels, probs):
#     fpr, tpr, thresholds = roc_curve(labels, probs)
#     J = tpr - fpr
#     ix = np.argmax(J)
#     best_thresh = thresholds[ix]
#     return best_thresh

def find_best_threshold(labels, probs):
    thresholds = np.linspace(0, 1, 101)  # Generate 101 thresholds between 0 and 1
    best_f1 = -1
    best_thresh = 0
    for thresh in thresholds:
        preds = (probs >= thresh).astype(int)
        f1 = f1_score(labels, preds)
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = thresh
    return best_thresh

def run_experiments(activations, labels, global_labels, save_file, layers):
    all_results = []
    for layer in layers:
        datasets = prepare_datasets(activations, labels, global_labels, layer)
        input_size = np.array(activations[layer]).shape[1]
        for domain, (train_set, val_set, test_set) in datasets.items():
            train_loader = DataLoader(train_set, batch_size=64, shuffle=True)
            val_loader = DataLoader(val_set, batch_size=64, shuffle=False)
            test_loader = DataLoader(test_set, batch_size=64, shuffle=False)

            # Hold-out domain testing
            other_domains_train_sets = [train for d, (train, _, _) in datasets.items() if d != domain]
            other_domains_train_set = ConcatDataset(other_domains_train_sets)
            other_train_loader = DataLoader(other_domains_train_set, batch_size=32, shuffle=True)
            hold_out_results = train_and_evaluate(other_train_loader, val_loader, test_loader, input_size)
            hold_out_results['Test Type'] = 'Hold-Out'
            hold_out_results['Layer'] = layer
            hold_out_results['Domain'] = domain
            all_results.append(hold_out_results)

            # In-domain testing
            in_domain_results = train_and_evaluate(train_loader, val_loader, test_loader, input_size)
            in_domain_results['Test Type'] = 'In-Domain'
            in_domain_results['Layer'] = layer
            in_domain_results['Domain'] = domain
            all_results.append(in_domain_results)

            # Out-of-domain testing
            other_domains_test_sets = [test for d, (_, _, test) in datasets.items() if d != domain]
            other_domains_test_set = ConcatDataset(other_domains_test_sets)
            other_test_loader = DataLoader(other_domains_test_set, batch_size=64, shuffle=False)
            other_domains_val_sets = [val for d, (_, val, _) in datasets.items() if d != domain]
            other_domains_val_set = ConcatDataset(other_domains_val_sets)
            other_val_loader = DataLoader(other_domains_val_set, batch_size=64, shuffle=False)
            out_of_domain_results = train_and_evaluate(train_loader, other_val_loader, other_test_loader, input_size)
            out_of_domain_results['Test Type'] = 'Out-of-Domain'
            out_of_domain_results['Layer'] = layer
            out_of_domain_results['Domain'] = domain
            all_results.append(out_of_domain_results)

    # Save results to CSV
    results_df = pd.DataFrame(all_results)
    results_df.to_csv(save_file, index=False)
    print("Results saved to", save_file)

pickle_file =  pickle_file = [
    "LLM_HARMS_LLAMA2_7B_EMBEDDINGS.pkl",
    "LLM_HARMS_LLAMA2_7B_CHAT_EMBEDDINGS.pkl",
    "LLM_HARMS_LLAMA2_13B_EMBEDDINGS.pkl",
    "LLM_HARMS_LLAMA2_70B_EMBEDDINGS.pkl",
    "LLM_HARMS_LLAMA2_70B_CHAT_EMBEDDINGS.pkl",
    "LLM_HARMS_LLAMA3_8B_EMBEDDINGS.pkl",
    "LLM_HARMS_LLAMA3_8B_INSTRUCT_EMBEDDINGS.pkl",
    "LLM_HARMS_LLAMA3_70B_EMBEDDINGS.pkl",
    "LLM_HARMS_LLAMA3_70B_INSTRUCT_EMBEDDINGS.pkl",
    "LLM_HARMS_MISTRAL_7B_EMBEDDINGS.pkl",
    "LLM_HARMS_MISTRAL_8x7B_EMBEDDINGS.pkl",
    "LLM_HARMS_BERT_EMBEDDINGS.pkl",
    "LLM_HARMS_ROBERTA_EMBEDDINGS.pkl",
    ]
layers = [
    [1, 4, 8, 12, 16, 20, 24, 28, 32],
    [1, 4, 8, 12, 16, 20, 24, 28, 32],
    [1, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40],
    [1, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 64, 68, 72, 76, 80],
    [1, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 64, 68, 72, 76, 80],
    [1, 4, 8, 12, 16, 20, 24, 28, 32],
    [1, 4, 8, 12, 16, 20, 24, 28, 32],
    [1, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 64, 68, 72, 76, 80],
    [1, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 64, 68, 72, 76, 80],
    [1, 4, 8, 12, 16, 20, 24, 28, 32],
    [1, 4, 8, 12, 16, 20, 24, 28, 32],
    ['1'],
    ['1'],
    ]
save_paths = [
    "logistic_regression_results_llama2_7b_mlp.csv",
    "logistic_regression_results_llama2_7b_chat_mlp.csv",
    "logistic_regression_results_llama2_13b_mlp.csv",
    "logistic_regression_results_llama2_70b_mlp.csv",
    "logistic_regression_results_llama2_70b_chat_mlp.csv",
    "logistic_regression_results_llama3_8b_mlp.csv",
    "logistic_regression_results_llama3_8b_instruct_mlp.csv",
    "logistic_regression_results_llama3_70b_mlp.csv",
    "logistic_regression_results_llama3_70b_instruct_mlp.csv",
    "logistic_regression_results_mistral_7b_mlp.csv",
    "logistic_regression_results_mistral_8x7b_mlp.csv",
    "logistic_regression_results_bert_mlp.csv",
    "logistic_regression_results_roberta_mlp.csv",
    ]
for i in range(len(pickle_file)):
    try:
        file_path = pickle_file[i]
        save_file = save_paths[i]
        activations, labels, global_labels = load_data(file_path)
        run_experiments(activations, labels, global_labels, save_file, layers[i])
    except Exception as e:
        print(e)
        print("Error in ",pickle_file[i])










