import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Results files mapped by model name
results_files = {
    'llama2_7b': 'logistic_regression_results_llama2_7b.csv',
    'llama2_7b_mlp': 'logistic_regression_results_llama2_7b_mlp.csv',
    'llama2_7b_chat': 'logistic_regression_results_llama2_7b_chat.csv',
    'llama2_7b_chat_mlp': 'logistic_regression_results_llama2_7b_chat_mlp.csv',
    'llama2_13b': 'logistic_regression_results_llama2_13b.csv',
    'llama2_13b_mlp': 'logistic_regression_results_llama2_13b_mlp.csv',
    'llama3_8b': 'logistic_regression_results_llama3_8b.csv',
    'llama3_8b_mlp': 'logistic_regression_results_llama3_8b_mlp.csv',
    'llama3_8b_instruct': 'logistic_regression_results_llama3_8b_instruct.csv',
    'llama3_8b_instruct_mlp': 'logistic_regression_results_llama3_8b_instruct_mlp.csv',
    'llama2_70b': 'logistic_regression_results_llama2_70b.csv',
    'llama2_70b_mlp': 'logistic_regression_results_llama2_70b_mlp.csv',
    'llama2_70b_chat': 'logistic_regression_results_llama2_70b_chat.csv',
    'llama2_70b_chat_mlp': 'logistic_regression_results_llama2_70b_chat_mlp.csv',
    'llama3_70b': 'logistic_regression_results_llama3_70b.csv',
    'llama3_70b_mlp': 'logistic_regression_results_llama3_70b_mlp.csv',
    'llama3_70b_instruct': 'logistic_regression_results_llama3_70b_instruct.csv',
    'llama3_70b_instruct_mlp': 'logistic_regression_results_llama3_70b_instruct_mlp.csv',
    'mistral_7b': 'logistic_regression_results_mistral_7b.csv',
    'mistral_7b_mlp': 'logistic_regression_results_mistral_7b_mlp.csv',
    'mistral_8x7b': 'logistic_regression_results_mistral_8x7b.csv',
    'mistral_8x7b_mlp': 'logistic_regression_results_mistral_8x7b_mlp.csv',
    'bert': 'logistic_regression_results_bert.csv',
    'bert_mlp': 'logistic_regression_results_bert_mlp.csv',
    'roberta': 'logistic_regression_results_roberta.csv',
    'roberta_mlp': 'logistic_regression_results_roberta_mlp.csv'
}

sns.set(style="whitegrid")

def plot_data(test_type, filter_func, title, filename):
    """ Plots F1 scores based on test types and model filters. """
    plt.figure(figsize=(14, 8))
    max_layer = max(pd.read_csv(file)['Layer'].max() for file in results_files.values())
    for model_label, file_path in results_files.items():
        if filter_func(model_label):
            data = pd.read_csv(file_path)
            subset = data[data['Test Type'] == test_type]
            grouped = subset.groupby('Layer')['F1 Score'].agg(['mean', 'std']).reset_index()
            if grouped['Layer'].max() < max_layer:
                last_mean = grouped['mean'].iloc[-1]
                last_std = grouped['std'].iloc[-1]
                missing_layers = range(grouped['Layer'].max() + 1, max_layer + 1)
                extended = pd.DataFrame({
                    'Layer': missing_layers,
                    'mean': [last_mean] * len(missing_layers),
                    'std': [last_std] * len(missing_layers)
                })
                grouped = pd.concat([grouped, extended], ignore_index=True)
            plt.plot(grouped['Layer'], grouped['mean'], label=f'{model_label}', marker='o', linestyle='-')
            plt.fill_between(grouped['Layer'], grouped['mean'] - grouped['std'], grouped['mean'] + grouped['std'], alpha=0.2)
    plt.title(f'{title} - {test_type}')
    plt.xlabel('Layer')
    plt.ylabel('F1 Score')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'{filename}_{test_type}.png')
    plt.close()

# Define filter functions
is_small_model = lambda x: '70b' not in x and 'mlp' not in x
is_large_model = lambda x: ('70b' in x and 'mlp' not in x) or (x=='bert' or x=='roberta' or x=='llama2_7b_chat' or x=='llama3_8b_instruct')
is_mlp = lambda x: 'mlp' in x

# Generate plots for small models
for test in ['In-Domain', 'Out-of-Domain', 'Hold-Out']:
    plot_data(test, is_small_model, 'Small Models', 'small_models')

# Generate plots for large models
for test in ['In-Domain', 'Out-of-Domain', 'Hold-Out']:
    plot_data(test, is_large_model, 'Large Models', 'large_models')

# Set plot style
sns.set(style="whitegrid")

# Plot comparisons for each base model name that has an MLP and non-MLP version
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set plot style
sns.set(style="whitegrid")

def plot_model_comparisons(model_base_name):
    """Plots MLP vs non-MLP models for each test type separately, includes Bert and RoBERTa."""
    plt.figure(figsize=(24, 24))  # Adjusted for a single column with three rows
    test_types = ['In-Domain', 'Out-of-Domain', 'Hold-Out']
    
    for i, test_type in enumerate(test_types, 1):
        plt.subplot(3, 1, i)  # Three plots in a single column
        max_layer = max(pd.read_csv(file)['Layer'].max() for file in results_files.values())
        
        # Include Bert and RoBERTa in every plot
        extra_models = ['bert', 'roberta']
        all_models = [model_base_name] + extra_models
        
        for model in all_models:
            for suffix in ['', '_mlp']:
                model_label = model + suffix
                file_path = results_files.get(model_label)
                if file_path:
                    data = pd.read_csv(file_path)
                    subset = data[data['Test Type'] == test_type]
                    grouped = subset.groupby('Layer')['F1 Score'].agg(['mean', 'std']).reset_index()
                    
                    # Extend F1 scores for models with fewer layers
                    if grouped['Layer'].max() < max_layer:
                        last_mean = grouped['mean'].iloc[-1]
                        last_std = grouped['std'].iloc[-1]
                        missing_layers = range(grouped['Layer'].max() + 1, max_layer + 1)
                        extended = pd.DataFrame({
                            'Layer': missing_layers,
                            'mean': [last_mean] * len(missing_layers),
                            'std': [last_std] * len(missing_layers)
                        })
                        grouped = pd.concat([grouped, extended], ignore_index=True)
                    
                    # Plot the mean F1 Score with a shaded area for the standard deviation
                    plt.plot(grouped['Layer'], grouped['mean'], label=f'{model_label}', marker='o', linestyle='-')
                    plt.fill_between(grouped['Layer'], grouped['mean'] - grouped['std'], grouped['mean'] + grouped['std'], alpha=0.2)
                
        plt.title(f'{test_type} F1 Scores for {model_base_name}')
        plt.xlabel('Layer')
        plt.ylabel('F1 Score')
        plt.legend(loc='best')
        plt.grid(True)

    plt.tight_layout()
    plt.savefig(f'mlp_vs_lr_{model_base_name}.png')
    plt.close()

# Example usage for specified models
model_bases = [
    'llama2_7b',
    'llama2_7b_chat',
    'llama2_13b',
    'llama3_8b',
    'llama3_8b_instruct',
    'llama2_70b',
    'llama2_70b_chat',
    'llama3_70b',
    'llama3_70b_instruct',
    'mistral_7b',
    'mistral_8x7b'
]

for base in model_bases:
    plot_model_comparisons(base)

import pandas as pd

# Results files mapped by model name
results_files = {
    'llama2_7b': 'logistic_regression_results_llama2_7b.csv',
    'llama2_7b_mlp': 'logistic_regression_results_llama2_7b_mlp.csv',
    'llama2_7b_chat': 'logistic_regression_results_llama2_7b_chat.csv',
    'llama2_7b_chat_mlp': 'logistic_regression_results_llama2_7b_chat_mlp.csv',
    'llama2_13b': 'logistic_regression_results_llama2_13b.csv',
    'llama2_13b_mlp': 'logistic_regression_results_llama2_13b_mlp.csv',
    'llama3_8b': 'logistic_regression_results_llama3_8b.csv',
    'llama3_8b_mlp': 'logistic_regression_results_llama3_8b_mlp.csv',
    'llama3_8b_instruct': 'logistic_regression_results_llama3_8b_instruct.csv',
    'llama3_8b_instruct_mlp': 'logistic_regression_results_llama3_8b_instruct_mlp.csv',
    'llama2_70b': 'logistic_regression_results_llama2_70b.csv',
    'llama2_70b_mlp': 'logistic_regression_results_llama2_70b_mlp.csv',
    'llama2_70b_chat': 'logistic_regression_results_llama2_70b_chat.csv',
    'llama2_70b_chat_mlp': 'logistic_regression_results_llama2_70b_chat_mlp.csv',
    'llama3_70b': 'logistic_regression_results_llama3_70b.csv',
    'llama3_70b_mlp': 'logistic_regression_results_llama3_70b_mlp.csv',
    'llama3_70b_instruct': 'logistic_regression_results_llama3_70b_instruct.csv',
    'llama3_70b_instruct_mlp': 'logistic_regression_results_llama3_70b_instruct_mlp.csv',
    'mistral_7b': 'logistic_regression_results_mistral_7b.csv',
    'mistral_7b_mlp': 'logistic_regression_results_mistral_7b_mlp.csv',
    'mistral_8x7b': 'logistic_regression_results_mistral_8x7b.csv',
    'mistral_8x7b_mlp': 'logistic_regression_results_mistral_8x7b_mlp.csv',
    'bert': 'logistic_regression_results_bert.csv',
    'bert_mlp': 'logistic_regression_results_bert_mlp.csv',
    'roberta': 'logistic_regression_results_roberta.csv',
    'roberta_mlp': 'logistic_regression_results_roberta_mlp.csv'
}

import pandas as pd

def generate_max_f1_summary():
    summary = pd.DataFrame()

    for model_label, file_path in results_files.items():
        data = pd.read_csv(file_path)  # Make sure your CSV has 'Layer' and 'F1 Score' columns
        # You would filter by test type here if your data combines them, or adjust the loading if they are separate.
        for test_type in ['In-Domain', 'Out-of-Domain', 'Hold-Out']:
            filtered_data = data[data['Test Type'] == test_type]
            max_f1 = filtered_data['F1 Score'].max()
            max_layer = filtered_data[filtered_data['F1 Score'] == max_f1]['Layer'].iloc[0]
            summary.loc[model_label, f'{test_type} Max F1'] = max_f1
            summary.loc[model_label, f'{test_type} Max Layer'] = max_layer

    return summary

# Generate the summary
max_f1_summary = generate_max_f1_summary()
max_f1_summary.to_csv('max_f1_summary.csv')
print(max_f1_summary)
