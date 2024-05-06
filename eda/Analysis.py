# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
# import numpy as np

# # Results files mapped by model name
# results_files = {
#     'llama2_7b': 'logistic_regression_results_llama2_7b.csv',
#     'llama2_7b_chat': 'logistic_regression_results_llama2_7b_chat.csv',
#     'llama2_13b': 'logistic_regression_results_llama2_13b.csv',
#     'llama3_8b': 'logistic_regression_results_llama3_8b.csv',
#     'llama3_8b_instruct': 'logistic_regression_results_llama3_8b_instruct.csv',
#     'llama2_70b': 'logistic_regression_results_llama2_70b.csv',
#     'llama3_70b': 'logistic_regression_results_llama3_70b.csv',
#     'mistral_7b': 'logistic_regression_results_mistral_7b.csv',
#     'mistral_8x7b': 'logistic_regression_results_mistral_8x7b.csv',
#     'bert': 'logistic_regression_results_bert.csv',
#     'roberta': 'logistic_regression_results_roberta.csv',
# }

# # Set plot style
# sns.set(style="whitegrid")

# # Determine the maximum layer number across all files to standardize the x-axis
# max_layer = max(pd.read_csv(file)['Layer'].max() for file in results_files.values())

# # Get test types from one example file
# example_data = pd.read_csv(next(iter(results_files.values())))
# test_types = ['In-Domain', 'Out-of-Domain', 'Hold-Out']

# # Loop through each test type to create plots
# for test_type in test_types:
#     plt.figure(figsize=(14, 8))  # Set the figure size for each plot

#     # Plot data from each model
#     for model_label, file_path in results_files.items():
#         data = pd.read_csv(file_path)
#         subset = data[data['Test Type'] == test_type]

#         # Group by layer and average F1 Scores across domains
#         grouped = subset.groupby('Layer')['F1 Score'].mean().reset_index()

#         # Extend F1 scores for models with fewer layers
#         if grouped['Layer'].max() < max_layer:
#             last_f1 = grouped['F1 Score'].iloc[-1]
#             missing_layers = range(grouped['Layer'].max() + 1, max_layer + 1)
#             extended_f1 = pd.DataFrame({
#                 'Layer': missing_layers,
#                 'F1 Score': [last_f1] * len(missing_layers)
#             })
#             grouped = pd.concat([grouped, extended_f1], ignore_index=True)

#         plt.plot(grouped['Layer'], grouped['F1 Score'], label=f'{model_label}', marker='o', linestyle='-')

#     # Plot customization
#     plt.title(f'Average F1 Scores Across Domains - {test_type}', fontsize=16)
#     plt.xlabel('Layer', fontsize=14)
#     plt.ylabel('F1 Score', fontsize=14)
#     plt.legend(title='Model', loc='best')
#     plt.grid(True)

#     # Save each plot to a separate file
#     plt.tight_layout()
#     plt.savefig(f'average_f1_scores_{test_type}.png', format='png', dpi=300)
#     plt.close()  # Close the figure to free memory


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

# Set plot style
sns.set(style="whitegrid")

# Determine the maximum layer number across all files to standardize the x-axis
max_layer = max(pd.read_csv(file)['Layer'].max() for file in results_files.values())

# Get test types from one example file
example_data = pd.read_csv(next(iter(results_files.values())))
test_types = ['In-Domain', 'Out-of-Domain', 'Hold-Out']

# Loop through each test type to create plots
for test_type in test_types:
    plt.figure(figsize=(14, 8))  # Set the figure size for each plot

    # Plot data from each model
    for model_label, file_path in results_files.items():
        data = pd.read_csv(file_path)
        subset = data[data['Test Type'] == test_type]

        # Group by layer and calculate mean and standard deviation of F1 Scores across domains
        grouped = subset.groupby('Layer')['F1 Score'].agg(['mean', 'std']).reset_index()

        # Extend F1 scores for models with fewer layers
        if grouped['Layer'].max() < max_layer:
            last_mean = grouped['mean'].iloc[-1]
            last_std = grouped['std'].iloc[-1]
            missing_layers = range(grouped['Layer'].max() + 1, max_layer + 1)
            extended_f1 = pd.DataFrame({
                'Layer': missing_layers,
                'mean': [last_mean] * len(missing_layers),
                'std': [last_std] * len(missing_layers)
            })
            grouped = pd.concat([grouped, extended_f1], ignore_index=True)

        # Plot the mean F1 Score with standard deviation as the shaded area
        plt.plot(grouped['Layer'], grouped['mean'], label=f'{model_label}', marker='o', linestyle='-')
        plt.fill_between(grouped['Layer'], grouped['mean'] - grouped['std'], grouped['mean'] + grouped['std'], alpha=0.2)

    # Plot customization
    plt.title(f'Average F1 Scores with Standard Deviation - {test_type}', fontsize=16)
    plt.xlabel('Layer', fontsize=14)
    plt.ylabel('F1 Score', fontsize=14)
    plt.legend(title='Model', loc='best')
    plt.grid(True)

    # Save each plot to a separate file
    plt.tight_layout()
    plt.savefig(f'average_f1_scores_with_std_{test_type}.png', format='png', dpi=300)
    plt.close()  # Close the figure to free memory
