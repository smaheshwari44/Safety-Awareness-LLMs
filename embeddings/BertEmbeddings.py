# import os
# import torch
# import pandas as pd
# from torch.utils.data import DataLoader
# import pickle
# from transformers import BertTokenizer, BertModel
# from tabulate import tabulate
# from torch.quantization import quantize_dynamic

# class BertEmbeddingHelper:
#     def __init__(self, model_name='bert-base-uncased'):
#         self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#         self.tokenizer = BertTokenizer.from_pretrained(model_name)
#         self.model = BertModel.from_pretrained(model_name)
#         self.model.eval()
#         self.model.to(self.device)

#     def get_bert_embeddings(self, text):
#         encoded_input = self.tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=512)
#         encoded_input = {key: val.to(self.device) for key, val in encoded_input.items()}
#         with torch.no_grad():
#             output = self.model(**encoded_input)
#         embeddings = output.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()
#         return embeddings

#     def save_embeddings(self, text, filename):
#         embeddings = self.get_bert_embeddings(text)
#         df = pd.DataFrame([embeddings], columns=[f'dim_{i}' for i in range(embeddings.shape[0])])
#         df['text'] = text
#         df.to_csv(filename, index=False)
#         print("Embeddings saved to", filename)

# # Example usage
# bert_helper = BertEmbeddingHelper()

# data = pd.read_csv("LLM_HARMS_df.csv")

# activations = []
# count = 0
# for index, row in data.iterrows():
#     count+=1
#     text = row['Text']
#     emb = bert_helper.get_bert_embeddings(text)
#     print(count)
#     activations.append(emb)
    
# result = {}
# result['Activations'] = {}
# result['Activations']['1'] = activations
# result['Labels'] = list(data['Label'])
# result['Global_Labels'] = list(data["Global_Label"])
# pickle.dump(result, open("LLM_HARMS_BERT_EMBEDDINGS.pkl", "wb"))


import os
import torch
import pandas as pd
from torch.utils.data import DataLoader
import pickle
from transformers import RobertaTokenizer, RobertaModel
from tabulate import tabulate
from torch.quantization import quantize_dynamic

class RobertaEmbeddingHelper:
    def __init__(self, model_name='roberta-base'):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = RobertaTokenizer.from_pretrained(model_name)
        self.model = RobertaModel.from_pretrained(model_name)
        self.model.eval()
        self.model.to(self.device)

    def get_roberta_embeddings(self, text):
        encoded_input = self.tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=512)
        encoded_input = {key: val.to(self.device) for key, val in encoded_input.items()}
        with torch.no_grad():
            output = self.model(**encoded_input)
        embeddings = output.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()
        return embeddings

    def save_embeddings(self, text, filename):
        embeddings = self.get_roberta_embeddings(text)
        df = pd.DataFrame([embeddings], columns=[f'dim_{i}' for i in range(embeddings.shape[0])])
        df['text'] = text
        df.to_csv(filename, index=False)
        print("Embeddings saved to", filename)

# Example usage
roberta_helper = RobertaEmbeddingHelper()

data = pd.read_csv("LLM_HARMS_df.csv")

activations = []
count = 0
for index, row in data.iterrows():
    count += 1
    text = row['Text']
    emb = roberta_helper.get_roberta_embeddings(text)
    print(count)
    activations.append(emb)
    
result = {}
result['Activations'] = {}
result['Activations']['1'] = activations
result['Labels'] = list(data['Label'])
result['Global_Labels'] = list(data["Global_Label"])
pickle.dump(result, open("LLM_HARMS_ROBERTA_EMBEDDINGS.pkl", "wb"))

