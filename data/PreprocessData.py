import pandas as pd
import warnings
import numpy as np


# Ignore all warnings
warnings.filterwarnings("ignore")


def is_integer(val):
    try:
        # Check if the float representation of a value is equal to its int representation
        if float(val) == int(val):
            return True
        else:
            return False
    except (ValueError, TypeError):
        # In case of ValueError or TypeError (i.e., when conversion to float/int is not possible)
        return False


# Discrimination, Exclusion, Toxicity (DET) Dataset Processing
csv_file_path = '/nethome/smaheshwari44/LLM-Safety-Assessment/LLM Safety Datasets/adult_content.csv'
df = pd.read_csv(csv_file_path)
df['Category'].replace({'Non_Adult': 0, 'Adult': 1}, inplace=True)
df.dropna(subset=['Description'], inplace=True)
df.dropna(subset=['Category'], inplace=True)
df.columns = ['Text', 'Label']
df = df[df['Label'].apply(is_integer)]
adult_content_df = df


csv_file_path = '/nethome/smaheshwari44/LLM-Safety-Assessment/LLM Safety Datasets/toxigen.csv'
df = pd.read_csv(csv_file_path)
df.dropna(subset=['Text'], inplace=True)
df.dropna(subset=['Label'], inplace=True)
df.columns = ['Text', 'Label']
df = df[df['Label'].apply(is_integer)]
toxigen_df = df


csv_file_path = '/nethome/smaheshwari44/LLM-Safety-Assessment/LLM Safety Datasets/hate_speech.csv'
df = pd.read_csv(csv_file_path)
df['label'].replace({'nothate': 0, 'hate': 1}, inplace=True)
df.dropna(subset=['text'], inplace=True)
df.dropna(subset=['label'], inplace=True)
df = df[['text', 'label']]
df.columns = ['Text', 'Label']
df = df[df['Label'].apply(is_integer)]
hate_speech_df = df

DET_df = pd.concat([adult_content_df, toxigen_df, hate_speech_df], axis=0).reset_index(drop=True)
DET_df['Global_Label'] = "DET"
DET_df = DET_df.groupby("Label").apply(lambda x: x.sample(n=10000, random_state=1)).reset_index(drop=True)
print(DET_df['Label'].value_counts())

# Human Chatbot Interaction Harms (HCIH) Dataset Processing
csv_file_path = '/nethome/smaheshwari44/LLM-Safety-Assessment/LLM Safety Datasets/student_anxiety.csv'
df = pd.read_csv(csv_file_path)
df.dropna(subset=['text'], inplace=True)
df.dropna(subset=['label'], inplace=True)
df.columns = ['Text', 'Label']
df['Label'] = df['Label'].astype(int)
df = df[df['Label'].apply(is_integer)]
student_anxiety_df = df




csv_file_path = '/nethome/smaheshwari44/LLM-Safety-Assessment/LLM Safety Datasets/Suicide_Detection.csv'
df = pd.read_csv(csv_file_path,encoding='ISO-8859-1')
df['class'].replace({'non-suicide': 0, 'suicide': 1}, inplace=True)
df.dropna(subset=['text'], inplace=True)
df.dropna(subset=['class'], inplace=True)
df = df[['text', 'class']]
df.columns = ['Text', 'Label']
df = df[df['Label'].apply(is_integer)]
suicide_detection_df = df


HCIH_df = pd.concat([student_anxiety_df, suicide_detection_df], axis=0).reset_index(drop=True)
HCIH_df['Global_Label'] = "HCIH"
HCIH_df= HCIH_df.groupby("Label").apply(lambda x: x.sample(n=10000, random_state=1)).reset_index(drop=True)
print(HCIH_df['Label'].value_counts())


# Malicious Uses (MU) Dataset Processing
csv_file_path = '/nethome/smaheshwari44/LLM-Safety-Assessment/LLM Safety Datasets/suspicious tweets.csv'
df = pd.read_csv(csv_file_path)
df['label'].replace({1: 0, 0: 1}, inplace=True)
df.dropna(subset=['message'], inplace=True)
df.dropna(subset=['label'], inplace=True)
df.columns = ['Text', 'Label']
df['Label'] = df['Label'].astype(int)
df = df[df['Label'].apply(is_integer)]
suspicious_tweet_df = df




csv_file_path = '/nethome/smaheshwari44/LLM-Safety-Assessment/LLM Safety Datasets/bullying.csv'
df = pd.read_csv(csv_file_path)
df.dropna(subset=['Text'], inplace=True)
df.dropna(subset=['oh_label'], inplace=True)
df.columns = ['Text', 'Label']
df = df[df['Label'].apply(is_integer)]
bullying_df = df


MU_df = pd.concat([suspicious_tweet_df, bullying_df], axis=0).reset_index(drop=True)
MU_df['Global_Label'] = "MU"
MU_df= MU_df.groupby("Label").apply(lambda x: x.sample(n=10000, random_state=1)).reset_index(drop=True)
print(MU_df['Label'].value_counts())


# Misinformations Harms (MH) Dataset Processing
csv_file_path = '/nethome/smaheshwari44/LLM-Safety-Assessment/LLM Safety Datasets/covid_fake_news.csv'
df = pd.read_csv(csv_file_path)
df['label'].replace({"real": 0, "fake": 1}, inplace=True)
df.dropna(subset=['tweet'], inplace=True)
df.dropna(subset=['label'], inplace=True)
df = df[['tweet', 'label']]
df.columns = ['Text', 'Label']
df['Label'] = df['Label'].astype(int)
df = df[df['Label'].apply(is_integer)]
misinformation_harms_df = df




csv_file_path = '/nethome/smaheshwari44/LLM-Safety-Assessment/LLM Safety Datasets/true_false.csv'
df = pd.read_csv(csv_file_path)
df['label'].replace({1: 0, 0: 1}, inplace=True)
df.dropna(subset=['statement'], inplace=True)
df.dropna(subset=['label'], inplace=True)
df.columns = ['Text', 'Label']
df = df[df['Label'].apply(is_integer)]
true_false_df = df


MH_df = pd.concat([misinformation_harms_df,true_false_df], axis=0).reset_index(drop=True)
MH_df['Global_Label'] = "MH"
MH_df= MH_df.groupby("Label").apply(lambda x: x.sample(n=10000, random_state=1)).reset_index(drop=True)
print(MH_df['Label'].value_counts())

#Harmful QA (HQA) Dataset Processing

csv_file_path = '/nethome/smaheshwari44/LLM-Safety-Assessment/LLM Safety Datasets/HQA1.csv'
df = pd.read_csv(csv_file_path)
df.dropna(subset=['Text'], inplace=True)
df.dropna(subset=['Label'], inplace=True)
df.columns = ['Text', 'Label']
df = df[df['Label'].apply(is_integer)]
HQA1_df = df


csv_file_path = '/nethome/smaheshwari44/LLM-Safety-Assessment/LLM Safety Datasets/HQA2.csv'
df = pd.read_csv(csv_file_path)
df.dropna(subset=['Text'], inplace=True)
df.dropna(subset=['Label'], inplace=True)
df.columns = ['Text', 'Label']
df = df[df['Label'].apply(is_integer)]
HQA2_df = df

HQA_df = pd.concat([HQA1_df,HQA2_df], axis=0).reset_index(drop=True)
HQA_df['Global_Label'] = "HQA"
HQA_df= HQA_df.groupby("Label").apply(lambda x: x.sample(n=10000, random_state=1)).reset_index(drop=True)
print(HQA_df['Label'].value_counts())


LLM_HARMS_df = pd.concat([DET_df, HCIH_df, MU_df, MH_df, HQA_df], axis=0).reset_index(drop=True)
print(LLM_HARMS_df['Label'].value_counts())
print(LLM_HARMS_df['Global_Label'].value_counts())
print(LLM_HARMS_df.head())
print(LLM_HARMS_df.count())
LLM_HARMS_df.to_csv('LLM_HARMS_df.csv', index=False)





