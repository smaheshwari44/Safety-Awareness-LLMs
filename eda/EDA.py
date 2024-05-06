import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings


# Ignore all warnings
warnings.filterwarnings("ignore")


def top_tfidf_words(text_series, top_n=1000):
    # Initialize the TF-IDF Vectorizer
    vectorizer = TfidfVectorizer(max_features=top_n, stop_words='english')

    # Fit and transform the text data
    tfidf_matrix = vectorizer.fit_transform(text_series)

    # Get feature names to use as DataFrame column headers
    feature_names = vectorizer.get_feature_names_out()

    # Get the dense version of the matrix to work with the scores
    dense_tfidf = tfidf_matrix.todense()

    # Sum tfidf scoring for each term in the documents
    sums = dense_tfidf.sum(axis=0)

    # Connect terms with their sums
    data = []
    for col, term in enumerate(feature_names):
        data.append((term, sums[0, col]))

    # Create a DataFrame with the terms and their corresponding TFIDF scores
    ranking = pd.DataFrame(data, columns=['term', 'score'])
    ranking = ranking.sort_values('score', ascending=False)

    return ranking

df = pd.read_csv("LLM_HARMS_df.csv")
top_terms_by_group = {}
groups = df.groupby(['Global_Label', 'Label'])

for (global_label, label), group_data in groups:
    top_terms = top_tfidf_words(group_data['Text'], 1000)
    top_terms_by_group[(global_label, label)] = top_terms
    print(f"Top terms for GlobalLabel: {global_label}, Label: {label}")
    print(top_terms.head()) 
    
    
# Assuming top_terms_by_group is available and filled as per previous discussions
domains = df['Global_Label'].unique()
safe_overlap = pd.DataFrame(index=domains, columns=domains, data=0)
unsafe_overlap = pd.DataFrame(index=domains, columns=domains, data=0)

for domain1 in domains:
    for domain2 in domains:
        if domain1 != domain2:
            # Get safe words for each domain
            safe_words1 = set(top_terms_by_group[(domain1, 0)]['term'])
            safe_words2 = set(top_terms_by_group[(domain2, 0)]['term'])
            # Calculate intersection and normalize by the smallest set of top words
            safe_overlap.loc[domain1, domain2] = len(safe_words1.intersection(safe_words2)) / min(len(safe_words1), len(safe_words2))
            
            # Get unsafe words for each domain
            unsafe_words1 = set(top_terms_by_group[(domain1, 1)]['term'])
            unsafe_words2 = set(top_terms_by_group[(domain2, 1)]['term'])
            # Calculate intersection
            unsafe_overlap.loc[domain1, domain2] = len(unsafe_words1.intersection(unsafe_words2)) / min(len(unsafe_words1), len(unsafe_words2))
            

# Heatmap for safe words overlap
plt.figure(figsize=(10, 8))
sns.heatmap(safe_overlap, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Overlap of Top 1000 Safe Words Between Domains')
plt.savefig('safe_words_overlap.png', dpi=300, bbox_inches='tight')  # Adjust dpi for resolution and format as needed
plt.close()  # Close the plot to free memory

# Heatmap for unsafe words overlap
plt.figure(figsize=(10, 8))
sns.heatmap(unsafe_overlap, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Overlap of Top 1000 Unsafe Words Between Domains')
plt.savefig('unsafe_words_overlap.png', dpi=300, bbox_inches='tight')  # Adjust dpi for resolution and format as needed
plt.close()  # Close the plot to free memory

