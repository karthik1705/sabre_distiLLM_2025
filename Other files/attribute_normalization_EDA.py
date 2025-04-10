#%%
import pandas as pd
from collections import Counter
import re
import spacy
from tqdm import tqdm
from collections import defaultdict
from fuzzywuzzy import fuzz, process
import numpy as np

#%%
dataset_raw = pd.read_csv('../Data/Product_Normalization_GRI.csv')
dataset_raw.head()

#%%
# Accessible Room without ADA or Accessible keywords
# Find rooms marked as Accessible but without ADA/Access/ible keywords in description => 0
access_analysis = dataset_raw[
    (dataset_raw['Guest Room Info'] == 'Accessible Room') & 
    (~dataset_raw['Room Description'].str.contains('ADA|ACC|ACESS', case=False, na=False))
]

# Display count and sample of these rows
print(f"\nNumber of rooms marked Accessible without ADA/Accessible keywords: {len(access_analysis)}")
print("\nSample of these rows:")
print(access_analysis[['Room Description', 'Guest Room Info']].head())


#%%

# Find rows where 'ADA|ACC\ACESS' appears in Room Description but RoomType is not 'Accessible Room'
access_analysis = dataset_raw[
    (dataset_raw['Room Description'].str.contains('ADA|ACC|ACESS', case=False, na=False)) & 
    (dataset_raw['Guest Room Info'] != 'Accessible Room')
]

# Display count and the matching rows
print(f"Number of rows with 'Accessible Room' in description but not marked as Accessible Room: {len(access_analysis)}")
print("\nSample of these rows:")
print(access_analysis[['Room Description', 'Guest Room Info']])

access_analysis['Guest Room Info'].value_counts()



#%%
# Tokenize and get word frequencies from Room Description

# Function to clean and tokenize text
def tokenize_text(text):
    if pd.isna(text):  # Handle NaN values
        return []
    
    # Convert to lowercase
    text = text.lower()
    
    # Replace specific compound terms before tokenization
    compound_terms = {
        'bed room': 'bedroom'
    }
    
    for term, replacement in compound_terms.items():
        text = text.replace(term, replacement)
    
    # Remove special characters and numbers, keep only letters
    words = re.findall(r'\b[a-zA-Z]+\b', text)
    return words

# Get all words from Room Description
all_words = []
for description in dataset_raw['Room Description']:
    all_words.extend(tokenize_text(description))

# Count word frequencies
word_freq = Counter(all_words)

# Convert to DataFrame for better visualization
word_freq_df = pd.DataFrame.from_dict(word_freq, orient='index', columns=['count'])
word_freq_df = word_freq_df.sort_values('count', ascending=False)

#%%

# Display top 20 most common words
print("Top 20 most frequent words in Room Description:")
print(word_freq_df.head(20))

# %%
dataset_raw['Guest Room Info'].value_counts()



# %%
# Text Analysis and Clusterization using Embeddings

# Import required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# For generating text embeddings
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import LabelEncoder
from sklearn.semi_supervised import LabelPropagation
from umap import UMAP

# For clustering
from sklearn.cluster import KMeans

# For visualization (dimensionality reduction)
from sklearn.manifold import TSNE

# %%
# ----------------------------
# 1. Sample Data Preparation
# ----------------------------
# Replace this sample data with your actual dataset
data = {
    "description": dataset_raw['Room Description'],
    "current_label": dataset_raw['Guest Room Info']
}

# Create a DataFrame
df = pd.DataFrame(data)

# ----------------------------------
# 2. Generate Embeddings for the Text
# ----------------------------------
# Load a pre-trained sentence transformer model
# You can choose another model if needed (e.g., 'all-MiniLM-L6-v2', 'paraphrase-MiniLM-L6-v2', etc.)
model = SentenceTransformer('all-MiniLM-L6-v2')

# Compute embeddings for each room description
embeddings = model.encode(df['description'].tolist(), convert_to_numpy=True)

# ----------------------------
# 3. Clustering the Embeddings
# ----------------------------
# Define the number of clusters (adjust as needed)
num_clusters = 35

# Use KMeans clustering on the embeddings
kmeans = KMeans(n_clusters=num_clusters, random_state=42)
clusters = kmeans.fit_predict(embeddings)

# Add the cluster labels to the DataFrame
df['cluster'] = clusters

# ----------------------------
# 4. Visualize the Clusters using t-SNE
# ----------------------------
# Reduce embeddings to 2 dimensions using t-SNE for visualization
tsne = TSNE(n_components=2, random_state=42)
embeddings_2d = tsne.fit_transform(embeddings)

# Add t-SNE coordinates to the DataFrame
df['tsne_x'] = embeddings_2d[:, 0]
df['tsne_y'] = embeddings_2d[:, 1]

#%%
# Plot the clusters
plt.figure(figsize=(10, 8))
sns.scatterplot(x='tsne_x', y='tsne_y', hue='cluster', data=df, palette='tab10', s=100, alpha=0.8)
# Optionally, annotate the points with the room description (or part of it)
#for i, row in df.iterrows():
#    plt.text(row['tsne_x'] + 0.2, row['tsne_y'] + 0.2, row['description'], fontsize=9)

plt.title("t-SNE Visualization of Room Description Clusters")
plt.xlabel("t-SNE Dimension 1")
plt.ylabel("t-SNE Dimension 2")
plt.legend(title="Cluster")
plt.show()

# ----------------------------
# 5. Review the Cluster Assignments
# ----------------------------
# Print the DataFrame with room descriptions, current labels, and assigned cluster
print(df[['description', 'current_label', 'cluster']])

# %%
df.to_csv('minilm_kmeans_clustered.csv', index=False)

#%%
df.groupby(['cluster', 'current_label']).size().reset_index(name='count')





#%% ##### TOPIC MODELING #####
# Topic Modeling Prep
dataset_normalized_attributes = pd.read_excel('Data/Normalized_product_attribute_name.xlsx', sheet_name = 'Normalized Product Attributes')
dataset_normalized_attributes

room_types = dataset_normalized_attributes['RoomType'].dropna().unique()
room_types_list = room_types.tolist()

# %%

# Collating the Normalized Product Attribute data to be excluded through custom stop words
bed_types = dataset_normalized_attributes['BedType'].dropna().unique()
bed_types_list = bed_types.tolist()

rate_plan_inclusives_raw = dataset_normalized_attributes['Rateplan Incl/Value-adds'].dropna().unique()
rate_plan_inclusives = rate_plan_inclusives_raw.tolist()

room_amenities_raw = dataset_normalized_attributes['Room Amenities'].dropna().unique()
room_amenities_list = room_amenities_raw.tolist()

meal_plan_raw = dataset_normalized_attributes['Mealplan'].dropna().unique()
meal_plan_list = meal_plan_raw.tolist()

room_view_raw = dataset_normalized_attributes['RoomView'].dropna().unique()
room_view_list = room_view_raw.tolist()

taxes_fees_raw = dataset_normalized_attributes['Taxes & Fees'].dropna().unique()
taxes_fees_list = taxes_fees_raw.tolist()



#%%
#Topic modeling
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from bertopic import BERTopic
import plotly.express as px

#%%
# Clean and preprocess text (reusing your existing tokenize_text function)
def preprocess_for_topic_modeling(text):
    if pd.isna(text):
        return ""
    words = tokenize_text(text)
    # Remove very common words that might not be in standard stop words
    custom_stops = {'a', 'an', 'the', 'in', 'on', 'at', 'with', 'and', 'or', 'to', 'for'}
    words = [w for w in words if w not in custom_stops]
    return ' '.join(words)


#%%
# Preprocess all descriptions
processed_texts = dataset_raw['Room Description'].apply(preprocess_for_topic_modeling)

# 1. LDA Approach
# --------------
def run_lda_analysis(texts, n_topics=35):
    # Create document-term matrix
    vectorizer = CountVectorizer(max_df=0.95, min_df=2, stop_words='english')
    doc_term_matrix = vectorizer.fit_transform(texts)
    
    # Train LDA model
    lda = LatentDirichletAllocation(
        n_components=n_topics,
        random_state=42,
        max_iter=20
    )
    lda_output = lda.fit_transform(doc_term_matrix)
    
    # Print top words for each topic
    feature_names = vectorizer.get_feature_names_out()
    for topic_idx, topic in enumerate(lda.components_):
        top_words = [feature_names[i] for i in topic.argsort()[:-10:-1]]
        print(f"Topic {topic_idx + 1}: {', '.join(top_words)}")
    
    return lda_output, lda, vectorizer

# 2. BERTopic Approach
# -------------------
def run_bertopic_analysis(texts, n_topics=35):
    topic_model = BERTopic(nr_topics=n_topics)
    topics, probs = topic_model.fit_transform(texts)
    
    # Get topic info
    topic_info = topic_model.get_topic_info()
    
    # Print topics
    for topic in topic_model.get_topics():
        if topic != -1:  # -1 is reserved for outliers
            print(f"Topic {topic}: {topic_model.get_topic(topic)}")
    
    return topics, probs, topic_model

#%%
# Run both analyses
print("Running LDA Analysis...")
lda_output, lda_model, vectorizer = run_lda_analysis(processed_texts)

print("\nRunning BERTopic Analysis...")
topics, probs, bertopic_model = run_bertopic_analysis(processed_texts)

#%%
# Visualization for LDA
# --------------------
def visualize_topics_distribution(lda_output):
    # Get the dominant topic for each document
    dominant_topics = np.argmax(lda_output, axis=1)
    topic_counts = pd.Series(dominant_topics).value_counts().sort_index()
    
    # Create visualization
    fig = px.bar(
        x=topic_counts.index + 1,  # Add 1 to make topics 1-based
        y=topic_counts.values,
        labels={'x': 'Topic Number', 'y': 'Number of Documents'},
        title='Distribution of Dominant Topics Across Documents'
    )
    fig.show()

# Visualization for BERTopic
# -------------------------
def visualize_bertopic_results(topic_model):
    # Topic visualization
    topic_model.visualize_topics()
    
    # Topic hierarchy
    topic_model.visualize_hierarchy()
    
    # Topic similarity
    topic_model.visualize_heatmap()

#%%
# Run visualizations
print("\nVisualizing LDA Results...")
visualize_topics_distribution(lda_output)

print("\nVisualizing BERTopic Results...")
visualize_bertopic_results(bertopic_model)

# Add topics to original dataframe
dataset_raw['LDA_Topic'] = np.argmax(lda_output, axis=1)
dataset_raw['BERTopic'] = topics

#%%
lda_output

#%%
bertopic_model.visualize_topics()

#%%
# Topic hierarchy
bertopic_model.visualize_hierarchy()

#%%
# Topic similarity
bertopic_model.visualize_heatmap()

#%% ##### TF-IDF Topic Modeling #####
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF  # Non-negative Matrix Factorization - works well with TF-IDF
from sklearn.decomposition import LatentDirichletAllocation
import numpy as np
import pandas as pd

#%%
def run_tfidf_topic_modeling(texts, n_topics=35, top_n_words=10):
    # Create TF-IDF matrix
    tfidf_vectorizer = TfidfVectorizer(
        max_df=0.95,         # Ignore terms that appear in >95% of docs
        min_df=2,           # Ignore terms that appear in <2 docs
        stop_words='english',
        ngram_range=(1, 2)  # Consider both unigrams and bigrams
    )
    tfidf_matrix = tfidf_vectorizer.fit_transform(texts)
    
    # 1. Using NMF (works better with TF-IDF)
    print("Running NMF with TF-IDF...")
    nmf_model = NMF(n_components=n_topics, random_state=42)
    nmf_output = nmf_model.fit_transform(tfidf_matrix)
    
    # 2. Using LDA with TF-IDF (less common but possible)
    print("\nRunning LDA with TF-IDF...")
    lda_model = LatentDirichletAllocation(
        n_components=n_topics,
        random_state=42
    )
    lda_output = lda_model.fit_transform(tfidf_matrix)
    
    # Get feature names (words)
    feature_names = tfidf_vectorizer.get_feature_names_out()

    return {
        'features': feature_names,
        'vectorizer': tfidf_vectorizer,
        'nmf_model': nmf_model,
        'nmf_output': nmf_output,
        'lda_model': lda_model,
        'lda_output': lda_output
    }

#%%
# Run the analysis
processed_texts = dataset_raw['Room Description'].apply(preprocess_for_topic_modeling)
results = run_tfidf_topic_modeling(processed_texts, n_topics=35)

#%%
# Print top words for each topic from NMF
print("\nNMF Topics:")
for topic_idx, topic in enumerate(results['nmf_model'].components_):
    top_words_idx = topic.argsort()[:-10-1:-1]
    feature_names = results['features']
    top_words = [feature_names[i] for i in top_words_idx]
    print(f"Topic {topic_idx + 1}: {', '.join(top_words)}")

#%%
# Print top words for each topic from LDA
print("\nLDA Topics:")
for topic_idx, topic in enumerate(lda_model.components_):
    top_words_idx = topic.argsort()[:-10-1:-1]
    feature_names = results['features']
    top_words = [feature_names[i] for i in top_words_idx]
    print(f"Topic {topic_idx + 1}: {', '.join(top_words)}")
    


#%%
# Add topic assignments to the original dataframe
dataset_raw['NMF_Topic'] = np.argmax(results['nmf_output'], axis=1)
dataset_raw['TFIDF_LDA_Topic'] = np.argmax(results['lda_output'], axis=1)

#%%
# Visualize topic distribution
import plotly.express as px

def plot_topic_distribution(topic_assignments, title):
    topic_counts = pd.Series(topic_assignments).value_counts().sort_index()
    fig = px.bar(
        x=topic_counts.index + 1,
        y=topic_counts.values,
        labels={'x': 'Topic Number', 'y': 'Number of Documents'},
        title=title
    )
    fig.show()

# Plot distributions
plot_topic_distribution(dataset_raw['NMF_Topic'], 'Distribution of NMF Topics')
plot_topic_distribution(dataset_raw['TFIDF_LDA_Topic'], 'Distribution of TF-IDF LDA Topics')





#%%
#Semi-supervised clustering using SentenceTransformer embeddings
from sklearn.preprocessing import LabelEncoder
from sklearn.semi_supervised import LabelPropagation
from umap import UMAP
import matplotlib.pyplot as plt

#%%
def run_semi_supervised_clustering(descriptions, labels, n_components=2):
    # 1. Prepare the labels
    le = LabelEncoder()
    encoded_labels = le.fit_transform(labels)
    
    # 2. Generate embeddings
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(descriptions, convert_to_numpy=True)
    
    # 3. Reduce dimensionality while preserving label information
    umap_supervised = UMAP(
        n_components=n_components,
        random_state=42,
        n_neighbors=15,
        min_dist=0.1,
        target_metric='categorical',  # This helps preserve label structure
        target_weight=0.5  # Balance between preserving labels and topology (0.0 to 1.0)
    )
    
    reduced_embeddings = umap_supervised.fit_transform(
        embeddings,
        y=encoded_labels  # Providing labels for semi-supervised learning
    )
    
    # 4. Use LabelPropagation for semi-supervised learning
    label_prop_model = LabelPropagation(kernel='knn', n_neighbors=7)
    label_prop_model.fit(reduced_embeddings, encoded_labels)
    
    # Get predicted labels
    predicted_labels = label_prop_model.predict(reduced_embeddings)
    predicted_label_names = le.inverse_transform(predicted_labels)
    
    # 5. Create visualization
    plt.figure(figsize=(12, 8))
    scatter = plt.scatter(
        reduced_embeddings[:, 0],
        reduced_embeddings[:, 1],
        c=predicted_labels,
        cmap='tab20',
        alpha=0.6
    )
    plt.title('Semi-supervised Clustering with Room Type Labels')
    plt.colorbar(scatter)
    
    # Add legend with original label names
    unique_labels = le.inverse_transform(np.unique(predicted_labels))
    legend_elements = [plt.Line2D([0], [0], marker='o', color='w', 
                                markerfacecolor=plt.cm.tab20(i/len(unique_labels)), 
                                label=label, markersize=10)
                      for i, label in enumerate(unique_labels)]
    plt.legend(handles=legend_elements, title='Room Types', bbox_to_anchor=(1.05, 1))
    plt.tight_layout()
    plt.show()
    
    return reduced_embeddings, predicted_label_names, le

#%%
# Run the analysis
reduced_embeddings, predicted_labels, label_encoder = run_semi_supervised_clustering(
    dataset_raw['Room Description'],
    dataset_raw['Guest Room Info']
)

#%%
# Add results to dataframe
dataset_raw['Supervised_Cluster'] = predicted_labels

# Analyze agreement between original and predicted labels
comparison_df = pd.DataFrame({
    'Original_Label': dataset_raw['Guest Room Info'],
    'Predicted_Label': predicted_labels
})

print("\nLabel Distribution Comparison:")
print("\nOriginal Labels:")
print(comparison_df['Original_Label'].value_counts())
print("\nPredicted Labels:")
print(comparison_df['Predicted_Label'].value_counts())

# Calculate agreement percentage
agreement = (comparison_df['Original_Label'] == comparison_df['Predicted_Label']).mean() * 100
print(f"\nLabel Agreement: {agreement:.2f}%")

#%%
# Analyze cases where predictions differ from original labels
disagreements = comparison_df[comparison_df['Original_Label'] != comparison_df['Predicted_Label']]
print("\nSample of Disagreements:")
sample_disagreements = disagreements.join(dataset_raw['Room Description']).sample(5)
print(sample_disagreements)




#%%
##### Need to try clustering with simplified labels
def run_simplified_clustering(descriptions, labels, n_components=2):
    # 1. Simplify labels by grouping similar categories
    label_mapping = {
        'Deluxe Room': 'Deluxe',
        'Deluxe Suite': 'Deluxe',
        'Executive/Club Room': 'Executive',
        'Executive/Club Suite': 'Executive',
        'Classic Room': 'Standard',
        'Standard Room': 'Standard',
        'Superior Room': 'Standard',
        'Premier Room': 'Premium',
        'Premier Suite': 'Premium',
        'Luxury Room': 'Premium',
        'Luxury Suite': 'Premium',
        'Presidential Suite': 'Premium',
        'Penthouse': 'Premium',

        # Add more mappings as needed
    }
    
    simplified_labels = labels.map(lambda x: label_mapping.get(x, x))
    
    # 2. Run the same clustering but with simplified labels
    le = LabelEncoder()
    encoded_labels = le.fit_transform(simplified_labels)
    
    # Generate embeddings
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(descriptions, convert_to_numpy=True)
    
    # UMAP with higher neighbor count for more global structure
    umap_supervised = UMAP(
        n_components=n_components,
        random_state=42,
        n_neighbors=30,  # Increased from 15
        min_dist=0.2,    # Increased from 0.1
        target_metric='categorical',
        target_weight=0.7  # Increased label influence
    )
    
    reduced_embeddings = umap_supervised.fit_transform(embeddings, y=encoded_labels)
    
    # Visualization with simplified categories
    plt.figure(figsize=(12, 8))
    scatter = plt.scatter(
        reduced_embeddings[:, 0],
        reduced_embeddings[:, 1],
        c=encoded_labels,
        cmap='tab10',  # Using a colormap with fewer colors
        alpha=0.6
    )
    plt.title('Simplified Room Type Clustering')
    
    # Add legend with simplified labels
    unique_labels = le.inverse_transform(np.unique(encoded_labels))
    legend_elements = [plt.Line2D([0], [0], marker='o', color='w', 
                                markerfacecolor=plt.cm.tab10(i/len(unique_labels)), 
                                label=label, markersize=10)
                      for i, label in enumerate(unique_labels)]
    plt.legend(handles=legend_elements, title='Room Types', bbox_to_anchor=(1.05, 1))
    plt.tight_layout()
    plt.show()
    
    return reduced_embeddings, simplified_labels, le

#%%
# Run the simplified analysis
reduced_embeddings, simplified_labels, label_encoder = run_simplified_clustering(
    dataset_raw['Room Description'],
    dataset_raw['Guest Room Info']
)

#%%
# Analyze the distribution of simplified categories
print("\nSimplified Label Distribution:")
print(pd.Series(simplified_labels).value_counts())


#%%
# Optional: Analyze key terms for each simplified category
from sklearn.feature_extraction.text import TfidfVectorizer

def analyze_category_terms(descriptions, labels, category):
    vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 2))
    mask = labels == category
    if mask.sum() > 0:
        tfidf = vectorizer.fit_transform(descriptions[mask])
        terms = pd.DataFrame(
            tfidf.mean(axis=0).A1,
            index=vectorizer.get_feature_names_out(),
            columns=['score']
        ).sort_values('score', ascending=False)
        print(f"\nTop terms for {category}:")
        print(terms.head(10))

# Analyze terms for each major category
for category in pd.Series(simplified_labels).unique():
    analyze_category_terms(dataset_raw['Room Description'], simplified_labels, category)






#%% ##### FUZZY MATCHING #####
def fuzzy_extract_attributes(descriptions, attribute_lists, threshold=80):
    """
    Extract attributes using fuzzy string matching with improved preprocessing
    """
    # Preprocess attribute lists
    processed_attributes = {}
    for attr_type, attr_list in attribute_lists.items():
        processed_attrs = []
        for attr in attr_list:
            if pd.isna(attr) or str(attr).strip() in ['', '-']:  # Skip empty or dash-only values
                continue
                
            # Handle both camelCase and regular strings
            words = re.findall('[A-Z][^A-Z]*', str(attr))
            if words:
                processed_attrs.append(' '.join(word.lower() for word in words))
            else:
                # Clean and add non-empty strings
                cleaned = str(attr).lower().strip()
                if cleaned and cleaned != '-':
                    processed_attrs.append(cleaned)
                    
        processed_attributes[attr_type] = list(set(processed_attrs))  # Remove duplicates
    
    results = []
    
    # Process each description
    for desc in tqdm(descriptions, desc="Processing descriptions"):
        if pd.isna(desc) or not str(desc).strip():  # Handle empty descriptions
            results.append({k: [] for k in attribute_lists.keys()})
            continue
            
        # Convert description to lowercase and clean
        desc_lower = str(desc).lower().strip()
        
        # Split description into words and word pairs, filtering out empty strings
        words = [w for w in desc_lower.split() if w and w != '-']
        word_pairs = [' '.join(pair) for pair in zip(words[:-1], words[1:])]
        
        # Initialize matches for this description
        matches = {attr_type: [] for attr_type in attribute_lists.keys()}
        
        # Check each attribute type
        for attr_type, attrs in processed_attributes.items():
            if not attrs:  # Skip if no valid attributes for this type
                continue
                
            # Try matching single words and word pairs
            for text in words + word_pairs:
                if not text.strip():  # Skip empty strings
                    continue
                    
                # Get closest matches above threshold
                closest = process.extractBests(
                    text, 
                    attrs,
                    scorer=fuzz.token_sort_ratio,
                    score_cutoff=threshold
                )
                
                # Add matches that meet threshold
                for match, score in closest:
                    if match not in matches[attr_type]:  # Avoid duplicates
                        matches[attr_type].append({
                            'matched_text': match,
                            'original_text': text,
                            'score': score
                        })
        
        results.append(matches)
    
    return pd.DataFrame(results)

#%%
# Test on a sample first
sample_size = 100
sample_texts = dataset_raw['Room Description'].sample(n=sample_size, random_state=42)

print("Running fuzzy matching analysis...")
fuzzy_results = fuzzy_extract_attributes(
    sample_texts,
    {
        'room_types': room_types_list,
        'bed_types': bed_types_list,
        'views': room_view_list,
        'amenities': room_amenities_list
    },
    threshold=80  # Adjust this threshold as needed
)

#%%
# Analyze the results
def analyze_fuzzy_results(df):
    for col in df.columns:
        print(f"\nAnalyzing {col}:")
        # Get all matches with their scores
        all_matches = [match for matches in df[col] if matches for match in matches]
        if all_matches:
            matches_df = pd.DataFrame(all_matches)
            print("\nTop matched terms:")
            print(matches_df['matched_text'].value_counts().head(10))
            print("\nAverage match scores:")
            print(matches_df.groupby('matched_text')['score'].mean().sort_values(ascending=False).head(10))
            print("\nSample of original texts that were matched:")
            print(matches_df.groupby('matched_text')['original_text'].agg(list).head(5))
        else:
            print("No matches found")

analyze_fuzzy_results(fuzzy_results)

#%%
print(fuzzy_results.head())

#%%
dataset_sample.to_excel('Data/sample_fuzzy_extraction_input.xlsx', index=False)
fuzzy_results.to_excel('Data/sample_fuzzy_extraction_output.xlsx', index=False)

# %%
full_fuzzy_results = fuzzy_extract_attributes(
    dataset_raw['Room Description'],
    {
        'room_types': room_types_list,
        'bed_types': bed_types_list,
        'views': room_view_list,
        'amenities': room_amenities_list
    }
)

#%%
'''
# Add results to original dataframe
dataset_new = dataset_raw.copy()
dataset_new['fuzzy_room_types'] = full_fuzzy_results['room_types']
dataset_new['fuzzy_bed_types'] = full_fuzzy_results['bed_types']
dataset_new['fuzzy_views'] = full_fuzzy_results['views']
dataset_new['fuzzy_amenities'] = full_fuzzy_results['amenities']

#%%
dataset_new.to_excel('Data/dataset_fuzzy_extraction_v1.xlsx', index=False)
'''

# %%
def evaluate_fuzzy_accuracy(fuzzy_results, full_data):
    """Evaluate fuzzy matching extraction accuracy against true labels"""
    # Ensure we're using the same number of samples
    if len(fuzzy_results) != len(full_data):
        print(f"Warning: Results ({len(fuzzy_results)}) and samples ({len(full_data)}) length mismatch")
        n_samples = min(len(fuzzy_results), len(full_data))
        fuzzy_results = fuzzy_results.iloc[:n_samples]
        full_data = full_data.iloc[:n_samples]
    
    correct = 0
    total = len(full_data)
    
    print("\nFuzzy Matching Evaluation:")
    print("==========================")
    
    for i, (_, row) in enumerate(full_data.iterrows()):
        true_label = str(row['Guest Room Info']).lower()
        # Extract room types from fuzzy matches
        predicted_types = [match['matched_text'].lower() for match in fuzzy_results.iloc[i]['room_types']]
        
        # Check for match
        match = any(true_label in pred or pred in true_label for pred in predicted_types)
        correct += match
        
        # Print first 5 examples
        if i < 5:
            print(f"\nDescription: {row['Room Description']}")
            print(f"True Label: {true_label}")
            print(f"Predicted Types: {predicted_types}")
            print(f"Match: {'✓' if match else '✗'}")
    
    accuracy = correct / total
    print(f"\nOverall Accuracy: {accuracy:.2%}")
    
    return accuracy

#%%
# Run fuzzy matching on full dataset and evaluate
fuzzy_results = fuzzy_extract_attributes(
    dataset_raw['Room Description'],
    {
        'room_types': room_types_list,
        'bed_types': bed_types_list,
        'views': room_view_list
    }
)
accuracy = evaluate_fuzzy_accuracy(fuzzy_results, dataset_raw)

# %%
