import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import NMF, LatentDirichletAllocation
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
from gensim.models.coherencemodel import CoherenceModel
from gensim.corpora.dictionary import Dictionary
import plotly.express as px
import plotly.graph_objects as go
from sklearn.manifold import TSNE
import re
from sklearn.metrics import v_measure_score, adjusted_rand_score
from scipy.optimize import linear_sum_assignment
from collections import defaultdict

def preprocess_for_topic_modeling(text):
    """Clean and preprocess text for topic modeling"""
    if pd.isna(text):
        return ""
    
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
    
    # Remove very common words that might not be in standard stop words
    custom_stops = {'a', 'an', 'the', 'in', 'on', 'at', 'with', 'and', 'or', 'to', 'for'}
    words = [w for w in words if w not in custom_stops]
    
    return ' '.join(words)

def run_tfidf_topic_modeling(texts, n_topics=35, top_n_words=10, ground_truth_labels=None):
    """
    Enhanced TF-IDF topic modeling with NMF and LDA, including evaluation metrics
    """
    # Create TF-IDF matrix with original parameters - using both unigrams and bigrams
    tfidf_vectorizer = TfidfVectorizer(
        max_df=0.95,         # Ignore terms that appear in >95% of docs
        min_df=2,           # Ignore terms that appear in <2 docs
        stop_words='english',
        ngram_range=(2, 2),  # Use only bigrams
        max_features=10000   # Increased vocabulary size
    )
    tfidf_matrix = tfidf_vectorizer.fit_transform(texts)
    
    # 1. NMF Model with original parameters
    print("Running NMF with TF-IDF...")
    nmf_model = NMF(
        n_components=n_topics,
        random_state=42,
        init='nndsvd',  # Back to original initialization
        max_iter=200,   # Original iterations
        alpha=0.1,      # Original regularization
        solver='cd'     # Original solver
    )
    nmf_output = nmf_model.fit_transform(tfidf_matrix)
    
    # Check topic distribution
    nmf_dominant_topics = np.argmax(nmf_output, axis=1)
    topic_counts = pd.Series(nmf_dominant_topics).value_counts()
    print("\nNMF Topic Distribution:")
    print(topic_counts)
    
    # 2. LDA Model with original parameters
    print("\nRunning LDA with TF-IDF...")
    lda_model = LatentDirichletAllocation(
        n_components=n_topics,
        random_state=42,
        max_iter=10,
        learning_method='batch',
        batch_size=128,
        doc_topic_prior=0.1,
        topic_word_prior=0.01
    )
    lda_output = lda_model.fit_transform(tfidf_matrix)
    
    # Get feature names
    feature_names = tfidf_vectorizer.get_feature_names_out()
    
    # Calculate evaluation metrics
    # For NMF
    nmf_kmeans = KMeans(n_clusters=n_topics, random_state=42)
    nmf_clusters = nmf_kmeans.fit_predict(nmf_output)
    nmf_silhouette = silhouette_score(nmf_output, nmf_clusters)
    
    # For LDA
    lda_kmeans = KMeans(n_clusters=n_topics, random_state=42)
    lda_clusters = lda_kmeans.fit_predict(lda_output)
    lda_silhouette = silhouette_score(lda_output, lda_clusters)
    
    # Calculate metrics against ground truth if provided
    if ground_truth_labels is not None:
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        true_labels_encoded = le.fit_transform(ground_truth_labels)
        
        # Calculate metrics against ground truth for NMF
        nmf_purity = calculate_purity(true_labels_encoded, nmf_clusters)
        nmf_v_measure = v_measure_score(true_labels_encoded, nmf_clusters)
        nmf_ari = adjusted_rand_score(true_labels_encoded, nmf_clusters)
        
        # Calculate metrics against ground truth for LDA
        lda_purity = calculate_purity(true_labels_encoded, lda_clusters)
        lda_v_measure = v_measure_score(true_labels_encoded, lda_clusters)
        lda_ari = adjusted_rand_score(true_labels_encoded, lda_clusters)
        
        print("\nEvaluation Metrics Against Ground Truth:")
        print(f"NMF Purity: {nmf_purity:.4f}")
        print(f"NMF V-Measure: {nmf_v_measure:.4f}")
        print(f"NMF Adjusted Rand Index: {nmf_ari:.4f}")
        print(f"LDA Purity: {lda_purity:.4f}")
        print(f"LDA V-Measure: {lda_v_measure:.4f}")
        print(f"LDA Adjusted Rand Index: {lda_ari:.4f}")
    
    # Print silhouette scores
    print(f"NMF Silhouette Score: {nmf_silhouette:.4f}")
    print(f"LDA Silhouette Score: {lda_silhouette:.4f}")
    
    # Get topic coherence scores
    # Prepare texts for coherence calculation
    texts_for_coherence = [text.split() for text in texts]
    dictionary = Dictionary(texts_for_coherence)
    corpus = [dictionary.doc2bow(text) for text in texts_for_coherence]
    
    # Calculate coherence for NMF
    nmf_topics = []
    for topic in nmf_model.components_:
        top_words_idx = topic.argsort()[:-top_n_words-1:-1]
        top_words = [feature_names[i] for i in top_words_idx]
        nmf_topics.append(top_words)
    
    # Skip coherence calculation for bigrams as it's not compatible
    print("Note: Coherence calculation skipped for bigram topics as it's not compatible with the current implementation.")
    nmf_coherence = 0.0
    
    # Calculate coherence for LDA
    lda_topics = []
    for topic in lda_model.components_:
        top_words_idx = topic.argsort()[:-top_n_words-1:-1]
        top_words = [feature_names[i] for i in top_words_idx]
        lda_topics.append(top_words)
    
    # Skip coherence calculation for bigrams as it's not compatible
    lda_coherence = 0.0
    
    print(f"NMF Coherence Score: {nmf_coherence:.4f}")
    print(f"LDA Coherence Score: {lda_coherence:.4f}")
    
    # Print top words for each topic
    print("\nNMF Topics:")
    for topic_idx, topic in enumerate(nmf_topics):
        print(f"Topic {topic_idx + 1}: {', '.join(topic)}")
    
    print("\nLDA Topics:")
    for topic_idx, topic in enumerate(lda_topics):
        print(f"Topic {topic_idx + 1}: {', '.join(topic)}")
    
    # Create result dictionary
    result = {
        'features': feature_names,
        'vectorizer': tfidf_vectorizer,
        'nmf_model': nmf_model,
        'nmf_output': nmf_output,
        'nmf_topics': nmf_topics,
        'nmf_metrics': {
            'silhouette': nmf_silhouette,
            'coherence': nmf_coherence
        },
        'lda_model': lda_model,
        'lda_output': lda_output,
        'lda_topics': lda_topics,
        'lda_metrics': {
            'silhouette': lda_silhouette,
            'coherence': lda_coherence
        }
    }
    
    # Add ground truth metrics if available
    if ground_truth_labels is not None:
        result['nmf_metrics'].update({
            'purity': nmf_purity,
            'v_measure': nmf_v_measure,
            'adjusted_rand_index': nmf_ari
        })
        result['lda_metrics'].update({
            'purity': lda_purity,
            'v_measure': lda_v_measure,
            'adjusted_rand_index': lda_ari
        })
    
    return result

def visualize_tfidf_results(results, dataset_raw):
    """
    Visualize TF-IDF topic modeling results
    """
    # 1. Topic Distribution
    nmf_dominant_topics = np.argmax(results['nmf_output'], axis=1)
    lda_dominant_topics = np.argmax(results['lda_output'], axis=1)
    
    # Create topic name mappings using top words
    def create_topic_name(topic_words, max_words=5):
        # Take top max_words
        return ' '.join(topic_words[:max_words])
    
    # Create topic names
    nmf_topic_names = {i: create_topic_name(topic) 
                      for i, topic in enumerate(results['nmf_topics'])}
    lda_topic_names = {i: create_topic_name(topic) 
                      for i, topic in enumerate(results['lda_topics'])}
    
    # Add topic numbers and names to dataset
    dataset_raw['NMF_Topic'] = nmf_dominant_topics
    dataset_raw['NMF_Topic_Name'] = dataset_raw['NMF_Topic'].map(nmf_topic_names)
    dataset_raw['TFIDF_LDA_Topic'] = lda_dominant_topics
    dataset_raw['TFIDF_LDA_Topic_Name'] = dataset_raw['TFIDF_LDA_Topic'].map(lda_topic_names)
    
    # Print topic names for verification
    print("\nNMF Topic Names:")
    for topic_idx, topic_name in sorted(nmf_topic_names.items()):
        print(f"Topic {topic_idx}: {topic_name}")
    
    print("\nLDA Topic Names:")
    for topic_idx, topic_name in sorted(lda_topic_names.items()):
        print(f"Topic {topic_idx}: {topic_name}")
    
    # Plot distributions
    fig_nmf = px.bar(
        x=pd.Series(nmf_dominant_topics).value_counts().index + 1,
        y=pd.Series(nmf_dominant_topics).value_counts().values,
        labels={'x': 'Topic Number', 'y': 'Number of Documents'},
        title='Distribution of NMF Topics'
    )
    fig_nmf.show()
    
    fig_lda = px.bar(
        x=pd.Series(lda_dominant_topics).value_counts().index + 1,
        y=pd.Series(lda_dominant_topics).value_counts().values,
        labels={'x': 'Topic Number', 'y': 'Number of Documents'},
        title='Distribution of LDA Topics'
    )
    fig_lda.show()
    
    # 2. Topic Similarity Heatmap
    def create_topic_similarity_heatmap(topic_matrix, title):
        similarity_matrix = np.zeros((topic_matrix.shape[1], topic_matrix.shape[1]))
        for i in range(topic_matrix.shape[1]):
            for j in range(topic_matrix.shape[1]):
                similarity_matrix[i,j] = np.dot(topic_matrix[:,i], topic_matrix[:,j])
        
        fig = go.Figure(data=go.Heatmap(
            z=similarity_matrix,
            x=list(range(1, topic_matrix.shape[1] + 1)),
            y=list(range(1, topic_matrix.shape[1] + 1)),
            colorscale='Viridis'
        ))
        fig.update_layout(title=title)
        fig.show()
    
    create_topic_similarity_heatmap(results['nmf_output'], 'NMF Topic Similarity')
    create_topic_similarity_heatmap(results['lda_output'], 'LDA Topic Similarity')
    
    # 3. t-SNE visualization
    def plot_tsne(topic_matrix, dominant_topics, title):
        tsne = TSNE(n_components=2, random_state=42)
        tsne_result = tsne.fit_transform(topic_matrix)
        
        fig = px.scatter(
            x=tsne_result[:, 0],
            y=tsne_result[:, 1],
            color=dominant_topics,
            title=title,
            labels={'x': 't-SNE 1', 'y': 't-SNE 2', 'color': 'Topic'}
        )
        fig.show()
    
    plot_tsne(results['nmf_output'], nmf_dominant_topics, 'NMF Topics t-SNE')
    plot_tsne(results['lda_output'], lda_dominant_topics, 'LDA Topics t-SNE')
    
    return dataset_raw

def calculate_clustering_accuracy(embeddings, true_labels, n_clusters=35):
    """
    Calculate clustering accuracy by finding the best mapping between clusters and labels.
    
    Args:
        embeddings: The embeddings from MiniLM
        true_labels: Ground truth labels (Guest Room Info)
        n_clusters: Number of clusters to use
        
    Returns:
        Dictionary containing accuracy metrics
    """
    # Perform K-means clustering on embeddings
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    predicted_labels = kmeans.fit_predict(embeddings)
    
    # Create a contingency matrix
    contingency_matrix = np.zeros((n_clusters, len(np.unique(true_labels))))
    for i in range(len(true_labels)):
        contingency_matrix[predicted_labels[i], true_labels[i]] += 1
    
    # Find the best mapping between clusters and true labels
    row_ind, col_ind = linear_sum_assignment(-contingency_matrix)
    
    # Calculate accuracy
    correct = 0
    for i in range(len(predicted_labels)):
        if col_ind[predicted_labels[i]] == true_labels[i]:
            correct += 1
    accuracy = correct / len(predicted_labels)
    
    return {
        'accuracy': accuracy,
        'mapping': dict(zip(row_ind, col_ind))  # Shows which cluster maps to which true label
    }

def calculate_purity(true_labels, predicted_clusters):
    """
    Calculate the purity score for clustering.
    
    Args:
        true_labels: Ground truth labels
        predicted_clusters: Predicted cluster assignments
        
    Returns:
        Purity score between 0 and 1
    """
    contingency = defaultdict(lambda: defaultdict(int))
    for true, pred in zip(true_labels, predicted_clusters):
        contingency[pred][true] += 1
    
    total_samples = len(true_labels)
    purity = sum(max(cluster.values()) for cluster in contingency.values()) / total_samples
    return purity

# Example usage:
if __name__ == "__main__":
    # Load your dataset
    dataset_raw = pd.read_csv('path_to_your_data.csv')
    
    # Preprocess texts
    processed_texts = dataset_raw['Room Description'].apply(preprocess_for_topic_modeling)
    
    # Run topic modeling
    results = run_tfidf_topic_modeling(processed_texts, n_topics=35)
    
    # Visualize results
    dataset_raw = visualize_tfidf_results(results, dataset_raw) 