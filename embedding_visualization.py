import torch
import torch.nn as nn
from transformers import AutoModelForTokenClassification
import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE, MDS
from sklearn.decomposition import PCA
from umap import UMAP
import seaborn as sns
import pandas as pd
from tqdm import tqdm
import auxiliary as aux
from kingbert import KingBert
import os

class EmbeddingExtractor(nn.Module):
    """Wrapper for KingBert model to extract embeddings before final layer"""
    def __init__(self, kingbert_model):
        super().__init__()
        self.model = kingbert_model
        
    def forward(self, distilbert_input_ids, albert_input_ids, distil_attention_mask, alb_attention_mask, distilbert_word_ids, albert_word_ids):
        # Get distilbert and albert outputs
        distilbert_output = self.model.distilbert(input_ids=distilbert_input_ids, attention_mask=distil_attention_mask)
        albert_output = self.model.albert(input_ids=albert_input_ids, attention_mask=alb_attention_mask)
        
        # Process through ensembler
        distilbert_fixed, albert_fixed = aux.ensembler(
            distilbert_output['logits'].squeeze(), 
            albert_output['logits'].squeeze(), 
            distilbert_word_ids.squeeze(), 
            albert_word_ids.squeeze()
        )
        
        # Get raw embeddings before applying softmax and weighting
        # This gives us the feature representation before the final classification layer
        return {
            'distilbert_embeddings': distilbert_fixed,
            'albert_embeddings': albert_fixed,
            'combined_raw': distilbert_fixed * self.model.alpha + albert_fixed * (torch.ones(47) - self.model.alpha)
        }

def extract_embeddings_from_dataset(model, dataset, num_samples=None):
    """Extract embeddings from the dataset"""
    if num_samples is None:
        num_samples = len(dataset)
    else:
        num_samples = min(num_samples, len(dataset))
    
    embeddings = []
    labels = []
    
    for i in tqdm(range(num_samples), desc="Extracting embeddings"):
        try:
            item = dataset[i]
            
            # Prepare inputs
            distilbert_input_ids = torch.tensor(item['distilbert_inputids']).unsqueeze(0)
            albert_input_ids = torch.tensor(item['albert_inputids']).unsqueeze(0)
            distil_attention_mask = torch.tensor(item['distilbert_attention_masks']).unsqueeze(0)
            alb_attention_mask = torch.tensor(item['albert_attention_masks']).unsqueeze(0)
            distilbert_word_ids = torch.tensor([-100] + item['distilbert_wordids'][1:-1] + [-100]).unsqueeze(0)
            albert_word_ids = torch.tensor([-100] + item['albert_wordids'][1:-1] + [-100]).unsqueeze(0)
            target = item['spacy_labels']
            
            # Extract embeddings
            with torch.no_grad():
                output = model(
                    distilbert_input_ids, 
                    albert_input_ids, 
                    distil_attention_mask, 
                    alb_attention_mask, 
                    distilbert_word_ids, 
                    albert_word_ids
                )
            
            # Get combined embeddings
            emb = output['combined_raw'].detach().cpu().numpy()
            
            # Store embeddings and labels
            for j, e in enumerate(emb):
                embeddings.append(e)
                labels.append(aux.id2label[target[j]])
                
        except Exception as ex:
            print(f"Error processing item {i}: {ex}")
            continue
    
    return np.array(embeddings), labels

def reduce_dimensions(embeddings, method='tsne', n_components=2, **kwargs):
    """Reduce dimensions of embeddings for visualization"""
    if method.lower() == 'tsne':
        reducer = TSNE(n_components=n_components, **kwargs)
    elif method.lower() == 'pca':
        reducer = PCA(n_components=n_components, **kwargs)
    elif method.lower() == 'umap':
        reducer = UMAP(n_components=n_components, **kwargs)
    elif method.lower() == 'mds':
        reducer = MDS(n_components=n_components, **kwargs)
    else:
        raise ValueError(f"Unknown dimensionality reduction method: {method}")
    
    return reducer.fit_transform(embeddings)

def plot_embeddings(reduced_embeddings, labels, method_name, save_path=None):
    """Plot the reduced embeddings with labels"""
    # Convert to dataframe for easier plotting
    df = pd.DataFrame({
        'x': reduced_embeddings[:, 0],
        'y': reduced_embeddings[:, 1],
        'label': labels
    })
    
    # Get unique labels and assign colors
    unique_labels = df['label'].unique()
    
    # Filter out 'O' label if present, as it's typically not interesting
    if 'O' in unique_labels:
        df_filtered = df[df['label'] != 'O']
    else:
        df_filtered = df
    
    # Create a large figure
    plt.figure(figsize=(14, 10))
    
    # Create scatter plot with seaborn for better aesthetics
    sns.scatterplot(
        data=df_filtered, 
        x='x', 
        y='y', 
        hue='label', 
        palette='viridis',
        alpha=0.7
    )
    
    # Add labels and title
    plt.title(f'Embedding Space Visualization using {method_name}', fontsize=16)
    plt.xlabel('Dimension 1', fontsize=14)
    plt.ylabel('Dimension 2', fontsize=14)
    
    # Adjust legend
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    plt.tight_layout()
    
    # Save figure if path is provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()

def analyze_label_clusters(reduced_embeddings, labels):
    """Analyze the distribution of labels in the embedding space"""
    # Create dataframe
    df = pd.DataFrame({
        'x': reduced_embeddings[:, 0],
        'y': reduced_embeddings[:, 1],
        'label': labels
    })
    
    # Calculate cluster centers
    cluster_centers = df.groupby('label')[['x', 'y']].mean().reset_index()
    
    # Calculate distances between cluster centers
    n_clusters = len(cluster_centers)
    dist_matrix = np.zeros((n_clusters, n_clusters))
    
    for i in range(n_clusters):
        for j in range(n_clusters):
            if i != j:
                center_i = cluster_centers.iloc[i][['x', 'y']].values
                center_j = cluster_centers.iloc[j][['x', 'y']].values
                dist_matrix[i, j] = np.linalg.norm(center_i - center_j)
    
    # Find closest and furthest labels for each label
    closest_pairs = []
    furthest_pairs = []
    
    for i in range(n_clusters):
        label_i = cluster_centers.iloc[i]['label']
        
        # Find closest label (excluding self)
        other_indices = np.arange(n_clusters) != i
        closest_idx = np.argmin(dist_matrix[i, other_indices])
        # Need to adjust index if it's after i
        if closest_idx >= i:
            closest_idx += 1
        closest_label = cluster_centers.iloc[closest_idx]['label']
        
        # Find furthest label
        furthest_idx = np.argmax(dist_matrix[i])
        furthest_label = cluster_centers.iloc[furthest_idx]['label']
        
        closest_pairs.append((label_i, closest_label, dist_matrix[i, closest_idx]))
        furthest_pairs.append((label_i, furthest_label, dist_matrix[i, furthest_idx]))
    
    return closest_pairs, furthest_pairs

def main():
    # Load the dataset
    try:
        dataset = aux.json_to_Dataset_ensemble('data/ensemble_train.json')
        print(f"Loaded dataset with {len(dataset)} samples")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return
    
    # Load the pre-trained models
    try:
        distilbert_tuned = AutoModelForTokenClassification.from_pretrained('distilbert_finetuned')
        albert_tuned = AutoModelForTokenClassification.from_pretrained('albert_finetuned')
        print("Loaded pre-trained models")
    except Exception as e:
        print(f"Error loading pre-trained models: {e}")
        return
    
    # Create KingBert model
    kingbert_model = KingBert(distilbert_tuned, albert_tuned)
    
    # Load saved state if available
    if os.path.exists('model_state.pth'):
        kingbert_model.load_state_dict(torch.load('model_state.pth'))
        print("Loaded saved model state")
    
    # Create embedding extractor
    embedding_extractor = EmbeddingExtractor(kingbert_model)
    
    # Extract embeddings from a subset of the dataset
    embeddings, labels = extract_embeddings_from_dataset(embedding_extractor, dataset, num_samples=500)
    print(f"Extracted {len(embeddings)} embeddings with shape {embeddings.shape}")
    
    # Create output directory for plots
    os.makedirs('embedding_plots', exist_ok=True)
    
    # Apply different dimensionality reduction techniques and visualize
    reduction_methods = {
        'PCA': {'perplexity': 30, 'random_state': 42},
        'TSNE': {'perplexity': 30, 'random_state': 42, 'learning_rate': 200},
        'UMAP': {'n_neighbors': 15, 'min_dist': 0.1, 'random_state': 42},
        'MDS': {'random_state': 42, 'n_jobs': -1}
    }
    
    for method, kwargs in reduction_methods.items():
        print(f"Applying {method} dimensionality reduction...")
        reduced_embeddings = reduce_dimensions(embeddings, method=method, **kwargs)
        
        # Plot and save embeddings
        save_path = f"embedding_plots/{method.lower()}_embeddings.png"
        plot_embeddings(reduced_embeddings, labels, method, save_path)
        
        # Analyze label clusters
        closest_pairs, furthest_pairs = analyze_label_clusters(reduced_embeddings, labels)
        
        print(f"\n{method} Analysis Results:")
        print("Closest label pairs:")
        for label1, label2, dist in closest_pairs[:10]:  # Show top 10
            print(f"  {label1} and {label2}: {dist:.4f}")
        
        print("\nFurthest label pairs:")
        for label1, label2, dist in furthest_pairs[:10]:  # Show top 10
            print(f"  {label1} and {label2}: {dist:.4f}")

if __name__ == "__main__":
    main() 