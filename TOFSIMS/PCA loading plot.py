# -*- coding: utf-8 -*-
"""
Created on Thu Feb 19 10:07:53 2026

@author: user
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# ==========================================
# 1. LOAD DATA
# ==========================================
PROJ = "MVSA"
doc = "140-420mc55.csv"

# Load data (m/z values are in the first column)
df = pd.read_csv(doc, sep=';', index_col=0)
print(f"Data Loaded. Shape: {df.shape}")

# ==========================================
# 2. PREPROCESSING
# ==========================================
# Transpose so samples are rows and m/z features are columns
df_t = df.T 

X = df_t 

# Poisson scaling
X_poisson = X / np.sqrt(X.mean(axis=0))

# Mean centering (like prcomp default)
X_scaled = X_poisson - X_poisson.mean(axis=0)

# ==========================================
# 3. COMPUTE PCA & EXTRACT LOADINGS
# ==========================================
pca = PCA(n_components=2)
pca.fit(X_scaled)

# Extract explained variance
expl_var = pca.explained_variance_ratio_
pc1_var = round(expl_var[0] * 100, 2)
pc2_var = round(expl_var[1] * 100, 2)

# Extract Loadings 
# pca.components_ has the shape (n_components, n_features). We transpose it.
loadings = pca.components_.T

# Create a DataFrame for the Loadings
# X.columns contains all your original m/z row names
loadings_df = pd.DataFrame(data=loadings, columns=['PC1', 'PC2'], index=X.columns)

# ==========================================
# 4. PLOTTING THE LOADINGS
# ==========================================
fig, ax = plt.subplots(figsize=(12, 9))

# Plot all m/z features as standard teal dots
ax.scatter(loadings_df['PC1'], loadings_df['PC2'], alpha=0.4, color='teal', s=30, label='m/z features')

# --- SMART LABELING ---
# Calculate distance from origin (0,0) to find the most important features
loadings_df['distance'] = np.sqrt(loadings_df['PC1']**2 + loadings_df['PC2']**2)

# Get the top 15 features furthest from the center
top_features = loadings_df.sort_values(by='distance', ascending=False).head(15)

# Highlight and label only the top features
for i, txt in enumerate(top_features.index):
    # Highlight the dot in red
    ax.scatter(top_features['PC1'].iloc[i], top_features['PC2'].iloc[i], color='red', s=50)
    # Add the text label
    ax.annotate(txt, (top_features['PC1'].iloc[i], top_features['PC2'].iloc[i]), 
                xytext=(5,5), textcoords='offset points', fontsize=10, color='red', fontweight='bold')

# Formatting
ax.axhline(0, color='black', linewidth=1, linestyle='--') # X-axis at 0
ax.axvline(0, color='black', linewidth=1, linestyle='--') # Y-axis at 0

ax.set_xlabel(f'Loading on PC1 ({pc1_var}%)', fontsize=12)
ax.set_ylabel(f'Loading on PC2 ({pc2_var}%)', fontsize=12)
ax.set_title('PCA Loadings Plot (Top 15 Most Influential m/z Peaks Labeled)', fontsize=15)
ax.grid(True, linestyle='--', alpha=0.5)

plt.tight_layout()
plt.show()

# Print the top features to the console for your records
print("\n--- TOP 15 INFLUENTIAL m/z PEAKS ---")
print(top_features[['PC1', 'PC2']])
