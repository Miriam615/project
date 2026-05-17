# -*- coding: utf-8 -*-
"""
Created on Tue Feb 17 09:48:27 2026

@author: user
"""


import os
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import zscore
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sys import exit
import matplotlib.lines as mlines
import numpy as np
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms



# dirs
cwd = os.getcwd()
print("Current working directory:", cwd)
os.chdir("C:/Users/user/OneDrive - Chalmers/Desktop/yanyang/peak list and statistics")
print(os.getcwd())


doc = "507.csv"


# Attempt to load both datasets !!!this is df
df = pd.read_csv(doc, sep=',', index_col=0)

print("Original Data Shape:", df.shape)

# ==========================================
# 2. DATA PREPROCESSING
# ==========================================

# Transpose the data: PCA needs Samples as Rows, M/Z as Columns
df_t = df.T 

# Create a grouping column based on the sample names (splitting at the hyphen)
# 'no22-2' -> 'no22'
groups = [name.split('-')[0] for name in df_t.index]
df_t['Group'] = groups

# Separate features (X) and target/group (y)
X = df_t.drop('Group', axis=1)
y = df_t['Group']

# Poisson scaling
X_poisson = X / np.sqrt(X.mean(axis=0))

# Mean centering (like prcomp default)
X_scaled = X_poisson - X_poisson.mean(axis=0)


# ==========================================
# 3. COMPUTE PCA
# ==========================================
pca = PCA(n_components=None)
principalComponents = pca.fit_transform(X_scaled)

# Create a DataFrame for the PCA results
pca_df = pd.DataFrame(data=principalComponents[:, 0:2], columns=['PC1', 'PC2'])
pca_df['Group'] = y.values
pca_df.index = df_t.index # Keep original sample names

# Calculate explained variance
expl_var = pca.explained_variance_ratio_
pc1_var = round(expl_var[0] * 100, 2)
pc2_var = round(expl_var[1] * 100, 2)


# ==========================================
# 4. ELLIPSE FUNCTION
# ==========================================
def confidence_ellipse(x, y, ax, n_std=2.0, facecolor='none', **kwargs):
    """
    Create a plot of the covariance confidence ellipse of *x* and *y*.
    n_std: number of standard deviations (2.0 approx 95% confidence)
    """
    if x.size != y.size:
        raise ValueError("x and y must be the same size")

    cov = np.cov(x, y)
    pearson = cov[0, 1]/np.sqrt(cov[0, 0] * cov[1, 1])
    
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    ellipse = Ellipse((0, 0), width=ell_radius_x * 2, height=ell_radius_y * 2,
                      facecolor=facecolor, **kwargs)
    scale_x = np.sqrt(cov[0, 0]) * n_std
    mean_x = np.mean(x)
    scale_y = np.sqrt(cov[1, 1]) * n_std
    mean_y = np.mean(y)
    transf = transforms.Affine2D() \
        .rotate_deg(45) \
        .scale(scale_x, scale_y) \
        .translate(mean_x, mean_y)
    ellipse.set_transform(transf + ax.transData)
    return ax.add_patch(ellipse)


# ==========================================
# 5. PLOTTING
# ==========================================
fig, ax = plt.subplots(figsize=(12, 8))

# Define colors for your groups
colors = {'no22': 'tomato', 'no23': 'cornflowerblue'}

# Iterate through groups to plot points and ellipses
unique_groups = pca_df['Group'].unique()

for group in unique_groups:
    subset = pca_df[pca_df['Group'] == group]
    
    # Plot Scatter points
    ax.scatter(subset['PC1'], subset['PC2'], c=colors[group], label=group, s=150, alpha=0.8)
    
    # Add labels to points (Sample names like no22-2)
    for i, txt in enumerate(subset.index):
        ax.annotate(txt, (subset['PC1'].iloc[i], subset['PC2'].iloc[i]), 
                    xytext=(5,5), textcoords='offset points', fontsize=12)
    
    # Draw Confidence Ellipse (needs > 2 points to calculate variance effectively)
    if len(subset) > 1: 
        confidence_ellipse(subset['PC1'], subset['PC2'], ax, n_std=2.0, 
                           edgecolor=colors[group], linestyle='--', linewidth=2)

# Formatting
ax.set_xlabel(f'Principal Component 1 ({pc1_var}%)', fontsize=12)
ax.set_ylabel(f'Principal Component 2 ({pc2_var}%)', fontsize=12)
ax.set_title('PCA scores plot with 95% Confidence Ellipses', fontsize=20)
ax.legend(fontsize=15,loc=1)
ax.grid(True, linestyle='--', alpha=0.5)

plt.tight_layout()
plt.show()



#scree plot
plt.figure(figsize=(8, 6)) # Create a separate figure for Scree Plot

# Get the number of components (likely 6)
num_components = len(pca.explained_variance_ratio_)
ind = np.arange(num_components) + 1 

# Plot the Line + Dots
plt.plot(ind, pca.explained_variance_ratio_ * 100, 'o-', linewidth=2, color='darkblue', markersize=8)

# Plot Bars (Optional, helps visualize drop-off)
plt.bar(ind, pca.explained_variance_ratio_ * 100, alpha=0.3, color='blue')

# Formatting
plt.xlabel('Principal Component', fontsize=12)
plt.ylabel('Variance Explained (%)', fontsize=12)
plt.title('Scree Plot', fontsize=15)
plt.xticks(ind) # Force x-axis to show integers (1, 2, 3...)
plt.grid(True, linestyle='--', alpha=0.5)

# Add text labels above each point
for i, v in enumerate(pca.explained_variance_ratio_):
    plt.text(i + 1, v * 100 + 1, f'{v*100:.1f}%', ha='center', fontsize=10)

plt.tight_layout()
plt.show()




