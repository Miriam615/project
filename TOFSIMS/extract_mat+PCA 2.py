# -*- coding: utf-8 -*-
"""
Created on Tue Apr  7 23:09:30 2026

@author: user
"""

import os
import glob
import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.transforms as transforms
from matplotlib.patches import Ellipse
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler



# 1. CONFIDENCE ELLIPSE (corrected rotation)
def confidence_ellipse(x, y, ax, n_std=2.447, facecolor='none', **kwargs):
    """
    Draw a confidence ellipse for 2D data (x, y).
    n_std=2.447 corresponds to ~95% confidence for 2D bivariate normal.
    Rotation is computed from the covariance matrix — NOT fixed at 45°.
    """
    if x.size != y.size:
        raise ValueError("x and y must be the same size")
    if x.size < 3:
        return None

    cov = np.cov(x, y)

    # Eigendecomposition gives the true principal axes of the ellipse
    eigenvalues, eigenvectors = np.linalg.eigh(cov)

    # Sort by descending eigenvalue (largest axis first)
    order = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]

    # Angle of the largest eigenvector with the x-axis
    angle_deg = np.degrees(np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0]))

    # Semi-axes lengths scaled by n_std
    width  = 2 * n_std * np.sqrt(eigenvalues[0])
    height = 2 * n_std * np.sqrt(eigenvalues[1])

    ellipse = Ellipse(
        xy=(np.mean(x), np.mean(y)),
        width=width,
        height=height,
        angle=angle_deg,
        facecolor=facecolor,
        **kwargs
    )
    return ax.add_patch(ellipse)



# 2. LOAD DATA
folder_path = "C:/Users/user/OneDrive - Chalmers/Desktop/yanyang/MVSA/imagepipeline/folded image"
os.chdir(folder_path)
mat_files = sorted(glob.glob('*.mat'))   # sorted for reproducibility

data_list  = []
file_sizes = []

print("Loading files...")
for file in mat_files:
    with h5py.File(file, 'r') as f:
        data_raw = np.transpose(f['DATA'][:])
    data_list.append(data_raw)
    file_sizes.append(data_raw.shape[0])
    print(f"  {file}  →  {data_raw.shape[0]} rows, {data_raw.shape[1]} cols")



# 3. COMBINED PCA
print("\nScaling and running PCA...")
combined_data = np.vstack(data_list)

scaler      = StandardScaler()
data_scaled = scaler.fit_transform(combined_data)

pca         = PCA(n_components=2)
pca_results = pca.fit_transform(data_scaled)

var_pc1 = pca.explained_variance_ratio_[0] * 100
var_pc2 = pca.explained_variance_ratio_[1] * 100
print(f"  PC1: {var_pc1:.1f}%   PC2: {var_pc2:.1f}%")



# 4. SLICE PCA RESULTS BACK TO EACH FILE
# Subgroup containers
subgroups = {
    '6ppd':  {'x': [], 'y': [], 'color': 'red'},
    '6ppdq': {'x': [], 'y': [], 'color': 'purple'},
    'no22':  {'x': [], 'y': [], 'color': 'brown'},
    'no23':  {'x': [], 'y': [], 'color': 'green'},
}

# Individual-file colours for the scatter
file_colors = plt.cm.tab10(np.linspace(0, 1, len(mat_files)))

fig, ax = plt.subplots(figsize=(12, 8))
scatter_handles = []   # for the first (scatter) legend

current_idx = 0
for i, file_name in enumerate(mat_files):
    size      = file_sizes[i]
    start_idx = current_idx
    end_idx   = current_idx + size
    current_idx += size

    pc1 = pca_results[start_idx:end_idx, 0]
    pc2 = pca_results[start_idx:end_idx, 1]

    clean_name = os.path.basename(file_name).replace('.mat', '')

    # --- scatter ---
    sc = ax.scatter(pc1, pc2, s=1, alpha=0.05,
                    color=file_colors[i], label=clean_name, marker='.')
    scatter_handles.append(sc)

    # --- route to subgroup (order matters: 6ppdq before 6ppd) ---
    if '6ppdq' in clean_name:
        subgroups['6ppdq']['x'].extend(pc1)
        subgroups['6ppdq']['y'].extend(pc2)
    elif '6ppd' in clean_name:
        subgroups['6ppd']['x'].extend(pc1)
        subgroups['6ppd']['y'].extend(pc2)
    elif 'no22' in clean_name:
        subgroups['no22']['x'].extend(pc1)
        subgroups['no22']['y'].extend(pc2)
    elif 'no23' in clean_name:
        subgroups['no23']['x'].extend(pc1)
        subgroups['no23']['y'].extend(pc2)
    else:
        print(f"  WARNING: '{clean_name}' did not match any subgroup — skipped from ellipses.")


# 5. DRAW ELLIPSES
ellipse_handles = []

for label, grp in subgroups.items():
    x_arr = np.array(grp['x'])
    y_arr = np.array(grp['y'])

    if len(x_arr) < 3:
        print(f"  WARNING: subgroup '{label}' has fewer than 3 points — ellipse skipped.")
        continue

    confidence_ellipse(
        x_arr, y_arr, ax,
        n_std=2.447,
        edgecolor=grp['color'],
        linestyle='--',
        linewidth=2.5,
        alpha=0.9,
    )

    # Dummy patch for the ellipse legend
    dummy = mpatches.Patch(
        edgecolor=grp['color'], facecolor='none',
        linestyle='--', linewidth=2.5,
        label=f'{label} (95% CI)'
    )
    ellipse_handles.append(dummy)



# 6. LEGENDS (two separate, both visible)
# Legend 1 — individual files (scatter), upper left
legend1 = ax.legend(
    handles=scatter_handles,
    labels=[os.path.basename(f).replace('.mat', '') for f in mat_files],
    loc='upper left',
    title='Replicates',
    fontsize=8,
    markerscale=8,        # make tiny dots visible in the legend
    framealpha=0.7,
)
# Fix alpha so legend markers are fully opaque
for handle in legend1.legend_handles:
    handle.set_alpha(1.0)

ax.add_artist(legend1)   # keep it when we add legend2

# Legend 2 — ellipses / subgroups, upper right
legend2 = ax.legend(
    handles=ellipse_handles,
    loc='upper right',
    title='Subgroups (95% CI)',
    fontsize=9,
    framealpha=0.7,
)


# 7. FORMATTING
ax.axhline(0, color='grey', linewidth=0.5, linestyle=':')
ax.axvline(0, color='grey', linewidth=0.5, linestyle=':')
ax.set_title('PCA Score Plot with 95% Confidence Ellipses', fontsize=13)
ax.set_xlabel(f'PC1 ({var_pc1:.1f}% explained variance)', fontsize=11)
ax.set_ylabel(f'PC2 ({var_pc2:.1f}% explained variance)', fontsize=11)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('pca_score_plot.png', dpi=150, bbox_inches='tight')
plt.show()
print("\nDone. Plot saved as pca_score_plot.png")
