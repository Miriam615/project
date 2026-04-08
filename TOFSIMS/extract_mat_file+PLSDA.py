# -*- coding: utf-8 -*-
"""
Created on Wed Mar 18 13:21:56 2026

@author: user
"""

import os
import glob
import h5py
import gc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.transforms as transforms
from matplotlib.patches import Ellipse
from sklearn.cross_decomposition import PLSRegression

# ==========================================
# 1. SETUP & ELLIPSE FUNCTION
# ==========================================
def confidence_ellipse(x, y, ax, n_std=2.447, facecolor='none', **kwargs):
    cov = np.cov(x, y)
    pearson = cov[0, 1]/np.sqrt(cov[0, 0] * cov[1, 1])
    ell_radius_x, ell_radius_y = np.sqrt(1 + pearson), np.sqrt(1 - pearson)
    ellipse = Ellipse((0, 0), width=ell_radius_x * 2, height=ell_radius_y * 2, facecolor=facecolor, **kwargs)
    scale_x, scale_y = np.sqrt(cov[0, 0]) * n_std, np.sqrt(cov[1, 1]) * n_std
    transf = transforms.Affine2D().rotate_deg(45).scale(scale_x, scale_y).translate(np.mean(x), np.mean(y))
    ellipse.set_transform(transf + ax.transData)
    return ax.add_patch(ellipse)

folder_path = "C:/Users/user/OneDrive - Chalmers/Desktop/yanyang/MVSA/imagepipeline/folded image"
os.chdir(folder_path)
mat_files = glob.glob('*.mat')

manual_groups = {
    '6ppd-1.mat': '6ppd', '6ppd-2.mat': '6ppd',
    '6ppdq-1.mat': '6ppdq', '6ppdq-2.mat': '6ppdq',
    'no22-2.mat': 'no22', 'no22-3.mat': 'no22', 'no22-4.mat': 'no22',
    'no23-2.mat': 'no23', 'no23-3.mat': 'no23', 'no23-4.mat': 'no23'
}

# ==========================================
# 2. LOAD DATA & DYNAMIC SIZING
# ==========================================
data_list, y_labels, file_sizes = [], [], []

print("Loading files...")
for file in mat_files:
    fname = os.path.basename(file)
    if fname not in manual_groups: continue
    
    group = manual_groups[fname]
    with h5py.File(file, 'r') as f:
        data_raw = np.transpose(f['DATA'][:])
        data_list.append(data_raw)
        file_sizes.append(data_raw.shape[0])
        y_labels.append(np.repeat(group, data_raw.shape[0]))
        print(f"Loaded {fname} ({data_raw.shape[0]} rows) -> Group: {group}")

combined_X = np.vstack(data_list)
combined_Y = np.concatenate(y_labels)

# ==========================================
# 3. POISSON SCALING & PLS-DA
# ==========================================
print("Poisson Scaling...")
f_means = np.mean(combined_X, axis=0, dtype=np.float64)
X_scaled = combined_X.astype(np.float32)
del combined_X
gc.collect()

X_scaled -= f_means.astype(np.float32)
X_scaled /= np.sqrt(np.where(f_means == 0, 1e-10, f_means)).astype(np.float32)

print("Running PLS-DA...")
Y_dummy = pd.get_dummies(combined_Y).values
plsda = PLSRegression(n_components=2)
plsda.fit(X_scaled, Y_dummy)
scores = plsda.x_scores_


# ==========================================
# 4. PLOTTING & SLICING
# ==========================================
fig, ax = plt.subplots(figsize=(12, 8))
colors = plt.cm.tab10(np.linspace(0, 1, len(mat_files)))
sub_x = {g: [] for g in set(manual_groups.values())}
sub_y = {g: [] for g in set(manual_groups.values())}

color_dict = {'6ppd':'red',
              '6ppdq':'purple',
              'no22':'brown',
              'no23':'green'}

curr = 0
for i, file in enumerate(mat_files):
    fname = os.path.basename(file)
    size = file_sizes[i]
    p1, p2 = scores[curr:curr+size, 0], scores[curr:curr+size, 1]
    curr += size
    
    # 1. Look up which group this specific file belongs to
    # (Using the manual_groups dictionary from earlier in the script)
    group = manual_groups[fname]
    
    # 2. Grab the specific color for that group
    point_color = color_dict[group]
    
    # 3. Pass that single color to the scatter plot
    ax.scatter(p1, p2, s=1, alpha=0.03, color=point_color, label=fname.replace('.mat',''))
    sub_x[group].extend(p1)
    sub_y[group].extend(p2)

# Draw Ellipses
e_colors = {'6ppd': 'red', '6ppdq': 'purple', 'no22': 'brown', 'no23': 'green'}
e_items = []
for g in sub_x:
    xs, ys = np.array(sub_x[g]), np.array(sub_y[g])
    if len(xs) > 2:
        confidence_ellipse(xs, ys, ax, n_std=2.447, edgecolor=e_colors[g], linestyle='--', linewidth=1)
        e_items.append((Ellipse((0,0),1,1, edgecolor=color_dict[g], facecolor='none', linestyle='--', linewidth=1), g))

# Legends
ax.legend([it[0] for it in e_items], [it[1] for it in e_items], loc='upper right', title="Subgroups (95% CI)")

plt.title('PLS-DA Score Plot (Poisson Scaled)')
plt.xlabel('Component 1'); plt.ylabel('Component 2')
plt.grid(True, alpha=0.3); plt.tight_layout(); plt.show()