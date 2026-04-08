import os
import glob
import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Ellipse
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


# ==========================================
# 0. SETTINGS  ← only section you need to edit
# ==========================================
folder_path    = "C:/Users/user/OneDrive - Chalmers/Desktop/yanyang/MVSA/imagepipeline/folded image"

# Spatial binning factor.
# 1  = no binning (original resolution)
# 2  = 2×2 pixel blocks averaged → image side /2,  pixels /4
# 4  = 4×4 pixel blocks averaged → image side /4,  pixels /16
# Must be an integer that divides evenly into your image side length.
BINNING_FACTOR = 3

# Image dimensions (pixels per side — assumed square).
# 128×128 = 16384 pixels, 256×256 = 65536 pixels, etc.
# Set to None to let the script infer the side length automatically.
IMAGE_SIZE = None   # e.g. 128, or None for auto-detect


# ==========================================
# 1. CONFIDENCE ELLIPSE (corrected rotation)
# ==========================================
def confidence_ellipse(x, y, ax, n_std=2.447, facecolor='none', **kwargs):
    """
    Draw a confidence ellipse for 2D data (x, y).
    n_std=2.447 corresponds to ~95% confidence for a 2D bivariate normal.
    Rotation is derived from eigendecomposition of the covariance matrix.
    """
    if x.size != y.size:
        raise ValueError("x and y must be the same size")
    if x.size < 3:
        return None

    cov = np.cov(x, y)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)

    # Sort descending so the first axis is always the longest
    order        = eigenvalues.argsort()[::-1]
    eigenvalues  = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]

    angle_deg = np.degrees(np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0]))
    width     = 2 * n_std * np.sqrt(eigenvalues[0])
    height    = 2 * n_std * np.sqrt(eigenvalues[1])

    ellipse = Ellipse(
        xy=(np.mean(x), np.mean(y)),
        width=width, height=height, angle=angle_deg,
        facecolor=facecolor, **kwargs
    )
    return ax.add_patch(ellipse)


# ==========================================
# 2. SPATIAL BINNING HELPER
# ==========================================
def bin_image(data_flat, image_side, binning_factor):
    """
    Spatially bin a flat pixel array by averaging non-overlapping blocks.

    Parameters
    ----------
    data_flat      : ndarray (n_pixels, n_masses)
                     Flat pixel-by-m/z matrix from one .mat file.
    image_side     : int
                     Side length of the square image (e.g. 128 for 128x128).
    binning_factor : int
                     Pixels to group per side (1 = no binning, 2 = 2x2, etc.)

    Returns
    -------
    binned : ndarray (n_pixels_binned, n_masses)
             Each row is the mean spectrum of one binned super-pixel.

    Steps
    -----
    1. Reshape flat (pixels, masses) -> (rows, cols, masses)      [3-D image cube]
    2. Reshape into blocks of size binning_factor
       -> (n_blocks_row, bf, n_blocks_col, bf, masses)
    3. Average over the two binning axes -> (n_blocks_row, n_blocks_col, masses)
    4. Flatten back to (n_blocks_row * n_blocks_col, masses)
    """
    if binning_factor == 1:
        return data_flat                          # nothing to do

    n_pixels, n_masses = data_flat.shape
    assert image_side ** 2 == n_pixels, (
        f"image_side={image_side} does not match n_pixels={n_pixels} "
        f"(expected {image_side}^2 = {image_side**2})"
    )
    assert image_side % binning_factor == 0, (
        f"image_side={image_side} is not evenly divisible "
        f"by binning_factor={binning_factor}"
    )

    n_blocks = image_side // binning_factor       # output blocks per side
    bf       = binning_factor

    # Step 1: unflatten to 3-D image cube (rows, cols, masses)
    cube = data_flat.reshape(image_side, image_side, n_masses)

    # Steps 2-3: split each axis into (n_blocks, bf) and average over bf
    binned = (cube
              .reshape(n_blocks, bf, n_blocks, bf, n_masses)
              .mean(axis=(1, 3)))                 # mean over the two bf axes

    # Step 4: flatten spatial dims back to a 1-D pixel list
    binned = binned.reshape(n_blocks * n_blocks, n_masses)

    return binned


# ==========================================
# 3. LOAD & BIN DATA
# ==========================================
os.chdir(folder_path)
mat_files = sorted(glob.glob('*.mat'))

data_list  = []
file_sizes = []   # binned pixel count per file — needed for PCA slicing later

print(f"Loading files  (binning factor = {BINNING_FACTOR})...")
for file in mat_files:
    with h5py.File(file, 'r') as f:
        data_raw = np.transpose(f['DATA'][:])   # shape: (n_pixels, n_masses)

    n_pixels, n_masses = data_raw.shape

    # Determine image side length
    if IMAGE_SIZE is not None:
        side = IMAGE_SIZE
    else:
        side = int(round(np.sqrt(n_pixels)))
        assert side ** 2 == n_pixels, (
            f"Cannot infer a square side from {n_pixels} pixels in '{file}'. "
            f"Set IMAGE_SIZE explicitly in the SETTINGS block."
        )

    # Apply spatial binning
    data_binned  = bin_image(data_raw, side, BINNING_FACTOR)
    binned_side  = side // BINNING_FACTOR

    print(f"  {file}")
    print(f"    Original : {side}x{side} = {n_pixels} pixels, {n_masses} m/z values")
    print(f"    Binned   : {binned_side}x{binned_side} = {data_binned.shape[0]} pixels")

    data_list.append(data_binned)
    file_sizes.append(data_binned.shape[0])


# ==========================================
# 4. COMBINED PCA ON BINNED DATA
# ==========================================
print("\nScaling and running PCA on binned data...")
combined_data = np.vstack(data_list)

scaler      = StandardScaler()
data_scaled = scaler.fit_transform(combined_data)

pca         = PCA(n_components=2)
pca_results = pca.fit_transform(data_scaled)

var_pc1 = pca.explained_variance_ratio_[0] * 100
var_pc2 = pca.explained_variance_ratio_[1] * 100
print(f"  PC1: {var_pc1:.1f}%   PC2: {var_pc2:.1f}%")


# ==========================================
# 5. SLICE PCA RESULTS BACK PER FILE & PLOT
# ==========================================
subgroups = {
    '6ppd':  {'x': [], 'y': [], 'color': 'red'},
    '6ppdq': {'x': [], 'y': [], 'color': 'purple'},
    'no22':  {'x': [], 'y': [], 'color': 'brown'},
    'no23':  {'x': [], 'y': [], 'color': 'green'},
}

file_colors     = plt.cm.tab10(np.linspace(0, 1, len(mat_files)))
fig, ax         = plt.subplots(figsize=(12, 8))
scatter_handles = []

current_idx = 0
for i, file_name in enumerate(mat_files):
    size      = file_sizes[i]           # binned pixel count for this file
    start_idx = current_idx
    end_idx   = current_idx + size
    current_idx += size

    pc1 = pca_results[start_idx:end_idx, 0]
    pc2 = pca_results[start_idx:end_idx, 1]

    clean_name = os.path.basename(file_name).replace('.mat', '')

    sc = ax.scatter(pc1, pc2, s=1, alpha=0.05,
                    color=file_colors[i], label=clean_name, marker='.')
    scatter_handles.append(sc)

    # Route to subgroup — 6ppdq must come before 6ppd to avoid partial match
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
        print(f"  WARNING: '{clean_name}' did not match any subgroup — excluded from ellipses.")


# ==========================================
# 6. CONFIDENCE ELLIPSES
# ==========================================
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
        linestyle='--', linewidth=2.5, alpha=0.9,
    )

    dummy = mpatches.Patch(
        edgecolor=grp['color'], facecolor='none',
        linestyle='--', linewidth=2.5,
        label=f'{label} (95% CI)'
    )
    ellipse_handles.append(dummy)


# ==========================================
# 7. LEGENDS (both kept)
# ==========================================
legend1 = ax.legend(
    handles=scatter_handles,
    labels=[os.path.basename(f).replace('.mat', '') for f in mat_files],
    loc='upper left',
    title='Replicates',
    fontsize=8,
    markerscale=8,
    framealpha=0.7,
)
for handle in legend1.legend_handles:
    handle.set_alpha(1.0)
ax.add_artist(legend1)          # preserve scatter legend when adding ellipse legend

ax.legend(
    handles=ellipse_handles,
    loc='upper right',
    title='Subgroups (95% CI)',
    fontsize=9,
    framealpha=0.7,
)


# ==========================================
# 8. FORMATTING & SAVE
# ==========================================
binning_label = (f"  [binning {BINNING_FACTOR}\u00d7{BINNING_FACTOR}]"
                 if BINNING_FACTOR > 1 else "")

ax.axhline(0, color='grey', linewidth=0.5, linestyle=':')
ax.axvline(0, color='grey', linewidth=0.5, linestyle=':')
ax.set_title(f'PCA Score Plot with 95% Confidence Ellipses{binning_label}', fontsize=13)
ax.set_xlabel(f'PC1 ({var_pc1:.1f}% explained variance)', fontsize=11)
ax.set_ylabel(f'PC2 ({var_pc2:.1f}% explained variance)', fontsize=11)
ax.grid(True, alpha=0.3)

plt.tight_layout()
out_name = f'pca_score_plot_bin{BINNING_FACTOR}.png'
plt.savefig(out_name, dpi=150, bbox_inches='tight')
plt.show()
print(f"\nDone. Plot saved as {out_name}")
