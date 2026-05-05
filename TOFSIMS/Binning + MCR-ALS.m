%% =========================================================
clear; clc; close all;

%% =========================================================

% --- File ---
DATA_FILE     = 'no22_wide_excluded_test-v73.mat';

% --- Image dimensions
IMG_ROWS      = 512;               % spatial rows  (128 × 128 = 16 384)
IMG_COLS      = 512;               % spatial cols

% --- Binning ---
BIN_FACTOR    = 6;                 % 1 = no binning, 2 = 2×2, 4 = 4×4 …

% --- Mass range filter (applied before PCA and MCR-ALS) ---
MZ_MIN        = [];    % — leave [] to skip lower bound
MZ_MAX        = [];    

% Excluded masses from contaminants
MZ_EXCLUDE    = [];
MZ_EXCL_TOL   = 0.1;  %set error boundary in case can't find the masses

% --- PCA component range to evaluate ---
K_MIN         = 2;
K_MAX         = 10;

% --- MCR-ALS ---
N_COMP        = 4;                 % set after inspecting scree plot
MAX_ITER      = 100;
CONV_CRIT     = 1e-6;             % relative change in lack-of-fit
APPLY_CLOSURE = false;          

%% =========================================================
%  SECTION 2 — LOAD DATA
%% =========================================================
fprintf('=== Loading data ===\n');

% --- Load DATA matrix and VARID mass list ---
if exist(DATA_FILE, 'file')
    loaded = load(DATA_FILE);

    % Data matrix: pixels × m/z
    assert(isfield(loaded,'DATA'), ...
        'Variable ''DATA'' not found in %s', DATA_FILE);
    D = double(loaded.DATA);
    fprintf('Loaded DATA: [%d × %d]\n', size(D,1), size(D,2));

    %mass list
    assert(isfield(loaded,'VARID6'), ...
        'Variable ''VARID2'' not found in %s', DATA_FILE);
    mz_labels = double(loaded.VARID2(:));   % ensure column vector
    fprintf('Loaded VARID2: %d masses  (%.2f – %.2f Da)\n', ...
            numel(mz_labels), mz_labels(1), mz_labels(end));

    assert(numel(mz_labels) == size(D,2), ...
        'VARID2 length (%d) must equal number of columns in DATA (%d).', ...
        numel(mz_labels), size(D,2));

end

% Sort mass axis into ascending order
[mz_labels, sort_mz_idx] = sort(mz_labels, 'ascend');
D = D(:, sort_mz_idx);
fprintf('Mass axis sorted: %.4f – %.4f Da\n', mz_labels(1), mz_labels(end));

% Basic sanity checks, can remove
assert(ndims(D)==2,       'D must be 2-D (pixels × m/z)');
assert(size(D,1)==IMG_ROWS*IMG_COLS, ...
    'Row count (%d) does not match IMG_ROWS×IMG_COLS (%d).', ...
    size(D,1), IMG_ROWS*IMG_COLS);

%% =========================================================
%  SECTION 3 — PREPROCESSING
%% =========================================================
fprintf('=== Preprocessing ===\n');

% Ensure non-negative (remove detector artefacts / baseline)
D = max(D, 0);

% Total-ion-count (TIC) normalisation — equalises pixel-to-pixel sensitivity
tic_vals = sum(D, 2);                       % n_pixels × 1
tic_vals(tic_vals == 0) = 1;               % avoid divide-by-zero
D_norm = D ./ tic_vals;                    % each pixel sums to 1

fprintf('TIC normalisation done.  Mean TIC = %.4g\n', mean(tic_vals));

%% =========================================================
%  SECTION 4 — BINNING
%% =========================================================
fprintf('=== Spatial binning (factor = %d) ===\n', BIN_FACTOR);

if BIN_FACTOR > 1
    D_bin = spatialBin(D_norm, IMG_ROWS, IMG_COLS, BIN_FACTOR);
    rows_bin = floor(IMG_ROWS / BIN_FACTOR);
    cols_bin = floor(IMG_COLS / BIN_FACTOR);
    fprintf('Binned image: %d × %d  →  %d pixels\n', ...
            rows_bin, cols_bin, size(D_bin,1));
else
    D_bin    = D_norm;
    rows_bin = IMG_ROWS;
    cols_bin = IMG_COLS;
    fprintf('No binning applied.\n');
end

%% =========================================================
%  SECTION 4b — MASS RANGE FILTER
%% =========================================================

% Build logical index over the full mass axis
mz_mask = true(size(mz_labels));

if ~isempty(MZ_MIN)
    mz_mask = mz_mask & (mz_labels >= MZ_MIN);
end
if ~isempty(MZ_MAX)
    mz_mask = mz_mask & (mz_labels <= MZ_MAX);
end

% Apply filter
D_filt      = D_bin(:, mz_mask);       % pixels × filtered_mz
mz_filt     = mz_labels(mz_mask);      % filtered mass axis (column vector)

n_kept   = sum(mz_mask);
n_total  = numel(mz_labels);

if n_kept == n_total
    fprintf('Mass range filter: all %d masses retained (no filter applied).\n', n_total);
elseif n_kept == 0
    error('Mass range filter removed ALL peaks. Check MZ_MIN / MZ_MAX values.');
else
    fprintf('Mass range filter: kept %d / %d masses  (%.2f – %.2f Da)\n', ...
            n_kept, n_total, mz_filt(1), mz_filt(end));
end


% --- Diagnostic: inspect filtered mass list and per-mass variance ---
fprintf('\n=== Filtered mass list (%.2f – %.2f Da, %d peaks) ===\n', ...
        mz_filt(1), mz_filt(end), numel(mz_filt));
col_vars = var(D_filt, 0, 1);   % variance across pixels for each mass
fprintf('  %-6s  %-12s  %-12s\n', 'm/z', 'Mean', 'Variance');
for ii = 1:numel(mz_filt)
    fprintf('  %-6.2f  %-12.4g  %-12.4g\n', mz_filt(ii), mean(D_filt(:,ii)), col_vars(ii));
end

% How many masses carry almost no variance?
low_var_thresh = 0.01 * max(col_vars);
n_lowvar = sum(col_vars < low_var_thresh);
fprintf('\nMasses with variance < 1%% of max: %d / %d\n', n_lowvar, numel(mz_filt));

%% =========================================================
%  SECTION 4c — Removes known contaminant m/z values
%% =========================================================
fprintf('=== Contaminant mass exclusion ===\n');

if ~isempty(MZ_EXCLUDE)
    % Build a logical mask: true = keep, false = contaminant
    excl_mask = false(size(mz_filt));   % marks columns to REMOVE

    for ci = 1:numel(MZ_EXCLUDE)
        hits = abs(mz_filt - MZ_EXCLUDE(ci)) <= MZ_EXCL_TOL;
        if any(hits)
            fprintf('  Excluding m/z %.4f  →  matched: %s Da\n', ...
                MZ_EXCLUDE(ci), ...
                num2str(mz_filt(hits)', '%.4f '));
        else
            fprintf('  WARNING: m/z %.4f not found within ±%.4f Da — skipped.\n', ...
                MZ_EXCLUDE(ci), MZ_EXCL_TOL);
        end
        excl_mask = excl_mask | hits;
    end

    keep_mask   = ~excl_mask;
    D_filt      = D_filt(:, keep_mask);
    mz_filt     = mz_filt(keep_mask);

    fprintf('Contaminant exclusion: removed %d mass(es), %d remaining.\n', ...
        sum(excl_mask), numel(mz_filt));
else
    fprintf('No contaminant masses specified — skipping.\n');
end
%% =========================================================
%  SECTION 5 — PCA / SVD  (component number estimation)
%% =========================================================
fprintf('=== PCA / SVD for component estimation ===\n');

% Mean-centre columns before SVD
D_mc   = D_filt - mean(D_filt, 1);
[~, S_svd, ~] = svd(D_mc, 'econ');
eigenvalues    = diag(S_svd).^2;
expl_var       = 100 * eigenvalues / sum(eigenvalues);
cum_var        = cumsum(expl_var);

% --- Scree plot ---
figure('Name','Scree Plot','NumberTitle','off','Color','w');
K_show = min(20, length(eigenvalues));
subplot(1,2,1);
semilogy(1:K_show, eigenvalues(1:K_show), 'o-', ...
         'Color',[0.2 0.4 0.8],'MarkerFaceColor',[0.2 0.4 0.8],'LineWidth',1.5);
xlabel('Principal component'); ylabel('Eigenvalue (log scale)');
title('Scree plot'); grid on; box off;
xline(N_COMP, '--r', sprintf('k = %d', N_COMP), 'LabelVerticalAlignment','bottom');

subplot(1,2,2);
bar(1:K_show, expl_var(1:K_show), 'FaceColor',[0.2 0.6 0.5],'EdgeColor','none');
hold on;
plot(1:K_show, cum_var(1:K_show), 's--', 'Color',[0.8 0.3 0.2], ...
     'MarkerFaceColor',[0.8 0.3 0.2],'LineWidth',1.2);
xlabel('Principal component'); ylabel('Variance explained (%)');
title('Variance explained'); legend('Individual','Cumulative','Location','east');
grid on; box off;
xline(N_COMP, '--r', sprintf('k = %d', N_COMP), 'LabelVerticalAlignment','bottom');

% Print summary table
fprintf('\n  PC   Eigenvalue   Expl.Var(%%)  Cum.Var(%%)\n');
fprintf('  --   ----------   -----------  ----------\n');
for i = 1:min(K_show, length(eigenvalues))
    marker = '';
    if i == N_COMP; marker = '  <-- selected k'; end
    fprintf('  %2d   %10.4g   %9.2f    %8.2f%s\n', ...
            i, eigenvalues(i), expl_var(i), cum_var(i), marker);
end
fprintf('\nReview the scree plot, then adjust N_COMP in Section 1 if needed.\n\n');

%% =========================================================
%  SECTION 6 — MCR-ALS
%% =========================================================
fprintf('=== MCR-ALS  (k = %d components) ===\n', N_COMP);

% --- Initialise spectra using SIMPLISMA-like: most dissimilar rows ---
S_init = initSpectra(D_filt, N_COMP);

% --- Run ALS ---
[C_mcr, S_mcr, lof_history] = mcrALS(D_filt, S_init, N_COMP, ...
                                       MAX_ITER, CONV_CRIT, APPLY_CLOSURE);

% --- Convergence plot ---
figure('Name','MCR-ALS Convergence','NumberTitle','off','Color','w');
plot(lof_history, 'o-', 'Color',[0.6 0.2 0.7], ...
     'MarkerFaceColor',[0.6 0.2 0.7],'LineWidth',1.5,'MarkerSize',4);
xlabel('Iteration'); ylabel('Lack of Fit (%)');
title('MCR-ALS convergence'); grid on; box off;

fprintf('Final lack-of-fit: %.4f %%\n', lof_history(end));

%% =========================================================
%  SECTION 7 — MCR-ALS LOADING SPECTRA
%  Each component is shown as a true mass spectrum:
%    • Vertical bar at each m/z position (height = normalised loading)
%    • No connecting line between peaks
%    • Top-5 peaks labelled with their m/z value
%% =========================================================
fprintf('=== Plotting loading spectra ===\n');

colors     = lines(N_COMP);
fig_height = max(250, 220 * N_COMP);

figure('Name','MCR-ALS Loading Spectra','NumberTitle','off', ...
       'Color','w','Position',[100 100 1050 fig_height]);

for k = 1:N_COMP
    ax = subplot(N_COMP, 1, k);

    % Retrieve this component's spectrum and normalise to max = 1
    sp    = S_mcr(k, :);
    sp    = sp / (max(sp) + eps);
    col_k = colors(k, :);

    % ----- True mass-spectrum: vertical bars only, no connecting line -----
    stem(mz_filt, sp, 'Marker', 'none', ...
         'Color', col_k, 'LineWidth', 1.0);
    hold on;

    % ----- Baseline -----
    plot([mz_filt(1) mz_filt(end)], [0 0], '-', ...
         'Color', [0.5 0.5 0.5], 'LineWidth', 0.5);

    % ----- Annotate top-5 peaks with m/z label -----
    [~, sort_idx] = sort(sp, 'descend');
    top_idx = sort_idx(1:min(5, numel(sp)));
    for p = 1:numel(top_idx)
        text(mz_filt(top_idx(p)), sp(top_idx(p)) + 0.05, ...
             sprintf('%.2f', mz_filt(top_idx(p))), ...
             'FontSize', 7.5, 'FontWeight', 'bold', ...
             'HorizontalAlignment', 'center', ...
             'Color', col_k * 0.65);
    end

    % ----- Axes formatting -----
    xlim([mz_filt(1) - 1,  mz_filt(end) + 1]);
    ylim([-0.05, 1.25]);
    ylabel('Norm. loading', 'FontSize', 9);
    title(sprintf('Component %d  —  loading spectrum  (m/z %.2f – %.2f Da)', ...
          k, mz_filt(1), mz_filt(end)), 'FontSize', 10);
    grid on; box off;
    ax.XMinorGrid = 'on';
    ax.GridAlpha  = 0.15;
end

xlabel('m/z (Da)', 'FontSize', 10);
sgtitle('MCR-ALS Loading Spectra', 'FontWeight', 'bold', 'FontSize', 12);

%% =========================================================
%  SECTION 8 — SPATIAL DISTRIBUTION MAPS
%% =========================================================
fprintf('=== Plotting spatial maps ===\n');

figure('Name','MCR-ALS Spatial Maps','NumberTitle','off', ...
       'Color','w','Position',[150 150 280*N_COMP 280]);

for k = 1:N_COMP
    subplot(1, N_COMP, k);
    conc_map = reshape(C_mcr(:,k), rows_bin, cols_bin);
    imagesc(conc_map);
    colormap(gca, hot);
    colorbar;
    axis image off;
    title(sprintf('Component %d', k));
end
sgtitle('Spatial Distribution Maps', 'FontWeight','bold');

fprintf('\n=== Done ===\n');

%% =========================================================
%  LOCAL FUNCTIONS
%% =========================================================

function D_bin = spatialBin(D, rows, cols, factor)
% Average pixels within each (factor × factor) block.
    rows_b = floor(rows / factor);
    cols_b = floor(cols / factor);
    n_mz   = size(D, 2);
    D_img  = reshape(D, rows, cols, n_mz);      % rows × cols × mz
    D_bin  = zeros(rows_b * cols_b, n_mz);
    idx = 0;
    for r = 1:rows_b
        r0 = (r-1)*factor + 1;
        r1 = r*factor;
        for c = 1:cols_b
            c0  = (c-1)*factor + 1;
            c1  = c*factor;
            idx = idx + 1;
            block = D_img(r0:r1, c0:c1, :);     % factor × factor × mz
            D_bin(idx,:) = mean(reshape(block, [], n_mz), 1);
        end
    end
end

function S_init = initSpectra(D, k)
% Greedy furthest-point initialisation (SIMPLISMA-inspired).
% Selects k pixels that are maximally dissimilar to each other.
    n_px  = size(D, 1);
    % Normalise rows for cosine-like distance
    norms = sqrt(sum(D.^2, 2)) + eps;
    Dn    = D ./ norms;
    
    selected = zeros(1, k);
    % First: pixel with highest variance (most informative)
    [~, selected(1)] = max(var(Dn, 0, 2));
    
    for i = 2:k
        % Similarity of every pixel to the already-selected set
        sim   = Dn * Dn(selected(1:i-1), :)';   % n_px × (i-1)
        max_s = max(sim, [], 2);                 % closest selected
        [~, selected(i)] = min(max_s);           % pick most distant
    end
    
    S_init = D(selected, :);   % k × n_mz  (raw, non-negative)
    % Ensure non-negative
    S_init = max(S_init, 0);
end

function [C, S, lof_history] = mcrALS(D, S_init, k, maxIter, convCrit, closure)
% MCR-ALS with non-negativity constraints on both C and S.
%
%   D        : n_px × n_mz   (data matrix)
%   S_init   : k  × n_mz    (initial spectral estimates)
%   Returns
%   C        : n_px × k      (concentration / score maps)
%   S        : k  × n_mz     (pure-component spectra)
%   lof_history : vector of lack-of-fit (%) per iteration

    n_mz        = size(D, 2);
    S           = S_init;
    lof_history = zeros(maxIter, 1);
    lof_prev    = Inf;

    for iter = 1:maxIter

        % --- Step 1: solve for C  (D ≈ C * S) ---
        % Least-squares: C = D * S' * (S * S')^-1
        C = D * S' / (S * S' + 1e-10 * eye(k));
        C = max(C, 0);                  % non-negativity on concentrations

        if closure
            row_sum = sum(C, 2) + eps;
            C = C ./ row_sum;           % rows sum to 1
        end

        % --- Step 2: solve for S  (D ≈ C * S) ---
        % Least-squares: S = (C'*C)^-1 * C' * D
        S = (C' * C + 1e-10 * eye(k)) \ (C' * D);
        S = max(S, 0);                  % non-negativity on spectra

        % --- Normalise S rows to unit length (avoids scale drift) ---
        row_norms = sqrt(sum(S.^2, 2)) + eps;
        S = S ./ row_norms;
        C = C .* row_norms';            % absorb scale into C

        % --- Lack of fit ---
        residual    = D - C * S;
        lof         = 100 * sqrt(sum(residual(:).^2) / sum(D(:).^2));
        lof_history(iter) = lof;

        % --- Convergence check ---
        rel_change = abs(lof_prev - lof) / (lof_prev + eps);
        if iter > 1 && rel_change < convCrit
            fprintf('  Converged at iteration %d  (LOF = %.4f %%)\n', iter, lof);
            lof_history = lof_history(1:iter);
            break;
        end
        lof_prev = lof;

        if mod(iter, 10) == 0
            fprintf('  Iter %3d  |  LOF = %.4f %%\n', iter, lof);
        end
    end

    % Final scale: normalise S to max-peak = 1, scale C accordingly
    for i = 1:k
        sc   = max(S(i,:)) + eps;
        S(i,:) = S(i,:) / sc;
        C(:,i) = C(:,i) * sc;
    end
end