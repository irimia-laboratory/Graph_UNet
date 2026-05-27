# Standard library
import os
import subprocess
from itertools import combinations

# Third-party libraries
import numpy as np
import pandas as pd
import nibabel as nib
import scipy.io
from scipy import sparse
from scipy.stats import ttest_ind
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests

# Class for post-processing model outputs
class PostProcessor():
    def __init__(self, 
                 first, # dependent on training data, i.e. whichever hemisphere is first in X, important this is never assumed
                 fsavg_path='/mnt/md0/softwares/freesurfer/subjects/fsaverage6/', # ico6
                 network_path='/mnt/md0/tempFolder/samAnderson/Yeo_JNeurophysiol11_FreeSurfer/fsaverage6/'
                 ):

        self.first = first
        self.fsavg_path = fsavg_path
        self.network_path = network_path

        # Get the labels and mask
        self.get_maps()

    # Get the FreeSurfer labels, names, and medial wall mask
    def get_maps(self):    
        
        # Load annotations
        lh_labels, _, lh_names = nib.freesurfer.read_annot(f'{self.fsavg_path}label/lh.aparc.a2009s.annot')
        rh_labels, _, rh_names = nib.freesurfer.read_annot(f'{self.fsavg_path}label/rh.aparc.a2009s.annot')

        # Decode names
        lh_names = [n.decode('utf-8') for n in lh_names]
        rh_names = [n.decode('utf-8') for n in rh_names]

        # Map each vertex to (region_name, hemisphere)
        lh_map = [(lh_names[label], 'lh') for label in lh_labels]
        rh_map = [(rh_names[label], 'rh') for label in rh_labels]

        # Combine in desired order
        if self.first == 'lh':
            vertex_map = np.array(lh_map + rh_map)
        elif self.first == 'rh':
            vertex_map = np.array(rh_map + lh_map)
        else:
            raise ValueError("self.first must be 'lh' or 'rh'")
        
        # Create a mask for the medial wall
        names = vertex_map[:, 0]
        names_lower = np.char.lower(names.astype(str))
        medial_wall_mask = np.isin(names_lower, ['unknown', 'medial_wall'])
                
        # Do the same for the networks
        lh_net_labels, _, lh_net_names = nib.freesurfer.read_annot(
            f'{self.network_path}label/lh.Yeo2011_7Networks_N1000.annot'
        )
        rh_net_labels, _, rh_net_names = nib.freesurfer.read_annot(
            f'{self.network_path}label/rh.Yeo2011_7Networks_N1000.annot'
        )

        # Decode names
        lh_net_names = [n.decode('utf-8') for n in lh_net_names]
        rh_net_names = [n.decode('utf-8') for n in rh_net_names]

        # Map exactly like regions
        lh_net_map = [lh_net_names[label] for label in lh_net_labels]
        rh_net_map = [rh_net_names[label] for label in rh_net_labels]

        # Rename networks
        id_to_name = {
            '7Networks_1': 'visual',
            '7Networks_2': 'somatomotor',
            '7Networks_3': 'dorsal_attn',
            '7Networks_4': 'ven_attn',
            '7Networks_5': 'limbic',
            '7Networks_6': 'fpn',
            '7Networks_7': 'dmn'
        }

        lh_net_map = [id_to_name.get(n, n) for n in lh_net_map]
        rh_net_map = [id_to_name.get(n, n) for n in rh_net_map]

        # Combine hemispheres
        if self.first == 'lh':
            network_map = np.array(lh_net_map + rh_net_map)
        else:
            network_map = np.array(rh_net_map + lh_net_map)
                                        
        # Save updated maps
        self.vertex_map = vertex_map
        self.network_map = network_map
        self.mask = ~medial_wall_mask # mask references all non-medial now

        return
    
    # Smooth the vertex data; helps to remove model artifact
    def _smooth_vertex_data(self, y_pred, y_true, n_iter=4, hops=2, medial_present=False):

        # Build mesh (full)
        _, faces = nib.freesurfer.read_geometry(f'{self.fsavg_path}surf/rh.pial')
        faces = np.vstack((faces, faces + (np.max(faces) + 1)))
        full_n_verts = faces.max() + 1

        # Identify which indices are cortical
        cortex_indices = np.where(self.mask)[0]
        n_verts = cortex_indices.size

        # Keep only cortex faces (from the original edges)
        valid_faces = np.all(self.mask[faces], axis=1)
        faces = faces[valid_faces]

        # Reindex to cortex space, creating index-based edges for a cortex-only array
        index_map = -np.ones(full_n_verts, dtype=int)
        index_map[cortex_indices] = np.arange(n_verts)
        faces = index_map[faces]

        # Build adjacency matrix
        row = np.concatenate([faces[:, 0], faces[:, 1], faces[:, 2]])
        col = np.concatenate([faces[:, 1], faces[:, 2], faces[:, 0]])
        data = np.ones(len(row), dtype=np.float32)

        adj = sparse.coo_matrix((data, (row, col)), shape=(n_verts, n_verts)).tocsr()
        adj = adj.maximum(adj.T)

        # Multi-hop neighborhood
        if hops > 1:
            neighborhood = adj.copy()
            for _ in range(hops - 1):
                neighborhood = neighborhood @ adj
            neighborhood = neighborhood + sparse.eye(n_verts)
        else:
            neighborhood = adj + sparse.eye(n_verts)

        # Normalize
        deg = np.array(neighborhood.sum(axis=1)).ravel()
        deg[deg == 0] = 1
        smoothing_op = sparse.diags(1.0 / deg) @ neighborhood

        # Handle input format to extract cortex array
        if medial_present: y_cortex = y_pred[:, self.mask]
        else: y_cortex = y_pred

        # Smooth array based on edges
        smoothed = y_cortex.copy()
        for _ in range(n_iter):
            smoothed = smoothed @ smoothing_op.T

        # Compute outputs
        vertex_means = np.mean(smoothed, axis=1)
        gbags = vertex_means - y_true
        lbags = smoothed - y_true[:, None]

        # Return format matches input
        if medial_present:
            full_smoothed = np.zeros_like(y_pred)
            full_smoothed[:, self.mask] = smoothed
            smoothed = full_smoothed

        return smoothed, gbags, lbags
        
    # Apply clipping and smoothing to raw predictions
    def clip_and_smooth(self, y_pred, y_true, medial_present=True, min_p=1, max_p=99):

        # Input validation
        assert y_pred.ndim == 2, f'y_pred must be 2D (subjects, vertices), got {y_pred.shape}'
        assert y_true.ndim == 1, f'y_true must be 1D (subjects), got {y_true.shape}' \
        f'Shape mismatch: y_pred {y_pred.shape}, y_true {y_true.shape}'

        # Remove the medial wall if present
        if medial_present: 
            y_pred = y_pred[:, self.mask]

        # Clip the outliers (1st to 99th percentile)
        lower_bound = np.percentile(y_pred, min_p, axis=1, keepdims=True)
        upper_bound = np.percentile(y_pred, max_p, axis=1, keepdims=True)
        y_pred_clipped = np.clip(y_pred, lower_bound, upper_bound)

        # Smooth the predictions
        y_pred_smoothed, gbags, lbags = self._smooth_vertex_data(
            y_pred_clipped,
            y_true
        )

        return y_pred_smoothed, gbags, lbags # medial wall removed

    # Account for model bias based on given or computed factors
    def bias_correct(self, y_pred, y_true, factors=None, method='behesti'):

        # Input validation
        assert len(y_true) == y_pred.shape[0], 'Mismatch in number of subjects'
        chr_ages_reshaped = y_true[:, np.newaxis]  # For broadcasting

        # Design matrix for linear regression (n_subjects, 2)
        X = np.column_stack([y_true, np.ones_like(y_true)])
        
        if method == 'behesti': # Behesti et al., 2019

            if factors is None:

                # Vectorized computation of age gap (n_subjects, n_vertices)
                age_gap = y_pred - chr_ages_reshaped
                
                # Vectorized least-squares solve for all vertices (2, n_vertices)
                coefficients = np.linalg.lstsq(X, age_gap, rcond=None)[0]

                # Get the average slope and bias
                avg_m = np.mean(coefficients[0])
                avg_b = np.mean(coefficients[1])
                factors = np.array([avg_m, avg_b])

                # Print out the factors
                print(f'Factors: {factors}')
            
            # Apply correction: pred - (mCA + b)
            bc_lbas = y_pred - ((factors[0] * chr_ages_reshaped) + factors[1])
            
        elif method == 'cole': # Cole et al., 2018
            
            if factors is None:

                # Vectorized solve for all vertices
                coefficients = np.linalg.lstsq(X, y_pred, rcond=None)[0]
                
                # Average local slopes (m) and intercepts (b)
                avg_m = np.mean(coefficients[0])
                avg_b = np.mean(coefficients[1])
                factors = np.array([avg_m, avg_b])  # Global (2,) factors

                # Print out the factors
                print(f'Factors: {factors}')
                
            # Apply correction: (pred - b)/m
            bc_lbas = (y_pred - factors[1]) / factors[0]
        
        # Compute errors
        bc_lbags = bc_lbas - chr_ages_reshaped
        bc_gbags = np.mean(bc_lbas, axis=1) - y_true
        
        return bc_lbags, bc_gbags, bc_lbas, factors

    # Add medial wall back into cortex array
    def _return_medial(self, arr):

        if arr.ndim == 1:
            full = np.zeros_like(self.mask, dtype=arr.dtype)
            full[self.mask] = arr

        elif arr.ndim == 2:
            full = np.zeros((arr.shape[0], self.mask.shape[0]), dtype=arr.dtype)
            full[:, self.mask] = arr

        else:
            raise ValueError("arr must be 1D or 2D")

        return full

    # Create the cortical plots through MATLAB
    def generate_cortical_plot(self, arr, save_to, medial_present, abs_limits=None, cbar_present=True):
        
        assert isinstance(cbar_present, bool)

        # Get the output directory
        output_dir = os.path.dirname(save_to)
        os.makedirs(output_dir, exist_ok=True)

        # Prepare the data to be visualized
        # add medial wall back in as 0s
        if not medial_present: arr = self._return_medial(arr)
        
        # align hemispheres
        if self.first == 'rh':  # visualization code assumes lh is first
            midpoint = len(arr) // 2
            arr = np.concatenate((arr[midpoint:], arr[:midpoint]))

        # create .mat file
        base, _ = os.path.splitext(save_to)
        mat_path = f'{base}.mat'
        scipy.io.savemat(mat_path, {'data': arr})

        # convert to MATLAB cell array syntax
        escaped_path = mat_path.replace("'", "''")
        matlab_file_list = f"{{'{escaped_path}'}}"

        # Run the MATLAB code (creates visualization in same location)
        # create the commands
        if abs_limits is not None:  # manual limits vs min and max limits
            command_primary = ["matlab", "-nodisplay", "-nosplash", "-r", f"generate_brain({matlab_file_list}, {{'lat_L','lat_R','med_R','med_L'}}, {abs_limits}, false); exit"]
            command_alt = ["matlab", "-nodisplay", "-nosplash", "-r", f"generate_brain({matlab_file_list}, {{'ant','dor','pos','ven'}}, {abs_limits}, {str(cbar_present).lower()}); exit"]                
        else:
            command_primary = ["matlab", "-nodisplay", "-nosplash", "-r", f"generate_brain({matlab_file_list}, {{'lat_L','lat_R','med_R','med_L'}}, [], false); exit"]
            command_alt = ["matlab", "-nodisplay", "-nosplash", "-r", f"generate_brain({matlab_file_list}, {{'ant','dor','pos','ven'}}, [], {str(cbar_present).lower()}); exit"]

        # send the commands to terminal (output suppressed)
        subprocess.run(command_primary, cwd="/mnt/md0/tempFolder/samAnderson/unet-gnn/visualization_code/", stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        subprocess.run(command_alt, cwd="/mnt/md0/tempFolder/samAnderson/unet-gnn/visualization_code/", stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        # remove the temp .mat file and return
        os.remove(mat_path)

        return

    # Compare cohorts BAGs to get LBAG difference and GBAGs
    def compare_cohorts(self, cohort_dict, n_bootstrap=500, bin_width=1.0):

        # Get CA and LBAs for each cohort
        y_trues = {k: np.load(v.y_test) for k, v in cohort_dict.items()}
        y_preds = {k: np.load(v.y_pred) for k, v in cohort_dict.items()}

        # Determine which cohort pairs have significantly different age distributions
        bootstrap_pairs = {}
        for c0, c1 in combinations(cohort_dict, 2):
            _, pval = ttest_ind(y_trues[c0], y_trues[c1], equal_var=False)
            bootstrap_pairs[(c0, c1)] = (pval < 0.05)

        # Helper: stratified age-matched sampling using bin_width
        def stratified_match_sample(y0, y1):

            # Define global age range across both cohorts
            min_age = min(y0.min(), y1.min())
            max_age = max(y0.max(), y1.max())

            # Create fixed-width bins (e.g., 1-year bins)
            bins = np.arange(min_age, max_age + bin_width, bin_width)

            # Will store sampled indices for each cohort across all bins
            idx0_all, idx1_all = [], []

            # Iterate through each age bin
            for i in range(len(bins) - 1):

                # Identify subjects in current bin for each cohort
                mask0 = (y0 >= bins[i]) & (y0 < bins[i+1])
                mask1 = (y1 >= bins[i]) & (y1 < bins[i+1])

                idx0_bin = np.where(mask0)[0]
                idx1_bin = np.where(mask1)[0]

                # If either cohort has no subjects in this age bin, skip
                if len(idx0_bin) == 0 or len(idx1_bin) == 0:
                    continue

                # Enforce equal representation by taking the smaller count
                # This removes density differences within this age bin
                n = min(len(idx0_bin), len(idx1_bin))

                # Sample WITH replacement within each bin
                # This preserves the age structure while allowing resampling variability
                s0 = np.random.choice(idx0_bin, n, replace=True)
                s1 = np.random.choice(idx1_bin, n, replace=True)

                # Accumulate sampled indices across bins
                idx0_all.append(s0)
                idx1_all.append(s1)

            if len(idx0_all) == 0:
                raise ValueError('No overlapping age bins between cohorts')

            # Concatenate all sampled bins into a single index array per cohort
            # Resulting idx0 and idx1 define matched samples with aligned age distributions
            return np.concatenate(idx0_all), np.concatenate(idx1_all)

        lbag_diff_maps = {}
        gbag_outputs = {}

        # Pairwise computation across all cohort combinations
        for c0, c1 in combinations(cohort_dict, 2):

            y0, p0 = y_trues[c0], y_preds[c0]
            y1, p1 = y_trues[c1], y_preds[c1]

            key = f'{c0}-{c1}'

            if bootstrap_pairs:
                print(f'Bootstrapping: {key}')

            # No bootstrap case (age distributions considered similar)
            if not bootstrap_pairs[(c0, c1)]:

                lbag0 = (p0 - y0[:, None]).mean(axis=0)
                lbag1 = (p1 - y1[:, None]).mean(axis=0)

                # Difference map between cohorts
                lbag_diff_maps[key] = lbag0 - lbag1

                gbag_outputs[key] = {
                    c0: (p0 - y0[:, None]).mean(axis=1),
                    c1: (p1 - y1[:, None]).mean(axis=1)
                }

            # Age-matched bootstrap case (age distributions differ)
            else:
                lbag_boot = []
                gbag0_boot = []
                gbag1_boot = []

                # Repeat resampling to build distributions
                for _ in range(n_bootstrap):

                    idx0, idx1 = stratified_match_sample(y0, y1)

                    lbag0 = (p0[idx0] - y0[idx0][:, None]).mean(axis=0)
                    lbag1 = (p1[idx1] - y1[idx1][:, None]).mean(axis=0)

                    lbag_boot.append(lbag0 - lbag1)

                    gbag0 = (p0[idx0] - y0[idx0][:, None]).mean(axis=1)
                    gbag1 = (p1[idx1] - y1[idx1][:, None]).mean(axis=1)

                    gbag0_boot.append(gbag0.mean())
                    gbag1_boot.append(gbag1.mean())

                # Stack bootstrap LBAG maps → shape: (n_bootstrap, n_vertices)
                lbag_diff_maps[key] = np.stack(lbag_boot, axis=0)

                # Store GBAG bootstrap distributions per cohort
                # shape: (n_bootstrap,)
                gbag_outputs[key] = {
                    c0: np.array(gbag0_boot),
                    c1: np.array(gbag1_boot)
                }

        return lbag_diff_maps, gbag_outputs

    # Get masks or averages per region or network
    def get_locations(self, arr=None, per='region', medial_present=False, return_type='mean'):

        # Ensure full-length array
        if return_type == 'mean':
            if arr is None: raise ValueError("arr must be provided when return_type='mean'")
            if not medial_present: arr = self._return_medial(arr)

        out = {}

        # If computing region-level values
        if per.lower() == 'region':

            names = self.vertex_map[:, 0]
            hemis = self.vertex_map[:, 1]

            # Create a dict with region-level values or masks
            for name, hemi in set(zip(names, hemis)):

                region_mask = (names == name) & (hemis == hemi) & (self.mask)

                if not np.any(region_mask):
                    continue

                if return_type == 'mean':
                    out[(name, hemi)] = arr[region_mask].mean()
                elif return_type == 'mask':
                    out[(name, hemi)] = region_mask
                else:
                    raise ValueError("return_type must be 'mean' or 'mask'")

            # Get the average across hemispheres (only for mean mode)
            if return_type == 'mean':
                region_groups = {}
                for (name, hemi), val in out.items():
                    region_groups.setdefault(name, []).append(val)

                for name, vals in region_groups.items():
                    out[(name, 'avg')] = np.mean(vals)

        # If computing network-level values
        elif per.lower() == 'network':

            networks = self.network_map
            names_lower = np.char.lower(networks.astype(str))

            # exclude medial wall
            valid_mask = (names_lower != 'freesurfer_defined_medial_wall') & (self.mask)

            unique_networks = np.unique(networks[valid_mask])

            for name in unique_networks:

                net_mask = (networks == name) & valid_mask

                if not np.any(net_mask):
                    continue

                if return_type == 'mean':
                    out[name] = arr[net_mask].mean()
                elif return_type == 'mask':
                    out[name] = net_mask
                else:
                    raise ValueError("return_type must be 'mean' or 'mask'")

        else:
            raise ValueError("per must be 'region' or 'network'")

        return out

    # Regress LBAGs against cognitive scores
    def regress_cognitive(self, cfg, medial_present=False):

        # === Prepare data === #

        # Load cognitive scores
        df = pd.read_csv(cfg.cog_path)

        # Remove max trail making score (ceiling effect; distorts regression)
        df.loc[df['TRABSCOR'] == 300, 'TRABSCOR'] = np.nan

        # Create subject-date key to align with imaging data
        df['subj_id_date'] = (
            df['PTID'] + "_" +
            pd.to_datetime(df['EXAMDATE'], errors='coerce').dt.strftime('%Y%m%d')
        )

        # Store regression outputs
        results = []

        # Iterate over cohorts
        for cohort, namespace in cfg.cohort_dict.items():

            # Subject IDs aligned with imaging arrays
            subj_id_date_cohort = np.array([
                s.strip() for s in np.load(namespace.subj_path).astype(str)
            ])

            # Map subject_id_date to an index
            subj_to_idx = {s: i for i, s in enumerate(subj_id_date_cohort)}

            # Load cohort arrays once
            lbags_all = np.load(namespace.lbag_path)
            if not medial_present: lbags_all = self._return_medial(lbags_all) # so masks align
            cas_all = np.load(namespace.ca_path, mmap_mode='r')

            # Restrict df to subjects present in this cohort
            df_cohort = df[df['subj_id_date'].isin(subj_id_date_cohort)].copy()

            # Iterate over cognitive tests
            for test in cfg.tests_to_include:

                # Select subjects with valid test scores
                valid_df = df_cohort[df_cohort[test].notna()].copy()
                if len(valid_df) == 0:
                    print(f'Warning: no valid tests found for {test} for {cohort} subjects')
                    continue

                # Enforce ordering to match imaging arrays
                valid_df['subj_id_date'] = pd.Categorical(
                    valid_df['subj_id_date'],
                    categories=subj_id_date_cohort,
                    ordered=True
                )
                valid_df = valid_df.sort_values('subj_id_date').drop_duplicates('subj_id_date')

                # Sub-select indices (aligned with cohort arrays)
                subj_indices = np.array(
                    [subj_to_idx[s] for s in valid_df['subj_id_date']],
                    dtype=int
                )

                # === Extract and normalize variables === #

                # Scores
                scores_raw = valid_df[test].values
                scores = (scores_raw - scores_raw.mean()) / scores_raw.std()

                # Invert if needed (higher = worse cognition)
                if not cfg.test_relations[test]: # require explicitly given
                    scores = -scores

                # Education
                edu_raw = valid_df['PTEDUCAT'].values
                edu = (edu_raw - edu_raw.mean()) / edu_raw.std()

                # Sex
                sex = valid_df['SEX'].values

                # Chronological age
                ca = cas_all[subj_indices]

                # Build design matrix
                X_df = pd.DataFrame({
                    'sex': sex,
                    'education': edu,
                    'test_score': scores,
                    'chronological_age': ca
                })

                # Iterate over the per-region or per-network masks
                for mask_name, location_mask in cfg.masks.items():

                    # Extract per-subject BAG averaged across the selected region
                    y_loc = lbags_all[subj_indices][:, location_mask].mean(axis=1) # (subjects)

                    # Fit regression
                    fit = sm.OLS(y_loc, sm.add_constant(X_df)).fit()
                    ci_low, ci_high = fit.conf_int().loc['test_score']

                    # Store results
                    results.append({
                        'cohort': cohort,
                        'test': test,
                        'location': mask_name,
                        'coef': fit.params['test_score'],
                        'raw_pval': fit.pvalues['test_score'],
                        'r_squared': fit.rsquared,
                        'ci_low' : ci_low,
                        'ci_high' : ci_high
                    })

                    # Sanity check (alignment)
                    assert all(subj_id_date_cohort[i] == s for i, s in zip(subj_indices, valid_df['subj_id_date']))

        # Convert results to dataframe
        df_results = pd.DataFrame(results)

        # Multiple comparison correction
        df_results['adj_pval'] = (
            df_results.groupby(['test'])['raw_pval'] # correct across every region for a given test, across cohorts
            .transform(lambda p: multipletests(p, method=cfg.pval_method)[1])
        )

        return df_results