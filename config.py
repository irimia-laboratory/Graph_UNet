from types import SimpleNamespace

# Set master configs

# Paths
raw_data_path = '/mnt/md0/projects/graph_unet/atlas_projected_surfaces/raw/' # where data is stored as raw FreeSurfer files
processed_data_path = '/mnt/md0/projects/graph_unet/atlas_projected_surfaces/processed/' # where data is stored as numpy files
postprocessed_data_path = '/mnt/md0/tempFolder/samAnderson/unet-gnn/model_outputs/'
metadata_path = '/mnt/md0/subjectdata/MetaData/'

# Training data paths
X_train = f'{processed_data_path}training/X_train_ico6.npy'
subj_train = f'{processed_data_path}training/subj_train_ico6.npy'
y_train = f'{processed_data_path}training/y_train_ico6.npy'

# Testing data paths
X_test_CN = f'{processed_data_path}ADNI/X_ADNI_CN_ico6_norm.npy'
subj_test_CN = f'{processed_data_path}ADNI/subj_ADNI_CN_ico6.npy'
y_test_CN = f'{processed_data_path}ADNI/y_ADNI_CN_ico6.npy'

X_test_MCI = f'{processed_data_path}ADNI/X_ADNI_MCI_ico6_norm.npy'
subj_test_MCI = f'{processed_data_path}ADNI/subj_ADNI_MCI_ico6.npy'
y_test_MCI = f'{processed_data_path}ADNI/y_ADNI_MCI_ico6.npy'

X_test_AD = f'{processed_data_path}ADNI/X_ADNI_AD_ico6_norm.npy'
subj_test_AD = f'{processed_data_path}ADNI/subj_ADNI_AD_ico6.npy'
y_test_AD = f'{processed_data_path}ADNI/y_ADNI_AD_ico6.npy'

# Training
training_batch_size = 8
num_epochs = 60
lr = 5e-4
weight_decay = 3e-4
# Scheduler settings
use_scheduler=True
scheduler_factor=0.2
scheduler_patience=4
scheduler_min_lr=1e-6   

# Model Structure
feature_sizes = [256, 512, 512, 256, 256]
dropout_levels = [0, 0, 0, 0, 0]

# Testing
testing_batch_size = 1 # batch size whenever testing the model
model_weights = f'{postprocessed_data_path}runs/trained_weights.pth' # where trained model weights reside

# Preprocessing settings
features = ['area', 'curvature', 'sulcal_depth', 'thickness', 'WM-GM_ratio'] # always sorted BASED ON FILE EXTENSION in preprocessing, so make sure these align
ico = 6
first = 'rh' # 'rh' or 'lh'; indicates which hemisphere is represented first (index 0) in the numpy files
n_vertices = 81924 # number of vertices in ico6; to compute: 2*(10*(4^ico)+2)

# Dict associating regions with lobes
region_to_lobe_dict = {
        'Frontal': [
            'G_front_sup', 'G_front_middle',
            'G_front_inf-Opercular', 'G_front_inf-Triangul', 'G_front_inf-Orbital',
            'G_precentral', 'G_rectus', 'G_orbital',
            'G_and_S_frontomargin', 'G_and_S_transv_frontopol',
            'S_front_sup', 'S_front_middle', 'S_front_inf',
            'S_precentral-sup-part', 'S_precentral-inf-part',
            'S_orbital-H_Shaped', 'S_orbital_lateral',
            'S_orbital_med-olfact', 'S_suborbital'
        ],

        'Parietal': [
            'G_postcentral', 'G_parietal_sup',
            'G_pariet_inf-Angular', 'G_pariet_inf-Supramar',
            'G_precuneus', 'G_and_S_paracentral',
            'S_postcentral', 'S_intrapariet_and_P_trans',
            'S_subparietal', 'S_parieto_occipital',
            'S_interm_prim-Jensen'
        ],

        'Temporal': [
            'G_temp_sup-Lateral', 'G_temp_sup-G_T_transv',
            'G_temp_sup-Plan_polar', 'G_temp_sup-Plan_tempo',
            'G_temporal_middle', 'G_temporal_inf',
            'G_oc-temp_lat-fusifor',
            'G_oc-temp_med-Lingual', 'G_oc-temp_med-Parahip',
            'S_temporal_sup', 'S_temporal_inf',
            'S_temporal_transverse',
            'S_oc-temp_lat', 'S_oc-temp_med_and_Lingual',
            'S_collat_transv_ant', 'S_collat_transv_post', 'Pole_temporal'
        ],

        'Occipital': [
            'G_occipital_sup', 'G_occipital_middle',
            'G_cuneus', 'Pole_occipital',
            'G_and_S_occipital_inf',
            'S_calcarine',
            'S_oc_sup_and_transversal',
            'S_oc_middle_and_Lunatus',
            'S_occipital_ant'
        ],

        'Cingulate': [
            'G_and_S_cingul-Ant',
            'G_and_S_cingul-Mid-Ant',
            'G_and_S_cingul-Mid-Post',
            'G_cingul-Post-dorsal',
            'G_cingul-Post-ventral',
            'G_subcallosal',
            'S_cingul-Marginalis',
            'S_pericallosal'
        ],

        'Insula': [
            'G_insular_short',
            'G_Ins_lg_and_S_cent_ins',
            'S_circular_insula_ant',
            'S_circular_insula_sup',
            'S_circular_insula_inf'
        ],

        'Central': [
            'S_central',
            'G_and_S_subcentral'
        ],

        'Lateral_Fissure': [
            'Lat_Fis-ant-Horizont',
            'Lat_Fis-ant-Vertical',
            'Lat_Fis-post'
        ],

        'Medial': [
            'Medial_wall'
        ]
    }    

# Define a function for visualizing GBAGs, from one or multiple cohorts
def gbag_kde(cohorts, xlim=None, save_to=None, return_plot=False):  # cohorts: {label: GBAGs}

    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns

    # Style
    sns.set_style('whitegrid')
    plt.figure(figsize=(11, 6))

    # Select KDE colors
    color_map = ['royalblue', 'darkorange',"#ce3b3b"]
    colors = color_map[:len(cohorts)]

    # Plot each cohort
    for (label, data), color in zip(cohorts.items(), colors):

        # KDE
        sns.kdeplot(
            data,
            color=color,
            label=label,
            fill=True,
            alpha=0.4,
            linewidth=2
        )

        # Median line
        plt.axvline(
            np.mean(data),
            color=color,
            linestyle='--',
            linewidth=2
        )

    # X-axis limits
    all_data = np.concatenate(list(cohorts.values()))

    if xlim is None:
        x_step = 1
        xtick_min = np.floor(np.min(all_data) / x_step) * x_step
        xtick_max = np.ceil(np.max(all_data) / x_step) * x_step
        plt.xlim(xtick_min, xtick_max)
    else:
        plt.xlim(xlim[0], xlim[1])

    # Ticks
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)

    # Labels
    plt.xlabel('Global Age Gap (years)', size=16, labelpad=10)
    plt.ylabel('Density', size=16, labelpad=10)

    plt.legend(fontsize=14)
    plt.tight_layout()

    # Save if requested, adding .png
    if save_to is not None:
        if not save_to.lower().endswith('.png'):
            save_to = f'{save_to}.png'
        plt.savefig(save_to, dpi=300, bbox_inches='tight')

    if return_plot: return
    else: plt.close(plt.gcf()); return

# Define a function for building a df of region and network level BAGs
def build_location_df(data_dict, postprocessing_object, per, 
                    medial_present=False, add_diffs=False, diff_pairs=None
                    ):

    import pandas as pd

    # Get location means from array
    out = {}
    for label, arr in data_dict.items():  # { cohort/sex : LBAGs }

        out[label] = postprocessing_object.get_locations(
            arr,
            per=per,
            medial_present=medial_present,
            return_type='mean'
        )

    # Convert to df
    df = pd.DataFrame(out)

    # If specified, add differences (AD-CN, etc.)
    if add_diffs and diff_pairs is not None:
        for name, (a, b) in diff_pairs.items():
            df[name] = df[a] - df[b]

    # Rename columns to reflect location type and hemisphere
    df = df.reset_index()

    # Add hemisphere column if applicable
    if per == 'region':
        df = df.rename(columns={
            df.columns[0]: per,
            df.columns[1]: 'hemisphere'
        })

        # Clarify hemisphere labelling
        df['hemisphere'] = df['hemisphere'].replace({
            'lh': 'Left',
            'rh': 'Right',
            'avg': 'Average'
        })
    else: # network
        df = df.rename(columns={
            df.columns[0]: per
        })

    return df

# Define a function for displaying cortical plots (diff map, saliency, etc.)
def plot_brain_pairs(plot_paths, plot_titles=None, subplot_labels=None, 
                     figsize=(12, None), save_path=None, dpi=1200, 
                     # Spacing params
                     wspace=-0.30, hspace=0.14, label_y_pos=[1.10, 0.035],
                     title_x_pos=0.5,
                     # Colorbar params
                     show_colorbar=False, cmap='coolwarm', 
                     vmin=-2.5, vmax=2.5, colorbar_label=None
                     ):

    import matplotlib.pyplot as plt
    import matplotlib.image as mpimg
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize
    import numpy as np
    import string

    n_images = len(plot_paths)
    n_rows = int(np.ceil(n_images / 2))

    # Default subplot labels: A, '', B, '', ...
    if subplot_labels is None:
        letters = list(string.ascii_uppercase)
        subplot_labels = []
        for i in range(n_rows):
            subplot_labels.extend([letters[i], ''])

    # Auto height scaling
    if figsize[1] is None:
        figsize = (figsize[0], 4 * n_rows)

    fig, axes = plt.subplots(n_rows, 2, figsize=figsize)
    axes = axes.ravel()

    row_idx = 0

    for i, ax in enumerate(axes):

        if i < n_images:
            img = mpimg.imread(plot_paths[i])
            ax.imshow(img)
            ax.axis("off")

            # Subplot label 
            ax.text(
                0.01, label_y_pos[0], subplot_labels[i],
                transform=ax.transAxes,
                fontsize=20, fontweight="bold",
                va="top", ha="left",
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.7)
            )

            # Row-level title (centered)
            if plot_titles is not None:
                if i % 2 == 1:
                    
                    y_pos = 1 - (row_idx / n_rows) - label_y_pos[1]
                    fig.text(
                        title_x_pos, y_pos,
                        plot_titles[row_idx],
                        ha='center', va='bottom',
                        fontsize=18
                    )
                    row_idx += 1
        else:
            ax.axis("off")

    plt.tight_layout()
    fig.subplots_adjust(wspace=wspace, hspace=hspace)
    
    # Shared colorbar
    if show_colorbar:

        norm = Normalize(
            vmin=vmin,
            vmax=vmax
        )

        sm = ScalarMappable(
            norm=norm,
            cmap=cmap
        )

        sm.set_array([])

        cbar = fig.colorbar(
            sm,
            ax=axes,
            orientation='vertical',
            fraction=0.025,
            pad=0.02
        )

        cbar.ax.tick_params(labelsize=18)

        if colorbar_label is not None:

            cbar.set_label(
                colorbar_label,
                fontsize=18
            )

    if save_path is not None:
        fig.savefig(
            save_path,
            dpi=dpi,
            bbox_inches="tight",
            pad_inches=0.1,
            pil_kwargs={"quality": 95}
        )

    return

# === Set configs === #

# preprocessing.py
preprocessing_config = SimpleNamespace(

    # Feature configs
    features = features,
    # converts common names to actual file extensions
    feature_map_dict = {
        'area' : 'area',
        'curvature' : 'curv',
        'sulcal_depth' : 'sulc',
        'thickness' : 'thickness',
        'WM-GM_ratio' : 'w-g.pct.mgh'
    },

    # Preprocessing setting configs
    ico = ico,
    first = first,

    # Path configs
    raw_data_path = raw_data_path,
    processed_data_path = processed_data_path,

    # Dataset configs
    datasets = SimpleNamespace(

        # Training
        UKBB = SimpleNamespace(
            type='train',
            folder=f'UKBB_ico{ico}_all_pruned/',
            metadata_path='/mnt/md0/tempFolder/samAnderson/datasets/UKBB_demographic_with_sex.csv',
            age_id_date_sex=['age', 'eid', 'date', 'sex'],
            sex_mapping={'0': 'Female', '1': 'Male'}
        ),

        NACC = SimpleNamespace(
            type='train',
            folder=f'NACC_ico{ico}_all/',
            metadata_path='/mnt/md0/tempFolder/samAnderson/datasets/NACC_master.csv',
            age_id_date_sex=['age', 'ID', 'study_time', 'sex'],
            sex_mapping={'Female': 'Female', 'Male': 'Male'}
        ),

        IXI = SimpleNamespace(
            type='train',
            folder=f'IXI_ico{ico}_all/',
            metadata_path='/mnt/md0/tempFolder/samAnderson/datasets/IXI_master.csv',
            age_id_date_sex=['AGE', 'IXI_ID', 'STUDY_DATE', 'SEX_ID (1=m, 2=f)'],
            sex_mapping={'2': 'Female', '1': 'Male'}
        ),

        # Testing
        ADNI_CN = SimpleNamespace(
            type='test',
            folder=f'ADNI_ico{ico}_all/',
            metadata_path='/mnt/md0/subjectdata/MetaData/ADNI/ADNI1-4_master.csv',
            age_id_date_sex=['Age', 'Subject ID', 'Study Date', 'Sex'],
            sex_mapping={'F': 'Female', 'M': 'Male'}
        ),

        ADNI_MCI = SimpleNamespace(
            type='test',
            folder=f'ADNI_ico{ico}_all/',
            metadata_path='/mnt/md0/subjectdata/MetaData/ADNI/ADNI1-4_master.csv',
            age_id_date_sex=['Age', 'Subject ID', 'Study Date', 'Sex'],
            sex_mapping={'F': 'Female', 'M': 'Male'}
        ),

        ADNI_AD = SimpleNamespace(
            type='test',
            folder=f'ADNI_ico{ico}_all/',
            metadata_path='/mnt/md0/subjectdata/MetaData/ADNI/ADNI1-4_master.csv',
            age_id_date_sex=['Age', 'Subject ID', 'Study Date', 'Sex'],
            sex_mapping={'F': 'Female', 'M': 'Male'}
        )
    )
)

# 1_precursor_analysis.ipynb
precursor_config = SimpleNamespace(

    # Paths
    model_weights = model_weights,
    postprocessed_data_path = postprocessed_data_path,

    # Feature configs
    features = features,

    # converts common names to actual file extensions
    feature_map_dict = {
        'area' : 'area',
        'curvature' : 'curv',
        'sulcal_depth' : 'sulc',
        'thickness' : 'thickness',
        'WM-GM_ratio' : 'w-g.pct.mgh'
    },

    # Dataset configs
    datasets = {
        'UKBB' :  {
            'set': 'training',
            'raw_data': f'{raw_data_path}UKBB_ico6_all_pruned/',
            'metadata': f'{metadata_path}UKBB/UKBB_demographic_with_sex.csv',
            'id_col': 'eid',
            'date_col': 'date',
            'age_col': 'age',
            'sex_col': 'sex',
            'sex_mapping': {'Female': 0, 'Male': 1},
            'data_preproc': ['all_str'],
            'select': 'all'
        },

        'IXI' : {
            'set': 'training',
            'raw_data': f'{raw_data_path}IXI_ico6_all/',
            'metadata': f'{metadata_path}IXI/IXI_master.csv',
            'id_col': 'IXI_ID',
            'date_col': 'STUDY_DATE',
            'age_col': 'AGE',
            'sex_col': 'SEX_ID (1=m, 2=f)',
            'sex_mapping': {'Female': 2, 'Male': 1},
            'data_preproc': ['DD/MM/YY conversion'],
            'select': 'all'
        },

        'NACC' : {
            'set': 'training',
            'raw_data': f'{raw_data_path}NACC_ico6_all/',
            'metadata': f'{metadata_path}NACC/NACC_master.csv',
            'id_col': 'NACCID',
            'date_col': 'MRIDATE',
            'age_col': 'AGEDECIMAL_MRI',
            'sex_col': 'SEX',
            'sex_mapping': {'Female': 2, 'Male': 1}, # unsure, whichever is more female
            'data_preproc': ['remove_-'],
            'select': 'DX==CN'
        },

        'ADNI_CN' : {
            'set': 'testing',
            'raw_data': f'{raw_data_path}ADNI_ico6_all/',
            'metadata': f'{metadata_path}ADNI/ADNI1-4_master.csv',
            'id_col': 'Subject ID',
            'date_col': 'Study Date',
            'age_col': 'Age',
            'sex_col': 'Sex',
            'sex_mapping': {'Female': 'F', 'Male': 'M'},
            'data_preproc': ['full_date'],
            'select': 'Research Group==CN'
        },

        'ADNI_MCI' : {
            'set': 'testing',
            'raw_data': f'{raw_data_path}ADNI_ico6_all/',
            'metadata': f'{metadata_path}ADNI/ADNI1-4_master.csv',
            'id_col': 'Subject ID',
            'date_col': 'Study Date',
            'age_col': 'Age',
            'sex_col': 'Sex',
            'sex_mapping': {'Female': 'F', 'Male': 'M'},
            'data_preproc': ['full_date'],
            'select': 'Research Group==MCI'
        },

        'ADNI_AD' : {
            'set': 'testing',
            'raw_data': f'{raw_data_path}ADNI_ico6_all/',
            'metadata': f'{metadata_path}ADNI/ADNI1-4_master.csv',
            'id_col': 'Subject ID',
            'date_col': 'Study Date',
            'age_col': 'Age',
            'sex_col': 'Sex',
            'sex_mapping': {'Female': 'F', 'Male': 'M'},
            'data_preproc': ['full_date'],
            'select': 'Research Group==AD'
        }
    },

    # Data config
    X_train=X_train,
    y_train=y_train,
    
    # Training configs
    batch_size=training_batch_size,
    num_epochs=num_epochs,
    lr=lr, 
    weight_decay=weight_decay,
    feature_sizes=feature_sizes,
    dropout_levels=dropout_levels,

    use_scheduler=use_scheduler,
    scheduler_factor=scheduler_factor,
    scheduler_patience=scheduler_patience,
    scheduler_min_lr=scheduler_min_lr,

    # Preprocessing configs
    first=first,
    n_vertices=n_vertices,
    
    # Configs for getting the unprocessed figure 
    unproc_config = SimpleNamespace(
        X_test = X_test_CN,
        y_test = y_test_CN,
        batch_size = testing_batch_size, 
        vis_path = f'{postprocessed_data_path}ADNI_CN/visualizations/',
        n_vertices=n_vertices
    )
)   

# 2_pathology_analysis.ipynb
pathology_config = SimpleNamespace(

    # Path configs
    cross_cohort_path = f'{postprocessed_data_path}cross_cohort/',

    # Preprocessing configs
    n_vertices = n_vertices,
    first = first,

    # Functions
    gbag_kde = gbag_kde,
    build_location_df = build_location_df,
    plot_brain_pairs = plot_brain_pairs,

    # Testing configs
    batch_size = testing_batch_size,
    feature_sizes = feature_sizes,
    dropout_levels = dropout_levels,
    model_weights = model_weights,
    postprocessed_data_path = postprocessed_data_path,

    # Data configs (training)
    X_train = X_train,
    y_train = y_train,
    training_factors_path = f'{postprocessed_data_path}train/',

    # Data structuring configs
    region_to_lobe_dict = region_to_lobe_dict,
    cohort_dict = {
        'CN': SimpleNamespace(
            X_test=X_test_CN,
            y_test=y_test_CN,
            vis_path=f'{postprocessed_data_path}ADNI_CN/visualizations/',
            array_path=f'{postprocessed_data_path}ADNI_CN/arrays/'
        ),
        'MCI': SimpleNamespace(
            X_test=X_test_MCI,
            y_test=y_test_MCI,
            vis_path=f'{postprocessed_data_path}ADNI_MCI/visualizations/',
            array_path=f'{postprocessed_data_path}ADNI_MCI/arrays/'
        ),
        'AD': SimpleNamespace(
            X_test=X_test_AD,
            y_test=y_test_AD,
            vis_path=f'{postprocessed_data_path}ADNI_AD/visualizations/',
            array_path=f'{postprocessed_data_path}ADNI_AD/arrays/'
        )
    }
)

# 3_cognition_analysis.ipynb
cognition_config = SimpleNamespace(
    
    # Path to cognitive scores
    cog_path=f'{metadata_path}ADNI/ADNI_master.csv',
    cross_cohort_path = f'{postprocessed_data_path}cross_cohort/all/',
    
    # Cohorts for cognitive test analysis
    cohort_dict = {
        'CN': SimpleNamespace(
            subj_path=subj_test_CN,
            ca_path=y_test_CN,
            lbag_path=f'{postprocessed_data_path}ADNI_CN/arrays/bc_lbas.npy'
        ),
        #'MCI': SimpleNamespace(
        #    subj_path=subj_test_MCI,
        #    ca_path=y_test_MCI,
        #    lbag_path=f'{postprocessed_data_path}ADNI_MCI/arrays/bc_lbas.npy'
        #),
        'AD': SimpleNamespace(
            subj_path=subj_test_AD,
            ca_path=y_test_AD,
            lbag_path=f'{postprocessed_data_path}ADNI_AD/arrays/bc_lbas.npy'
        )        
    },
    
    # Tests to include in cognitive analysis
    tests_to_include=[
        'ADAS11', 'CDRSB', 'DIGITSCOR', 'EcogPtTotal', 
        'EcogSPTotal', 'FAQ', 'LDELTOTAL', 'MMSE', 'MOCA', 
        'RAVLT_immediate', 'RAVLT_learning', 'RAVLT_perc_forgetting', 
        'TRABSCOR'
    ],
    
    # Test directionality (if True, higher score = worse cognition)
    test_relations={
        'ADAS11': True,
        'CDRSB': True,
        'DIGITSCOR': False,
        'EcogPtTotal': True,
        'EcogSPTotal': True,
        'FAQ': True,
        'LDELTOTAL': False,
        'MMSE': False,
        'MOCA': False,
        'RAVLT_immediate': False,
        'RAVLT_learning': False,
        'RAVLT_perc_forgetting': True,
        'TRABSCOR': True
    },

    # Statistics
    pval_thresh=0.05,
    pval_method='fdr_bh'
    
)

# 4_feature_analysis.ipynb
feature_config = SimpleNamespace(
    
    # Paths
    cross_cohort_path = f'{postprocessed_data_path}cross_cohort/',

    # Functions
    plot_brain_pairs = plot_brain_pairs,
    build_location_df = build_location_df,
    
    # Preprocessing
    features=features,
    first=first,
    n_vertices = n_vertices,
    
    # Testing
    testing_batch_size = testing_batch_size, # it saves multiple models at once
    feature_sizes = feature_sizes,
    dropout_levels = dropout_levels,
    model_weights = model_weights,
    CN_factors = f'{postprocessed_data_path}ADNI_CN/arrays/CN_factors.npy',

    # Data structuring configs
    region_to_lobe_dict = region_to_lobe_dict,
    cohort_dict = {
        'CN': SimpleNamespace(
            X_test=X_test_CN,
            y_test=y_test_CN,
            vis_path=f'{postprocessed_data_path}ADNI_CN/visualizations/ablation/',
            array_path=f'{postprocessed_data_path}ADNI_CN/arrays/ablation/'
        ),
        'MCI': SimpleNamespace(
            X_test=X_test_MCI,
            y_test=y_test_MCI,
            vis_path=f'{postprocessed_data_path}ADNI_MCI/visualizations/ablation/',
            array_path=f'{postprocessed_data_path}ADNI_MCI/arrays/ablation/'
        ),
        'AD': SimpleNamespace(
            X_test=X_test_AD,
            y_test=y_test_AD,
            vis_path=f'{postprocessed_data_path}ADNI_AD/visualizations/ablation/',
            array_path=f'{postprocessed_data_path}ADNI_AD/arrays/ablation/'
        )
    }

    # Integrated gradients
    #grad_vis_path=f'{postprocessed_data_path}ADNI_CN/visualizations/saliency/',
    #grad_array_path=f'{postprocessed_data_path}ADNI_CN/arrays/saliency/',
    #n_steps = 50
    # set_baseline = ... # right now this uses stochastic samplying from the baseline. Probably want to modify
)

# 5_sex_analysis.ipynb
sex_config = SimpleNamespace(
    
    # Paths
    postprocessed_data_path = postprocessed_data_path,
    array_path = f'{postprocessed_data_path}ADNI_CN/sex/arrays/',
    vis_path   = f'{postprocessed_data_path}ADNI_CN/sex/visualizations/',

    # Functions
    gbag_kde = gbag_kde,
    plot_brain_pairs = plot_brain_pairs,
    build_location_df = build_location_df,
    
    # Dict for mapping regions and lobes
    region_to_lobe_dict = region_to_lobe_dict,
        
    # Preprocessing configs
    first=first,
    
    # Data
    y_test_CN=y_test_CN,
    sex_labels=f'{processed_data_path}ADNI/sex_ADNI_CN_ico6.npy',
    subj_test_CN=subj_test_CN,
    bc_lbags = f'{postprocessed_data_path}ADNI_CN/arrays/bc_lbags.npy',
    bc_gbags = f'{postprocessed_data_path}ADNI_CN/arrays/bc_gbags.npy',
    
)

# 6_model_comparisons.ipynb
model_comparison_config = SimpleNamespace(

    # Path configs
    vis_path = f'{postprocessed_data_path}cross_cohort/AD-CN/visualizations/',

    # Preprocessing configs
    first = first,
    n_vertices = n_vertices,

    # Data configs
    # cnn csv
    cnn_path = '/mnt/md0/tempFolder/samAnderson/unet-gnn/ADNI_Complete_regionalAGs.csv',
    gnn_path = '/mnt/md0/tempFolder/samAnderson/unet-gnn/GNN_regionalAGs.csv',
    # lbags (need for subject matching)
    adni_cn_lbags = f'{postprocessed_data_path}ADNI_CN/arrays/bc_lbags.npy',
    adni_mci_lbags = f'{postprocessed_data_path}ADNI_MCI/arrays/bc_lbags.npy',
    adni_ad_lbags = f'{postprocessed_data_path}ADNI_AD/arrays/bc_lbags.npy',
    # subjects
    adni_cn_subjs = f'{processed_data_path}ADNI/subj_ADNI_CN_ico6.npy',
    adni_mci_subjs = f'{processed_data_path}ADNI/subj_ADNI_MCI_ico6.npy',
    adni_ad_subjs = f'{processed_data_path}ADNI/subj_ADNI_AD_ico6.npy',

    # Dict associating regions with lobes or networks
    region_to_lobe_dict = region_to_lobe_dict,
    network_to_region_dict = {
        "Visual": [
            "G_occipital_sup", "G_occipital_middle",
            "G_cuneus", "Pole_occipital",
            "G_and_S_occipital_inf",
            "S_calcarine",
            "S_oc_sup_and_transversal",
            "S_oc_middle_and_Lunatus",
            "S_occipital_ant",
        ],

        "Somatomotor": [
            "G_precentral", "G_postcentral",
            "S_precentral-sup-part", "S_precentral-inf-part",
            "S_postcentral",
            "S_central",
            "G_and_S_paracentral",
            "G_and_S_subcentral",
        ],

        "DorsalAttention": [
            "G_parietal_sup",
            "G_pariet_inf-Angular",
            "G_pariet_inf-Supramar",
            "G_precuneus",
            "S_intrapariet_and_P_trans",
            "S_parieto_occipital",
            "S_interm_prim-Jensen",
        ],

        "VentralAttention": [
            "G_front_inf-Opercular",
            "G_front_inf-Triangul",
            "G_insular_short",
            "G_Ins_lg_and_S_cent_ins",
            "S_circular_insula_ant",
            "S_circular_insula_sup",
            "S_circular_insula_inf",
            "Lat_Fis-ant-Horizont",
            "Lat_Fis-ant-Vertical",
            "Lat_Fis-post",
        ],

        "Frontoparietal": [
            "G_front_middle",
            "G_front_sup",
            "S_front_middle",
            "S_front_sup",
            "G_parietal_sup",
        ],

        "DefaultMode": [
            "G_precuneus",
            "G_cingul-Post-dorsal",
            "G_cingul-Post-ventral",
            "G_and_S_cingul-Mid-Post",
            "S_pericallosal",
            "S_cingul-Marginalis",
            "G_temporal_middle",
            "G_temporal_inf",
        ],

        "Limbic": [
            "G_orbital",
            "G_rectus",
            "G_subcallosal",
            "Pole_temporal",
            "G_oc-temp_med-Parahip",
            "G_and_S_cingul-Ant",
            "G_and_S_cingul-Mid-Ant",
            "S_orbital_med-olfact",
            "S_suborbital",
        ]
    }

)

# 7_figure_displays.ipynb
display_config = SimpleNamespace(

    # Function configs
    plot_brain_pairs = plot_brain_pairs,

    # Path configs
    postprocessed_data_path = postprocessed_data_path,
    figure_path = '/mnt/md0/tempFolder/samAnderson/unet-gnn/figures/'
    
)