# Standard libraries
import os
import sys
import glob
import subprocess

# External libraries
import numpy as np
import pandas as pd
import nibabel as nib
from tqdm import tqdm

# Add custom site-packages (if needed)
sys.path.append('/home/samuelA/.local/lib/python3.10/site-packages')
sys.path.append('/mnt/md0/tempFolder/samAnderson/gnn_model/unet-gnn/')
sys.path.append('/mnt/md0/tempFolder/samAnderson/gnn_model/unet-gnn/functions/')

# Import config for consistency
from config import preprocessing_config as cfg

def make_fsaverage(input_path, output_path, file_type, ico=7, dataset='ukbb'):
    '''
    Convert a FreeSurfer file to fsaverage space (ico-7) with dataset-specific output naming.

    param input_path: full path to the FreeSurfer file
    param output_path: directory to store converted file
    param file_type: type of the file to validate (e.g., 'thickness.mgh')
    param ico: target icosahedral order (default=7)
    param dataset: name of the dataset ('ukbb', 'ixi', 'nacc', 'slim', 'adni')

    return: absolute path to converted file if successful, None otherwise
    '''

    subj_dir = -3  # path position of subject directory
    freesurfer_dir = '/'.join(input_path.split('/')[:subj_dir])
    set_dir = f"export SUBJECTS_DIR='{freesurfer_dir}'; "
    surf = 'mri_surf2surf'

    subject = ' --s ' + (input_path.split('/')[subj_dir]) + ' '
    hemi = ' --hemi ' + input_path.split('/')[-1][:2]
    srcsurfval = f' --srcsurfval {input_path} '
    target_subj = '--trgsubject ico '
    trig_order = f'--trgicoorder {ico} '
    log_output = f' > {output_path}last_conversion.log'

    # === Determine output file name based on dataset logic ===
    hemi_code = hemi[-2:]
    if 'w-g.pct.mgh' in input_path:
        if dataset == 'nacc':
            output_fname_abs = f'{output_path}{input_path.split("/")[subj_dir-3]}_{input_path.split("/")[subj_dir-1]}_{hemi_code}.w-g.pct.mgh'
        else:
            output_fname_abs = f'{output_path}{input_path.split("/")[subj_dir-2]}_{input_path.split("/")[subj_dir]}_{hemi_code}.w-g.pct.mgh'
    else:
        extension = input_path.split(".")[-1]
        if dataset == 'nacc':
            output_fname_abs = f'{output_path}{input_path.split("/")[subj_dir-3]}_{input_path.split("/")[subj_dir-1]}_{hemi_code}_{extension}.mgh'
        else:
            output_fname_abs = f'{output_path}{input_path.split("/")[subj_dir-2]}_{input_path.split("/")[subj_dir]}_{hemi_code}_{extension}.mgh'

    # === Special case for IXI output filename ===
    if dataset == 'ixi':
        id_start = output_fname_abs.find("all/IXI") + 7
        underscore_pos = output_fname_abs.find("_", id_start)
        output_fname_abs = f'{output_fname_abs[:id_start-3]}{output_fname_abs[id_start:id_start+3]}{output_fname_abs[underscore_pos:]}'

    # === Input validation ===
    if not validate_inputs(input_path, file_type[:-3]):
        if os.path.exists(output_fname_abs):
            os.remove(output_fname_abs)
        return None

    # === Construct and execute command ===
    output_fname_cmd = ' --o ' + output_fname_abs
    shell_command = set_dir + surf + subject + target_subj + trig_order + hemi + srcsurfval + output_fname_cmd + log_output
    result = subprocess.run(shell_command, shell=True, executable='/bin/bash')

    if result.returncode == 0 and os.path.isfile(output_fname_abs):
        return output_fname_abs
    else:
        if os.path.exists(output_fname_abs):
            os.remove(output_fname_abs)
        return None

def validate_inputs(input_path, file_type):

    file_type_ranges = {
        'area': {'min': 0, 'max': 100},
        'curv': {'min': -40, 'max': 40},
        'w-g.pct.mgh': {'min': -200, 'max': 200},
        'sulc': {'min': -20, 'max': 20},
        'thickness': {'min': 0, 'max': 10}
    }
    
    # Verify that the file values are within the expected range
    min_val = file_type_ranges[file_type]['min']
    max_val = file_type_ranges[file_type]['max']
    
    try:
        # For MGH/MGZ files
        if file_type.endswith('.mgh') or file_type == 'w-g.pct.mgh':
            img = nib.load(input_path)
            data = img.get_fdata()

        # For FreeSurfer curvature files
        elif file_type in ['curv', 'area', 'sulc', 'thickness']:
            data = nib.freesurfer.read_morph_data(input_path)
        
        # For surface geometry files
        elif file_type in ['white', 'pial', 'inflated']:
            verts, faces = nib.freesurfer.read_geometry(input_path)
            data = verts  # Validate vertex coordinates

        else:
            print(f"Unsupported file type: {file_type}")
    
    except KeyError:
        return False

    # Verify within expected range
    if np.any(data < min_val) or np.any(data > max_val):
        print(f"Value out of range in {input_path}")
        print(f"min_val: {np.min(data)}\n max_val: {np.max(data)}")
        return False
    
    return True

def make_data(root_path='/mnt/md0/tempFolder/samAnderson/', file_types=['w-g.pct.mgh', 'curv', 'thickness'], # used to be ratio so older sorts are incorrect
             data_path='lab/lab_organized/subject-*/freesurfer_output/*/',
             output_dir=None,
             damaged_subjects = ['None'], ico=7,
             dataset='ukbb'):
    
    '''
    function for projecting surf files to ico-7
    returns dictionary
    '''

    # alphabetize file types for consistency across datasets
    file_types = sorted(file_types)

    # create a dictionary with the subject paths
    all_paths = {}
    
    for f in file_types:

        rh_temp = glob.glob(root_path + data_path + f'surf/rh.{f}') # right hemisphere
        lh_temp = glob.glob(root_path + data_path + f'surf/lh.{f}') # left hemisphere

        # remove damaged subjects (where not all files are available)
        for subj in damaged_subjects:
            rh_temp = [x for x in rh_temp if subj not in x]
            lh_temp = [x for x in lh_temp if subj not in x]

        all_paths[f'{f}_rh'] = sorted(rh_temp) # so that all of the paths align
        all_paths[f'{f}_lh'] = sorted(lh_temp)

    # set the output directory, then either make or load in atlas-projected subjects
    if output_dir == None:
        output_dir = root_path + f'gnn_model/datasets/{data_path.split("/")[0]}/'  

    # create the output directory
    try: os.mkdir(output_dir)
    except FileExistsError: pass

    # get the data from each of the paths, or convert the data and get the location data
    damaged_subjects = []
    for file_type in tqdm(all_paths, desc="Converting file types.."): # rh.curv, lh.curv, etc.
        for file in all_paths[file_type]: # each individual subject/timepoint combination
            damaged_subjects.append(make_fsaverage(file, output_dir, file_type, ico, dataset))

    return [item for item in damaged_subjects if item is not None] # return the damaged subjects

def make_npy(dataset_dir, 
              metadata_path, 
              file_suffixes = ['_lh_curv.mgh', '_rh_curv.mgh', '_lh_thickness.mgh', '_rh_thickness.mgh', '_lh.w-g.pct.mgh', '_rh.w-g.pct.mgh'],
              age_id_date_sex=[],
              first_hemi='rh',
              dataset=None,
              sex_mapping={'Male' : 'Male', 'Female' :'Female'}):
    '''
    This function is intended to load the processed files
    get_data does technically work aswell but it can overload the system, and takes a while

    param dataset_dir: the path containing the processed files, str
    param metadata_path: the path to the metadata, must have subject id and age
    param file_suffixes: a list of
    all of the file types we want each valid subject to have
    param age_id_date: the column titles for the age, id, and date
    param massive: whether to split the data into 100ths, and save the resulting arrays as pickle files 
    param chunk_path: where to output chunk files
    param first_hemi: which hemi should be first in the resulting array, i.e. is node 0 right or left?

    return: X, y
    '''

    # Find which subjects were not completely processed
    subject_files = {}
    
    for suffix in file_suffixes:
        files = glob.glob(os.path.join(dataset_dir, f'*{suffix}'))
        subject_ids = {os.path.basename(f).replace(suffix, '') for f in files}
        subject_files[suffix] = subject_ids

    # Find all unique subject IDs
    all_subject_ids = set()
    for ids in subject_files.values():
        all_subject_ids.update(ids)

    # Find subject IDs that are missing from the complete set
    subjects_without_all_files = [subject for subject in all_subject_ids if not all(subject in subject_files[suffix] for suffix in file_suffixes)]
    #print(f"Subjects without all files: {subjects_without_all_files}")

    # Alphabetize the file suffixes, ignoring hemisphere for consistency
    file_suffixes = sorted(file_suffixes, key=lambda x: x[4:])
    
    # Load in the demographic info / metadata
    if '.xlsx' in metadata_path:
        metadata_csv = pd.read_excel(metadata_path, dtype=str)
    elif 'csv' in metadata_path:
        metadata_csv = pd.read_csv(metadata_path, dtype=str)
    else:
        raise Exception('Error: Invalid metadata file type')
    age_column, id_column, date_column, sex_column = age_id_date_sex
    
    # Apply changes specific to each dataset
    if dataset == 'IXI':

        # convert x/y/zz to ZZZZYYXX assuming it refers to 20XX for the date
        metadata_csv[date_column] = metadata_csv[date_column].apply(
        lambda x: (
            f"20{x.split('/')[2].zfill(2)}"
            f"{x.split('/')[0].zfill(2)}"
            f"{x.split('/')[1].zfill(2)}"
        ) if pd.notna(x) else None )

    elif dataset == 'NACC':

        # remove the .0 from the dates
        metadata_csv[date_column] = metadata_csv[date_column].astype(str).str.replace(r"\.0$", "", regex=True)

        # remove non-CNs
        subjects_without_all_files.extend(metadata_csv[metadata_csv['TBI_status'] != 'Control'].apply(lambda row: f"{row[id_column]}_{row[date_column]}", axis=1).tolist())

    elif 'ADNI' in dataset:
        
        # convert the dates format
        metadata_csv[date_column] = pd.to_datetime(metadata_csv[date_column], format='%m/%d/%Y').dt.strftime('%Y%m%d')

        if dataset[-2:] == 'CN':
            # remove non-CNs
            subjects_without_all_files.extend(metadata_csv[metadata_csv['Research Group'] != 'CN'].apply(lambda row: f"{row[id_column]}_{row[date_column]}", axis=1).tolist())
        
        elif dataset[-3:] == 'MCI':
            subjects_without_all_files.extend(
                metadata_csv[~metadata_csv['Research Group'].str.contains('MCI', na=False)]
                .apply(lambda row: f"{row[id_column]}_{row[date_column]}", axis=1)
                .tolist())

        elif dataset[-2:] == 'AD':
            # remove non-ADs
            subjects_without_all_files.extend(metadata_csv[metadata_csv['Research Group'] != 'AD'].apply(lambda row: f"{row[id_column]}_{row[date_column]}", axis=1).tolist())            

        else: raise Exception

    elif 'OASIS4' in dataset:

        if dataset[-2:] == 'CN':
            pass

        else: # AD
            pass

    elif 'UKBB' in dataset: pass
    else: pass

    # Get subjects
    if dataset_dir[-1] != '/': dataset_dir.append('/')
    all_files = glob.glob(f'{dataset_dir}*{file_suffixes[0]}') # [0] is arbitrary, to get subjects, make sure all
    subject_ids = [os.path.basename(f).replace(file_suffixes[0], '') for f in all_files]

    # Remove damaged subjects
    try: subject_ids = [s for s in subject_ids if s not in subjects_without_all_files]
    except ValueError: print('Value Error')

    # Get the number of nodes
    sample_file = nib.load(all_files[0]).get_fdata()
    num_nodes = sample_file.shape[0]

    num_subjects = len(subject_ids)
    num_features = len(file_suffixes)
    
    # Initialize the main data array
    X = np.zeros((num_subjects, (num_nodes*2), (num_features//2))) # to account for the hemispheres being independent nodes
    
    # initialize the empty lists
    y = []
    sex = []
    subjects = []

    # save the damaged subjects (that have nan in them)
    damaged_subjects = []

    # for each subject
    for subj_idx, subject_id in enumerate(subject_ids):
        
        ####
        # Update y
        ####

        # add the age to the corresponding list
        if date_column == None: age_df = metadata_csv[[age_column, id_column, sex_column]]
        else: age_df = metadata_csv[[age_column, id_column, date_column, sex_column]]

        # get the working file's date and subject id
        target_id, target_date = subject_id.rsplit('_', 1)

        # get the desired row
        if date_column == None: age_value = age_df[(age_df[id_column] == target_id)]
        else: age_value = age_df[(age_df[id_column] == target_id) & (age_df[date_column] == target_date)]
        try:
            if 'Y' in age_value[age_column].values[0]: # from '64Y' format if relevant
                as_float = float(age_value[age_column].values[0][1:-1])
            else:
                as_float = float(age_value[age_column].values[0])

            # add the age value
            y.append(as_float) # from '64Y' format
            # add the sex value
            sex_value = age_value[sex_column].values[0]
            sex.append(sex_mapping[sex_value])

        except IndexError: # if there isn't an associated age value, we don't consider this subject
            #print(target_id)
            #print(target_date)
            # error is here
            continue

        subjects.append(f'{target_id}_{target_date}')

        ####
        # Update X
        ####

        # for each file type
        for suffix_idx, suffix in enumerate(file_suffixes):

            # get the path of the target file
            file_path = os.path.join(dataset_dir, f'{subject_id}{suffix}')

            # load in the data
            data = nib.load(file_path).get_fdata().flatten() # flatten: from (163842, 1, 1) to (163842)
            # if referencing the left hemisphere, i.e. the earlier nodes

            if first_hemi in suffix:
                X[subj_idx, :(X.shape[1]//2), suffix_idx//2] = data
            else:
                X[subj_idx, (X.shape[1]//2):, suffix_idx//2] = data

    # make y an numpy array
    y = np.array(y)
    sex = np.array(sex)

    # get rid of the excess 0s due to skipped subjects
    non_zero_subjects = np.any(X != 0, axis=(1, 2))
    X = X[non_zero_subjects]

    return X, y, sex, damaged_subjects, subjects

# ============================================== Preprocessing Pipeline (cfg-driven) ========================================================= #

processed_dir = cfg.processed_data_path
raw_data_dir = cfg.raw_data_path

# ===================== Build suffixes from cfg ===================== #

file_types = [cfg.feature_map_dict[f] for f in cfg.features]
hemispheres = ['_lh', '_rh']

file_suffixes = [
    f"{hemi}.{feature}" if feature == "w-g.pct.mgh" else f"{hemi}_{feature}.mgh"
    for feature in file_types
    for hemi in hemispheres
]

# ===================== Split datasets ===================== #

dataset_names = list(vars(cfg.datasets).keys())

train_datasets = [n for n in dataset_names if getattr(cfg.datasets, n).type == 'train']
test_datasets  = [n for n in dataset_names if getattr(cfg.datasets, n).type == 'test']

# ===================== Build Training Set ===================== #

X_list, y_list, sex_list, subjects_list = [], [], [], []

for dataset_name in train_datasets:

    dset = getattr(cfg.datasets, dataset_name)
    print(f'Processing {dataset_name}...')

    dataset_dir = f'{raw_data_dir}{dset.folder}'

    X_tmp, y_tmp, sex_tmp, damaged_subjects_tmp, subjects_tmp = make_npy(
        dataset_dir=dataset_dir,
        metadata_path=dset.metadata_path,
        file_suffixes=file_suffixes,
        age_id_date_sex=dset.age_id_date_sex,
        first_hemi=cfg.first,
        dataset=dataset_name,
        sex_mapping=dset.sex_mapping
    )

    # Save raw dataset
    save_dir = f'{processed_dir}{dataset_name}/'
    os.makedirs(save_dir, exist_ok=True)

    np.save(f'{save_dir}X_{dataset_name}_ico{cfg.ico}.npy', X_tmp)
    np.save(f'{save_dir}y_{dataset_name}_ico{cfg.ico}.npy', y_tmp)
    np.save(f'{save_dir}sex_{dataset_name}_ico{cfg.ico}.npy', sex_tmp)
    np.save(f'{save_dir}subj_{dataset_name}_ico{cfg.ico}.npy', subjects_tmp)

    # Accumulate training only
    X_list.append(X_tmp)
    y_list.append(y_tmp)
    sex_list.append(sex_tmp)
    subjects_list.append(subjects_tmp)

# Concatenate training
X_train = np.concatenate(X_list, axis=0)
y_train = np.concatenate(y_list)
sex_train = np.concatenate(sex_list)
subjects_train = np.concatenate(subjects_list)

del X_list, y_list, sex_list, subjects_list

# ===================== Compute stats ===================== #

mean = np.mean(X_train, axis=(0, 1), keepdims=True)
std = np.std(X_train, axis=(0, 1), keepdims=True)

X_train_standardized = (X_train - mean) / std

print(X_train_standardized.shape)
print(y_train.shape)

# ===================== Save training ===================== #

train_dir = f'{processed_dir}training/'
os.makedirs(train_dir, exist_ok=True)

np.save(f'{train_dir}X_train_ico{cfg.ico}.npy', X_train_standardized)
np.save(f'{train_dir}y_train_ico{cfg.ico}.npy', y_train)
np.save(f'{train_dir}sex_train_ico{cfg.ico}.npy', sex_train)
np.save(f'{train_dir}subj_train_ico{cfg.ico}.npy', subjects_train)
np.save(f'{train_dir}mean_train_ico{cfg.ico}.npy', mean)
np.save(f'{train_dir}std_train_ico{cfg.ico}.npy', std)

# Free memory before testing
del X_train, X_train_standardized, y_train, sex_train, subjects_train


# ===================== Process Testing (streamed) ===================== #

for dataset_name in test_datasets:

    dset = getattr(cfg.datasets, dataset_name)
    print(f'Processing {dataset_name}...')

    dataset_dir = f'{raw_data_dir}{dset.folder}'

    X_test, y_test, sex_test, damaged_subjects_test, subjects_test = make_npy(
        dataset_dir=dataset_dir,
        metadata_path=dset.metadata_path,
        file_suffixes=file_suffixes,
        age_id_date_sex=dset.age_id_date_sex,
        first_hemi=cfg.first,
        dataset=dataset_name,
        sex_mapping=dset.sex_mapping
    )

    # Normalize using training stats
    X_test_standardized = (X_test - mean) / std

    save_dir = f'{processed_dir}{dataset_name}/'
    os.makedirs(save_dir, exist_ok=True)

    np.save(f'{save_dir}X_{dataset_name}_ico{cfg.ico}_norm.npy', X_test_standardized)
    np.save(f'{save_dir}y_{dataset_name}_ico{cfg.ico}.npy', y_test)
    np.save(f'{save_dir}sex_{dataset_name}_ico{cfg.ico}.npy', sex_test)
    np.save(f'{save_dir}subj_{dataset_name}_ico{cfg.ico}.npy', subjects_test)

    # Free immediately
    del X_test, X_test_standardized, y_test, sex_test, subjects_test