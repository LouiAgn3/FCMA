# ==============================================================================
# preprocess_nbaiot.py
# Preprocesses the N-BaIoT dataset for Fed-CMA experiments
#
# The N-BaIoT dataset comes as separate CSVs per device and per traffic type.
# This script:
#   1. Walks the directory structure to find all CSVs
#   2. Infers device_id, attack family (Mirai/Bashlite), and attack subtype
#      from the folder/file names
#   3. Combines everything into a single DataFrame with columns:
#      - 115 original features (already statistical, no further FE needed)
#      - device_id: which IoT device generated this traffic
#      - device_name: human-readable device name
#      - Label: 'benign' or attack subtype string (e.g. 'mirai_ack', 'gafgyt_combo')
#      - BinaryLabel: 0 = benign, 1 = attack
#   4. Handles NaN/Inf values
#   5. Saves to a single CSV for the FCMA pipeline
#
# Usage:
#   python preprocess_nbaiot.py --data_dir ./data --output nbaiot_processed.csv
#
# Expected directory layout (UCI download):
#   data/
#     <DeviceName>/
#       benign_traffic.csv
#       mirai_attacks/
#         ack.csv, scan.csv, syn.csv, udp.csv, udpplain.csv
#       gafgyt_attacks/           (or bashlite_attacks/)
#         combo.csv, junk.csv, scan.csv, tcp.csv, udp.csv
#
# Some versions from Kaggle use slightly different naming - the script handles both.
# ==============================================================================

import os
import sys
import argparse
import numpy as np
import pandas as pd
import time
import gc

# ---- Device metadata for context similarity ----
# The 9 devices in N-BaIoT with their types and which botnets targeted them
DEVICE_METADATA = {
    'Danmini_Doorbell':                      {'type': 'Doorbell',        'botnets': ['mirai', 'gafgyt']},
    'Ecobee_Thermostat':                     {'type': 'Thermostat',      'botnets': ['mirai', 'gafgyt']},
    'Ennio_Doorbell':                        {'type': 'Doorbell',        'botnets': ['gafgyt']},
    'Philips_B120N10_Baby_Monitor':          {'type': 'Baby_Monitor',    'botnets': ['mirai', 'gafgyt']},
    'Provision_PT_737E_Security_Camera':     {'type': 'Security_Camera', 'botnets': ['mirai', 'gafgyt']},
    'Provision_PT_838_Security_Camera':      {'type': 'Security_Camera', 'botnets': ['mirai', 'gafgyt']},
    'Samsung_SNH_1011_N_Webcam':             {'type': 'Webcam',          'botnets': ['gafgyt']},
    'SimpleHome_XCS7_1002_WHT_Security_Camera': {'type': 'Security_Camera', 'botnets': ['mirai', 'gafgyt']},
    'SimpleHome_XCS7_1003_WHT_Security_Camera': {'type': 'Security_Camera', 'botnets': ['mirai', 'gafgyt']},
}


def find_csv_files(data_dir):
    """
    Walk the directory tree and classify each CSV by device and traffic type.
    Returns a list of dicts: {path, device_name, device_id, label, attack_family, attack_subtype}
    """
    csv_files = []
    device_dirs = sorted([d for d in os.listdir(data_dir)
                          if os.path.isdir(os.path.join(data_dir, d))])

    if not device_dirs:
        # Maybe CSVs are directly in data_dir with device names in filenames
        print("No device subdirectories found, looking for CSVs directly...")
        for f in sorted(os.listdir(data_dir)):
            if f.endswith('.csv'):
                csv_files.append({
                    'path': os.path.join(data_dir, f),
                    'device_name': 'Unknown',
                    'device_id': 0,
                    'label': 'benign' if 'benign' in f.lower() else f.replace('.csv', ''),
                    'attack_family': 'none',
                    'attack_subtype': 'none',
                })
        return csv_files

    for dev_idx, device_dir in enumerate(device_dirs):
        device_path = os.path.join(data_dir, device_dir)
        device_name = device_dir

        # Walk through all files in this device's directory
        for root, dirs, files in os.walk(device_path):
            for filename in sorted(files):
                if not filename.endswith('.csv'):
                    continue

                filepath = os.path.join(root, filename)
                rel_path = os.path.relpath(root, device_path).lower()
                fname_lower = filename.lower().replace('.csv', '')

                # Determine label from path/filename
                if 'benign' in fname_lower or 'benign' in rel_path:
                    label = 'benign'
                    attack_family = 'none'
                    attack_subtype = 'none'
                elif 'mirai' in rel_path or 'mirai' in fname_lower:
                    attack_family = 'mirai'
                    # Extract subtype: ack, scan, syn, udp, udpplain
                    attack_subtype = fname_lower.replace('mirai_', '').replace('mirai', '')
                    if not attack_subtype:
                        attack_subtype = 'unknown'
                    label = f'mirai_{attack_subtype}'
                elif 'gafgyt' in rel_path or 'bashlite' in rel_path or \
                     'gafgyt' in fname_lower or 'bashlite' in fname_lower:
                    attack_family = 'gafgyt'
                    attack_subtype = fname_lower.replace('gafgyt_', '').replace('bashlite_', '') \
                                                 .replace('gafgyt', '').replace('bashlite', '')
                    if not attack_subtype:
                        attack_subtype = 'unknown'
                    label = f'gafgyt_{attack_subtype}'
                else:
                    # Try to infer from filename
                    label = fname_lower
                    attack_family = 'unknown'
                    attack_subtype = fname_lower

                csv_files.append({
                    'path': filepath,
                    'device_name': device_name,
                    'device_id': dev_idx,
                    'label': label,
                    'attack_family': attack_family,
                    'attack_subtype': attack_subtype,
                })

    return csv_files


def load_and_combine(csv_files, sample_cap=None):
    """
    Load all CSVs and combine into a single DataFrame.
    Adds device_id, device_name, Label (attack subtype), BinaryLabel (0/1).
    Optional: cap samples per file to manage memory.
    """
    all_dfs = []
    total_files = len(csv_files)

    print(f"\nLoading {total_files} CSV files...")
    for idx, info in enumerate(csv_files):
        try:
            df = pd.read_csv(info['path'])
            if sample_cap and len(df) > sample_cap:
                df = df.sample(n=sample_cap, random_state=42)

            df['device_id'] = info['device_id']
            df['device_name'] = info['device_name']
            df['Label'] = info['label']
            df['BinaryLabel'] = 0 if info['label'] == 'benign' else 1
            df['attack_family'] = info['attack_family']

            all_dfs.append(df)

            if (idx + 1) % 10 == 0 or idx == total_files - 1:
                print(f"  Loaded {idx+1}/{total_files}: {info['device_name']}/{info['label']} "
                      f"({len(df)} rows)")
        except Exception as e:
            print(f"  ERROR loading {info['path']}: {e}")
            continue

    if not all_dfs:
        raise ValueError("No data loaded! Check your data directory.")

    print("\nCombining all files...")
    combined = pd.concat(all_dfs, ignore_index=True)
    del all_dfs
    gc.collect()

    return combined


def clean_features(df):
    """
    Clean the 115 numeric features:
    - Replace inf with NaN
    - Fill NaN with column median
    - Clip extreme outliers (beyond 5 std devs)
    """
    print("Cleaning features...")

    # Identify the 115 feature columns (everything except metadata)
    meta_cols = ['device_id', 'device_name', 'Label', 'BinaryLabel', 'attack_family']
    feature_cols = [c for c in df.columns if c not in meta_cols]
    print(f"  {len(feature_cols)} feature columns detected")

    # Replace inf
    df[feature_cols] = df[feature_cols].replace([np.inf, -np.inf], np.nan)

    # Fill NaN with column median
    nan_counts = df[feature_cols].isna().sum()
    nan_total = nan_counts.sum()
    if nan_total > 0:
        print(f"  Filling {nan_total} NaN values with column medians")
        medians = df[feature_cols].median()
        df[feature_cols] = df[feature_cols].fillna(medians)

    # Clip extreme outliers
    for col in feature_cols:
        mean = df[col].mean()
        std = df[col].std()
        if std > 0:
            df[col] = df[col].clip(mean - 5*std, mean + 5*std)

    return df, feature_cols


def main():
    parser = argparse.ArgumentParser(description='Preprocess N-BaIoT dataset for Fed-CMA')
    parser.add_argument('--data_dir', type=str, default='./data',
                        help='Path to N-BaIoT data directory')
    parser.add_argument('--output', type=str, default='nbaiot_processed.csv',
                        help='Output CSV filename')
    parser.add_argument('--sample_cap', type=int, default=None,
                        help='Max samples per CSV file (for memory). None = use all.')
    args = parser.parse_args()

    print("=" * 60)
    print("  N-BaIoT Dataset Preprocessor for Fed-CMA")
    print("=" * 60)

    start = time.time()

    # Step 1: Find all CSVs
    csv_files = find_csv_files(args.data_dir)
    if not csv_files:
        print(f"ERROR: No CSV files found in '{args.data_dir}'")
        print("Expected structure: data/<DeviceName>/benign_traffic.csv etc.")
        sys.exit(1)

    print(f"\nFound {len(csv_files)} CSV files across "
          f"{len(set(f['device_name'] for f in csv_files))} devices:")
    for dev in sorted(set(f['device_name'] for f in csv_files)):
        dev_files = [f for f in csv_files if f['device_name'] == dev]
        labels = [f['label'] for f in dev_files]
        print(f"  {dev}: {len(dev_files)} files ({', '.join(labels[:3])}{'...' if len(labels) > 3 else ''})")

    # Step 2: Load and combine
    combined = load_and_combine(csv_files, sample_cap=args.sample_cap)
    print(f"\nCombined dataset: {len(combined)} rows × {combined.shape[1]} columns")

    # Step 3: Clean features
    combined, feature_cols = clean_features(combined)

    # Step 4: Report statistics
    print(f"\n--- Dataset Summary ---")
    print(f"Total samples: {len(combined):,}")
    print(f"Features: {len(feature_cols)}")
    print(f"Devices: {combined['device_id'].nunique()}")
    print(f"\nPer-device breakdown:")
    for dev_id in sorted(combined['device_id'].unique()):
        dev_data = combined[combined['device_id'] == dev_id]
        dev_name = dev_data['device_name'].iloc[0]
        n_benign = (dev_data['BinaryLabel'] == 0).sum()
        n_attack = (dev_data['BinaryLabel'] == 1).sum()
        attack_types = sorted(dev_data.loc[dev_data['BinaryLabel'] == 1, 'Label'].unique())
        print(f"  Device {dev_id} ({dev_name}): {len(dev_data):,} total | "
              f"Benign: {n_benign:,} | Attack: {n_attack:,}")
        print(f"    Attack types: {', '.join(attack_types) if attack_types else 'None'}")

    print(f"\nOverall class balance:")
    print(f"  Benign: {(combined['BinaryLabel'] == 0).sum():,} "
          f"({(combined['BinaryLabel'] == 0).mean()*100:.1f}%)")
    print(f"  Attack: {(combined['BinaryLabel'] == 1).sum():,} "
          f"({(combined['BinaryLabel'] == 1).mean()*100:.1f}%)")

    # Step 5: Save
    print(f"\nSaving to '{args.output}'...")
    combined.to_csv(args.output, index=False)
    print(f"Saved successfully ({os.path.getsize(args.output) / 1024**2:.1f} MB)")

    print(f"\nTotal preprocessing time: {time.time() - start:.1f}s")
    print("\nTo run Fed-CMA experiment:")
    print(f"  python FCMA_NBaIoT.py --data {args.output}")


if __name__ == '__main__':
    main()
