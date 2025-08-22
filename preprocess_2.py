import numpy as np
import pandas as pd
import time
import os
import gc
from sklearn.preprocessing import StandardScaler

# --- Configuration ---
WINDOW_SIZE = 10
TOP_N_IDS = 20

def safe_hex_convert(hex_string):
    """Safely converts a hexadecimal string to an integer, handling errors."""
    try:
        return int(str(hex_string).strip(), 16)
    except (ValueError, TypeError):
        return np.nan

def safe_byte_convert(byte_string):
    """Safely converts a byte string (as hex) to an integer."""
    try:
        if isinstance(byte_string, str) and byte_string.strip():
            return int(byte_string.strip(), 16)
        # As per the paper, fill missing values with -1 instead of 0
        elif pd.isna(byte_string):
            return -1
        else:
            return -1
    except (ValueError, TypeError):
        return -1

def calculate_paper_features_for_group(group):
    """
    Calculates the novel byte-level features from the research paper for a single ID group.
    These are: Byte Flip Rate, Byte-Level Change Rate, and Byte-Level Distinct Value Rate.
    """
    # Initialize state trackers
    n = 0
    prev_bytes = {}
    distinct_values = {f'byte_{i}': set() for i in range(8)}
    cumulative_flips = {f'byte_{i}': 0 for i in range(8)}
    cumulative_bit_changes = {f'byte_{i}': 0.0 for i in range(8)}

    # Initialize lists to store results
    results = {
        'bfr': {f'bfr_{i}': [] for i in range(8)}, # Byte Flip Rate
        'bcr': {f'bcr_{i}': [] for i in range(8)}, # Byte-Level Change Rate
        'dvr': {f'dvr_{i}': [] for i in range(8)}  # Distinct Value Rate
    }

    byte_cols = [f'byte_{i}' for i in range(8)]

    for _, row in group.iterrows():
        n += 1
        for i, col in enumerate(byte_cols):
            current_val = int(row[col])
            # Skip calculations for filled values (-1)
            if current_val == -1:
                results['bfr'][f'bfr_{i}'].append(0)
                results['bcr'][f'bcr_{i}'].append(0)
                results['dvr'][f'dvr_{i}'].append(0)
                continue

            # --- Distinct Value Rate (DVR) ---
            distinct_values[col].add(current_val)
            dvr = len(distinct_values[col]) / 256.0
            results['dvr'][f'dvr_{i}'].append(dvr)

            # --- Flip Rate (BFR) & Change Rate (BCR) ---
            if n > 1 and col in prev_bytes:
                prev_val = prev_bytes[col]
                # BFR Calculation
                if current_val != prev_val:
                    cumulative_flips[col] += 1
                # BCR Calculation (Hamming distance)
                bit_diff = bin(current_val ^ prev_val).count('1')
                cb_in = bit_diff / 8.0
                cumulative_bit_changes[col] += cb_in
            
            bfr = cumulative_flips[col] / n if n > 0 else 0
            bcr = cumulative_bit_changes[col] / n if n > 0 else 0
            results['bfr'][f'bfr_{i}'].append(bfr)
            results['bcr'][f'bcr_{i}'].append(bcr)

            # Update previous value
            prev_bytes[col] = current_val

    # Assign new feature columns to the group
    for i in range(8):
        group[f'bfr_{i}'] = results['bfr'][f'bfr_{i}']
        group[f'bcr_{i}'] = results['bcr'][f'bcr_{i}']
        group[f'dvr_{i}'] = results['dvr'][f'dvr_{i}']

    return group

def preprocess_and_feature_engineer(df):
    """Applies preprocessing and feature engineering to the raw CAN bus DataFrame."""
    print("Starting preprocessing and feature engineering...")
    start_time = time.time()
    initial_rows = len(df)
    print(f"Initial rows: {initial_rows}")

    # 1. Convert Arbitration_ID (Hex to Integer)
    print("Converting Arbitration ID...")
    df['Arbitration_ID'] = df['Arbitration_ID'].apply(safe_hex_convert)
    df.dropna(subset=['Arbitration_ID'], inplace=True)
    df['Arbitration_ID'] = df['Arbitration_ID'].astype(int)

    # 2. Handle Timestamp and Calculate Time Difference
    print("Processing Timestamps and Time Differences...")
    df['Timestamp'] = pd.to_numeric(df['Timestamp'], errors='coerce')
    df.dropna(subset=['Timestamp'], inplace=True)
    df.sort_values(by=['Arbitration_ID', 'Timestamp'], inplace=True) # Sort by ID then Time

    df['IPT'] = df['Timestamp'].diff().fillna(0)
    df['TSLSP'] = df.groupby('Arbitration_ID')['Timestamp'].diff().fillna(0)

    # 3. Process Data Bytes
    print("Extracting and Converting Data Bytes...")
    df['Data'] = df['Data'].astype(str).str.strip()
    byte_cols_raw = df['Data'].str.split(r'\s+', expand=True, n=8).iloc[:, :8]
    for i in range(8):
        col_name = f'byte_{i}'
        df[col_name] = byte_cols_raw.get(i, pd.Series(index=df.index)).apply(safe_byte_convert)
        df[col_name] = df[col_name].astype(np.int16) # Use a larger int type for -1

    # 4. NEW: Calculate Byte-Level Features from Paper
    print("Calculating novel byte-level features (BFR, BCR, DVR)...")
    df = df.groupby('Arbitration_ID', group_keys=False).apply(calculate_paper_features_for_group)

    # 5. Calculate DLC and Mismatch (using original byte strings)
    print("Calculating DLC Mismatch...")
    df['Actual_DLC'] = byte_cols_raw.notna().sum(axis=1)
    df['DLC'] = pd.to_numeric(df['DLC'], errors='coerce').fillna(0).astype(int)
    df['DLC_Mismatch'] = (df['DLC'] != df['Actual_DLC']).astype(np.int8)

    # 6. Create Attack SubClass Flags
    print("Creating Attack SubClass Flags...")
    attack_types = ['Flooding', 'Fuzzing', 'Replay', 'Spoofing']
    df['SubClass'] = df['SubClass'].astype(str)
    for attack in attack_types:
        df[f'SubClass_{attack}'] = (df['SubClass'] == attack).astype(bool)
    attack_flag_cols = [f'SubClass_{a}' for a in attack_types]
    df['SubClass_Normal'] = ~df[attack_flag_cols].any(axis=1)
    
    # 7. Create Final Label Column
    print("Creating final labels...")
    attack_type_cols = ['SubClass_Flooding', 'SubClass_Fuzzing', 'SubClass_Replay', 'SubClass_Spoofing']
    df['Label'] = df[attack_type_cols].idxmax(axis=1).str.replace('SubClass_', '')
    df['Label'] = df['Label'].where(~df['SubClass_Normal'], 'Normal')

    # 8. Drop Redundant Columns
    print("Dropping intermediate columns...")
    columns_to_drop = [
        'Timestamp', 'Data', 'Class', 'SubClass', 'Actual_DLC',
        'SubClass_Flooding', 'SubClass_Fuzzing', 'SubClass_Replay',
        'SubClass_Spoofing', 'SubClass_Normal'
    ]
    # Drop original byte columns as they are now replaced by the new features
    columns_to_drop.extend([f'byte_{i}' for i in range(8)])
    df.drop(columns=columns_to_drop, inplace=True, errors='ignore')

    final_rows = len(df)
    print(f"Preprocessing finished in {time.time() - start_time:.2f} seconds.")
    print(f"Final rows: {final_rows} (Dropped {initial_rows - final_rows} total rows)")
    return df

def main():
    """Main function to load, process, and save CAN bus data."""
    print("--- DATA PREPROCESSING AND FEATURE ENGINEERING ---\n")
    
    input_file = 'Pre_train_S_1.csv'
    output_file = 'preprocessed_paper_features_can_data.csv'

    try:
        if not os.path.exists(input_file):
            print(f"Error: Make sure '{input_file}' is in the directory.")
            return

        print(f"Loading {input_file}...")
        initial_df = pd.read_csv(input_file)
        
        processed_df = preprocess_and_feature_engineer(initial_df)
        del initial_df
        gc.collect()
        
        print(f"\nFiltering for Top {TOP_N_IDS} most frequent Arbitration IDs...")
        top_ids = processed_df['Arbitration_ID'].value_counts().head(TOP_N_IDS).index
        processed_df = processed_df[processed_df['Arbitration_ID'].isin(top_ids)].copy()

        print("\nScaling numeric features using StandardScaler...")
        numeric_cols = processed_df.select_dtypes(include=np.number).columns.tolist()
        if 'Arbitration_ID' in numeric_cols:
            numeric_cols.remove('Arbitration_ID')
        
        scaler = StandardScaler()
        if numeric_cols:
             processed_df[numeric_cols] = scaler.fit_transform(processed_df[numeric_cols])
        print("Scaling complete.")

        print(f"\nSaving final processed data to '{output_file}'...")
        processed_df.to_csv(output_file, index=False)
        print("Data saved successfully.")

        print("\n--- Final Dataframe Info ---")
        processed_df.info()

        print("\nPreprocess and feature engineering complete.")

    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
