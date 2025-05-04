# (Parallel Version - Adapted for Many Small Clusters with Top K Selection - USING MEME)
# Combines Motif Extraction/Validation pipeline for batched processing - PARALLELIZED.
"""
Example Command (Using Top K with MEME):
python top_k_meme_motif_parallel_pipeline.py \
    --fasta_dir ./Data/v1_cluster_windows_15mers \
    --validation_csv ./Data/validation_dataset.csv \
    --master_fasta ./Data/spider-silkome-database.v1.prot.fasta \
    --output_dir ./Results/v1_Top500_MEME_Run \
    --conda_env memesuite \
    --top_k_clusters 500 \
    --meme_nmotifs 5 \
    --meme_mod anr \
    --evalue_thresh 0.05 \
    --generate_plots \
    --num_workers 8

"""
import os
import subprocess
import re
import pandas as pd
import numpy as np
from scipy.stats import pearsonr, spearmanr
from statsmodels.stats.multitest import multipletests
import matplotlib.pyplot as plt
import argparse
import glob
import logging
import sys
from contextlib import redirect_stderr
from datetime import datetime
import concurrent.futures # For parallelism
import time # For timing
import operator # For sorting clusters by size

# --- Configuration ---
# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(processName)s - %(levelname)s - %(message)s', stream=sys.stdout)

# --- Function Definitions ---

def run_meme(fasta_file, output_dir, minw, maxw, mod, nmotifs, evt, conda_env_name):
    """Runs MEME on a given FASTA file using a specified conda environment."""
    base_name = os.path.splitext(os.path.basename(fasta_file))[0]
    # NOTE: output_dir here is the UNIQUE run directory already
    meme_output_subdir = os.path.join(output_dir, f"meme_out_{base_name}")
    # MEME uses -oc which behaves like mkdir -p
    # os.makedirs(meme_output_subdir, exist_ok=True) # Not strictly needed if using -oc

    cmd = [
        'conda', 'run', '-n', conda_env_name, '--no-capture-output',
        'meme',
        fasta_file, # Input file directly after 'meme'
        '-protein', # Specify protein sequences
        '-oc', meme_output_subdir, # Output directory (will be created)
        '-mod', str(mod), # Motif distribution model (oops, zoops, anr)
        '-nmotifs', str(nmotifs), # Max number of motifs to find
        '-minw', str(minw), # Minimum motif width
        '-maxw', str(maxw), # Maximum motif width
        '-evt', str(evt), # E-value threshold for reporting motifs
        '-nostatus', # Suppress progress status to stderr
        '-time', '1800', # Add timeout (e.g., 30 minutes)
        # '-p', '1' # Explicitly run sequentially within the worker process
                     # Avoids oversubscribing cores if num_workers is high.
                     # Remove if you want MEME to use multiple cores *within* a worker.
    ]

    logger = logging.getLogger() # Get root logger
    logger.info(f"[{base_name}] Running MEME for {fasta_file}...")
    # logger.debug(f"[{base_name}] Command: {' '.join(cmd)}") # Log full command only if debugging

    try:
        # Basic check for empty file before running
        if os.path.getsize(fasta_file) == 0:
            logger.warning(f"[{base_name}] Input FASTA file is empty. Skipping MEME.")
            return None

        result = subprocess.run(cmd, check=True, capture_output=True, text=True) # MEME timeout is within the command args

        # Check MEME's stderr for potential warnings even on success (exit code 0)
        if result.stderr:
             stderr_summary = (result.stderr[:500] + '...') if len(result.stderr) > 500 else result.stderr
             if "warning" in stderr_summary.lower(): # Basic check for warnings
                 logger.warning(f"[{base_name}] MEME stderr reported warnings:\n{stderr_summary}")
             # else: # Optionally log non-warning stderr if debugging
             #    logger.debug(f"[{base_name}] MEME stderr:\n{stderr_summary}")


        logger.info(f"[{base_name}] MEME completed successfully. Output in {meme_output_subdir}")
        # MEME primarily outputs HTML, but meme.txt contains parseable motif info
        meme_txt_path = os.path.join(meme_output_subdir, 'meme.txt')
        if os.path.exists(meme_txt_path):
            return meme_txt_path
        else:
            logger.error(f"[{base_name}] MEME ran successfully but meme.txt output file not found at {meme_txt_path}")
            return None

    except FileNotFoundError:
        logger.error(f"[{base_name}] Error: 'conda' command not found. Is Conda installed and in PATH?")
        raise # Reraise critical error
    # Timeout is handled by MEME itself via '-time' argument, less likely to hit subprocess timeout unless MEME hangs completely
    except subprocess.CalledProcessError as e:
        logger.error(f"[{base_name}] MEME failed with exit code {e.returncode}.")
        # Log full error output upon failure
        logger.error(f"[{base_name}] STDOUT:\n{e.stdout}")
        logger.error(f"[{base_name}] STDERR:\n{e.stderr}")
        return None # Indicate failure for this cluster
    except Exception as e:
        logger.error(f"[{base_name}] An unexpected error occurred during MEME execution: {e}")
        raise # Reraise unexpected errors

def extract_motifs(meme_txt_file, cluster_name):
    """Extracts unique motif consensus sequences from a meme.txt file."""
    logger = logging.getLogger()
    if not meme_txt_file or not os.path.exists(meme_txt_file):
        logger.warning(f"[{cluster_name}] MEME output file not found or not provided: {meme_txt_file}. Skipping motif extraction.")
        return []

    consensi = []
    try:
        with open(meme_txt_file, 'r') as fh:
            in_motif_section = False
            current_motif_lines = []
            for line in fh:
                # Find the start of the command line section which follows motifs
                if line.startswith("COMMAND LINE SUMMARY"):
                    break # Stop processing after motifs

                if line.startswith('MOTIF '):
                    # Process the previous motif block if one exists
                    if current_motif_lines:
                         # Try to find the consensus from MOTIF line itself (MEME 5 format)
                         parts = current_motif_lines[0].split() # First line is MOTIF line
                         if len(parts) > 1:
                              # Expecting "MOTIF <name>" or "MOTIF <name> <altname>"
                              # The actual consensus sequence might be on later lines (older MEME?)
                              # or inferred from the name (STREME like "MOTIF 1-CONS")
                              # Or often "MOTIF <name> width=... alength=... etc"
                              potential_consensus_name = parts[1]
                              # Check if it follows STREME "rank-consensus" format
                              if '-' in potential_consensus_name and potential_consensus_name.count('-') == 1:
                                  try:
                                      rank, consensus = potential_consensus_name.split('-', 1)
                                      int(rank)
                                      consensi.append(consensus)
                                      logger.debug(f"[{cluster_name}] Parsed consensus (STREME style): {consensus}")
                                  except ValueError:
                                      # Doesn't look like rank-consensus, treat name as placeholder
                                      # Look for consensus on subsequent lines
                                      pass
                              # Try to find consensus on subsequent "Multilevel consensus sequence" line (MEME 5+)
                              found_consensus_line = False
                              for motif_line in current_motif_lines[1:]: # Check lines after MOTIF line
                                  if "Multilevel consensus sequence" in motif_line:
                                       consensus = motif_line.split(":")[-1].strip()
                                       if consensus:
                                           consensi.append(consensus)
                                           logger.debug(f"[{cluster_name}] Parsed consensus (MEME multilevel): {consensus}")
                                           found_consensus_line = True
                                           break
                              if not found_consensus_line:
                                   logger.warning(f"[{cluster_name}] Could not determine consensus sequence for motif starting: {current_motif_lines[0].strip()}")


                         current_motif_lines = [line] # Start new motif block
                    else:
                         current_motif_lines = [line] # Start the first motif block

                elif current_motif_lines: # If we are inside a motif block
                    current_motif_lines.append(line)


            # Process the last motif block after loop finishes
            if current_motif_lines:
                parts = current_motif_lines[0].split()
                if len(parts) > 1:
                    potential_consensus_name = parts[1]
                    found_consensus_line = False
                    if '-' in potential_consensus_name and potential_consensus_name.count('-') == 1:
                        try:
                            rank, consensus = potential_consensus_name.split('-', 1)
                            int(rank)
                            consensi.append(consensus)
                            found_consensus_line = True
                            logger.debug(f"[{cluster_name}] Parsed consensus (STREME style): {consensus}")
                        except ValueError: pass # Fall through to check multilevel line
                    if not found_consensus_line:
                        for motif_line in current_motif_lines[1:]:
                            if "Multilevel consensus sequence" in motif_line:
                                consensus = motif_line.split(":")[-1].strip()
                                if consensus:
                                    consensi.append(consensus)
                                    logger.debug(f"[{cluster_name}] Parsed consensus (MEME multilevel): {consensus}")
                                    found_consensus_line = True
                                    break
                    if not found_consensus_line:
                         logger.warning(f"[{cluster_name}] Could not determine consensus sequence for final motif block starting: {current_motif_lines[0].strip()}")

        # Deduplicate while preserving order
        seen = set()
        unique_motifs = [c for c in consensi if c and not (c in seen or seen.add(c))] # Ensure c is not empty

        # Save the extracted unique motifs to a file in the same directory
        output_dir = os.path.dirname(meme_txt_file)
        motif_list_file = os.path.join(output_dir, 'motif_candidates_export.txt')
        with open(motif_list_file, 'w') as out:
            for c in unique_motifs:
                out.write(c + '\n')
        logger.info(f"[{cluster_name}] Extracted {len(unique_motifs)} unique motif candidates from {meme_txt_file} and saved to {motif_list_file}")

        return unique_motifs
    except Exception as e:
        logger.error(f"[{cluster_name}] Failed to extract motifs from {meme_txt_file}: {e}", exc_info=True)
        return []

# --- load_validation_data, count_and_correlate, visualize_correlations functions remain unchanged ---
# Paste the unchanged functions here:
# def load_validation_data(...): ...
# def count_and_correlate(...): ...
# def visualize_correlations(...): ...
# --- [Functions pasted below - identical to previous version] ---
def load_validation_data(validation_csv_path, master_fasta_path):
    """Loads validation data and merges sequences from the master FASTA."""
    # This runs only once in the main process, logging is fine here
    logger = logging.getLogger()
    logger.info(f"Loading validation data from {validation_csv_path} and {master_fasta_path}")

    seq_dict = {}
    try:
        with open(master_fasta_path) as f:
            current_id = None
            current_seq = []
            for line in f:
                line = line.strip()
                if not line: continue # Skip empty lines
                if line.startswith(">"):
                    # Process previous sequence if exists
                    if current_id is not None:
                         seq_dict[current_id] = "".join(current_seq)

                    # Parse new ID
                    try:
                        # Expecting format like >1|... or just >1
                        header_parts = line[1:].split("|", 1)
                        current_id = int(header_parts[0])
                        current_seq = [] # Reset sequence
                    except (IndexError, ValueError):
                        logger.warning(f"Could not parse integer ID from header: {line}. Skipping this entry.")
                        current_id = None # Skip sequence lines until next valid header
                elif current_id is not None:
                    # Append sequence lines (handle potential non-protein chars later if needed)
                    current_seq.append(line)

            # Add the last sequence in the file
            if current_id is not None:
                 seq_dict[current_id] = "".join(current_seq)

    except FileNotFoundError:
        logger.error(f"Master FASTA file not found: {master_fasta_path}")
        raise
    except Exception as e:
        logger.error(f"Error reading master FASTA {master_fasta_path}: {e}", exc_info=True)
        raise

    if not seq_dict:
        logger.error(f"No sequences loaded from {master_fasta_path}. Check file format and headers (expecting '>int|...' or '>int').")
        raise ValueError("Failed to load sequences from master FASTA.")
    logger.info(f"Loaded {len(seq_dict)} sequences from master FASTA.")

    try:
        val_df = pd.read_csv(validation_csv_path)
        logger.info(f"Loaded validation CSV {validation_csv_path} with shape {val_df.shape}")
    except FileNotFoundError:
        logger.error(f"Validation CSV file not found: {validation_csv_path}")
        raise
    except Exception as e:
        logger.error(f"Error reading validation CSV {validation_csv_path}: {e}", exc_info=True)
        raise

    # Check for 'idv_id' column before attempting conversion
    if "idv_id" not in val_df.columns:
        logger.error(f"'idv_id' column not found in {validation_csv_path}. Required for merging.")
        raise KeyError("'idv_id' column missing from validation CSV.")

    try:
        # Attempt conversion, handle potential errors like non-numeric values
        val_df["idv_id_orig"] = val_df["idv_id"] # Keep original for reference if needed
        val_df["idv_id"] = pd.to_numeric(val_df["idv_id"], errors='coerce').astype('Int64') # Use nullable Int64
        # Report rows that couldn't be converted
        invalid_ids = val_df["idv_id"].isna().sum()
        if invalid_ids > 0:
             logger.warning(f"{invalid_ids} rows in validation CSV had non-integer 'idv_id' values. These rows will be dropped.")
             val_df.dropna(subset=["idv_id"], inplace=True)

    except Exception as e:
         logger.error(f"Unexpected error converting 'idv_id' to integer: {e}", exc_info=True)
         raise

    initial_rows = len(val_df)
    logger.info(f"Attempting to merge sequences for {initial_rows} validation entries...")
    val_df["sequence"] = val_df["idv_id"].map(seq_dict)

    missing_seqs = val_df["sequence"].isna().sum()
    if missing_seqs > 0:
        logger.warning(f"{missing_seqs} entries in validation CSV (out of {initial_rows}) did not have a matching sequence ID in the master FASTA. These rows will be dropped.")
        # Log some missing IDs for debugging
        missing_ids_sample = val_df.loc[val_df["sequence"].isna(), "idv_id"].head().tolist()
        logger.warning(f"Sample of missing IDs: {missing_ids_sample}")
        val_df.dropna(subset=["sequence"], inplace=True)
        logger.info(f"{len(val_df)} rows remaining in validation data after removing missing sequences.")

    if val_df.empty:
         logger.error("Validation DataFrame is empty after merging sequences. Check IDs in CSV and FASTA headers.")
         raise ValueError("Validation data became empty after sequence merge.")

    # Add sequence length column for normalization
    val_df["seq_length"] = val_df["sequence"].str.len()
    zero_len_seqs = (val_df["seq_length"] == 0).sum()
    if zero_len_seqs > 0:
        logger.warning(f"Found {zero_len_seqs} entries with zero sequence length after merging. These will be excluded from correlation.")
        val_df = val_df[val_df["seq_length"] > 0].copy() # Keep only sequences with length > 0

    if val_df.empty:
         logger.error("Validation DataFrame is empty after removing zero-length sequences.")
         raise ValueError("Validation data became empty after removing zero-length sequences.")


    logger.info(f"Validation data loaded and merged successfully. Final shape for analysis: {val_df.shape}")
    return val_df

def count_and_correlate(val_df, motifs, properties, cluster_name):
    """Counts motifs, normalizes by sequence length, calculates correlations on normalized frequency, and applies FDR correction."""
    # Note: This function now runs within the worker process.
    # Imports needed within the function if it runs in a separate process
    # (though they are likely inherited, explicit import is safer)
    import logging, re
    import numpy as np
    import pandas as pd
    from scipy.stats import pearsonr, spearmanr
    from statsmodels.stats.multitest import multipletests

    logger = logging.getLogger()
    if not motifs:
        logger.warning(f"[{cluster_name}] No motifs provided. Skipping correlation.")
        return pd.DataFrame()

    # Ensure properties is a list
    if isinstance(properties, str):
        properties = [properties]

    logger.info(f"[{cluster_name}] Calculating correlations for {len(motifs)} motifs against properties: {properties}...")

    # Work on copy to avoid mutating original DataFrame passed between processes/iterations
    df = val_df.copy()

    # Ensure sequence length column exists (should be added in load_validation_data)
    if "seq_length" not in df.columns:
        logger.error(f"[{cluster_name}] 'seq_length' column missing from validation data. Cannot normalize counts.")
        # Attempt to calculate it here as a fallback, but ideally it should exist
        df["seq_length"] = df["sequence"].str.len()
        df = df[df["seq_length"] > 0].copy()
        if "seq_length" not in df.columns or df.empty:
             logger.error(f"[{cluster_name}] Failed fallback calculation of sequence length. Skipping correlation.")
             return pd.DataFrame()


    # --- Count and compute normalized frequency ---
    motifs_counted = [] # Keep track of motifs successfully counted
    for m in motifs:
        count_col = f"count_{m}"
        freq_col = f"freq_{m}"
        try:
            # Escape special regex characters in motif
            esc_m = re.escape(m)
            # Use lookahead assertion `(?=...)` to find overlapping matches
            df[count_col] = df["sequence"].apply(
                lambda s: len(re.findall(f"(?={esc_m})", str(s))) if pd.notna(s) else 0
            )
            # Normalize count by sequence length for frequency
            # Avoid division by zero using np.where
            df[freq_col] = np.where(df["seq_length"] > 0,
                                      df[count_col] / df["seq_length"],
                                      0)
            motifs_counted.append(m)
        except re.error as e:
            logger.error(f"[{cluster_name}] Regex error for motif '{m}': {e}. Skipping this motif.")
            # Drop columns if partially created
            df.drop(columns=[count_col, freq_col], errors='ignore', inplace=True)
            continue # Skip to the next motif
        except Exception as e:
            logger.error(f"[{cluster_name}] Error counting motif '{m}': {e}. Skipping this motif.")
            df.drop(columns=[count_col, freq_col], errors='ignore', inplace=True)
            continue # Skip to the next motif

    if not motifs_counted:
        logger.warning(f"[{cluster_name}] No motifs were successfully counted. Skipping correlation.")
        return pd.DataFrame()
    logger.info(f"[{cluster_name}] Successfully counted {len(motifs_counted)} motifs.")

    # --- Calculate Correlations ---
    results = []
    for m in motifs_counted:
        freq_col = f"freq_{m}"
        for p in properties:
            if p not in df.columns:
                logger.warning(f"[{cluster_name}] Property column '{p}' not found in validation data. Skipping correlation for {m} vs {p}.")
                continue

            # Get the two columns for correlation
            x_freq = df[freq_col]
            y_prop = df[p]

            # Ensure property column is numeric, coercing errors to NaN
            y_prop_numeric = pd.to_numeric(y_prop, errors='coerce')
            num_non_numeric = y_prop.notna().sum() - y_prop_numeric.notna().sum()
            if num_non_numeric > 0:
                 logger.warning(f"[{cluster_name}] Property '{p}' contained {num_non_numeric} non-numeric values which were converted to NaN.")


            # Filter out pairs with NaN in either column for correlation calculation
            mask = x_freq.notna() & y_prop_numeric.notna()
            n_pairs = mask.sum()

            if n_pairs > 2: # Need at least 3 pairs for meaningful correlation
                x_masked = x_freq[mask]
                y_masked = y_prop_numeric[mask]

                # Check for zero variance (prevents correlation calculation errors)
                if np.std(x_masked) > 1e-9 and np.std(y_masked) > 1e-9:
                    try:
                        r, p_r = pearsonr(x_masked, y_masked)
                        rho, p_s = spearmanr(x_masked, y_masked)
                        results.append((cluster_name, m, p, r, p_r, rho, p_s, n_pairs))
                    except Exception as e:
                        logger.warning(f"[{cluster_name}] Correlation calculation error for motif '{m}' vs property '{p}': {e}. Recording NaNs.")
                        results.append((cluster_name, m, p, np.nan, np.nan, np.nan, np.nan, n_pairs))
                else:
                    logger.warning(f"[{cluster_name}] Skipping correlation for motif '{m}' vs property '{p}': Insufficient variance in data (std_freq={np.std(x_masked):.2g}, std_prop={np.std(y_masked):.2g}).")
                    results.append((cluster_name, m, p, np.nan, np.nan, np.nan, np.nan, n_pairs))
            else:
                # Not enough valid pairs
                logger.warning(f"[{cluster_name}] Skipping correlation for motif '{m}' vs property '{p}': Insufficient valid data pairs ({n_pairs} found, need > 2).")
                results.append((cluster_name, m, p, np.nan, np.nan, np.nan, np.nan, n_pairs))

    if not results:
        logger.warning(f"[{cluster_name}] No correlation results were generated.")
        return pd.DataFrame()

    corr_df = pd.DataFrame(
        results,
        columns=["cluster", "motif", "property", "pearson_r", "p_pearson",
                 "spearman_rho", "p_spearman", "n_pairs"]
    )

    # --- FDR Correction (Benjamini/Hochberg) on Pearson p-values within this cluster ---
    valid_p_mask = corr_df["p_pearson"].notna()
    if valid_p_mask.any():
        pvals = corr_df.loc[valid_p_mask, "p_pearson"].values
        # Handle potential edge case of single p-value
        if len(pvals) > 1:
             reject, p_adj, _, _ = multipletests(pvals, method="fdr_bh")
        elif len(pvals) == 1:
             reject = [pvals[0] <= 0.05] # Simple comparison for single value
             p_adj = pvals
        else: # Should not happen if valid_p_mask.any() is true, but for safety
             reject = []
             p_adj = []

        corr_df.loc[valid_p_mask, "p_adj_pearson"] = p_adj
        corr_df.loc[valid_p_mask, "significant_pearson"] = reject
        corr_df["significant_pearson"] = corr_df["significant_pearson"].fillna(False).astype(bool)
    else:
        # No valid p-values to correct
        corr_df["p_adj_pearson"] = np.nan
        corr_df["significant_pearson"] = False

    num_significant = corr_df['significant_pearson'].sum()
    if num_significant > 0:
        logger.info(f"[{cluster_name}] Correlation complete. Found {num_significant} significant correlations (Pearson p_adj <= 0.05) after FDR.")
    else:
         logger.info(f"[{cluster_name}] Correlation complete. No significant correlations found after FDR.")

    # Add motif count and frequency columns back to the results df for potential plotting later
    # Need to merge based on validation data index if we want plots vs count/freq
    # For simplicity here, just return the correlation stats. Plotting might need adjustment.

    return corr_df

def visualize_correlations(val_df_with_counts, corr_df, output_plot_dir, cluster_name): # Add cluster_name
    """Generates scatter plots for significant correlations."""
    # Note: This function now runs within the worker process.
    import logging, re, os
    import matplotlib.pyplot as plt
    import pandas as pd # Ensure pandas is available here

    logger = logging.getLogger()
    if corr_df is None or corr_df.empty:
         logger.info(f"[{cluster_name}] No correlation data provided. Skipping plot generation.")
         return

    # Filter for significant correlations (use the FDR adjusted p-value)
    if 'significant_pearson' not in corr_df.columns:
        logger.warning(f"[{cluster_name}] 'significant_pearson' column missing in correlation results. Cannot determine significant plots.")
        return

    sig_df = corr_df[corr_df['significant_pearson']].copy()

    if sig_df.empty:
        logger.info(f"[{cluster_name}] No significant correlations found to plot.")
        return

    logger.info(f"[{cluster_name}] Generating {len(sig_df)} plots for significant correlations...")
    # Create plot directory ONLY if there are significant results
    os.makedirs(output_plot_dir, exist_ok=True)

    # Check if count/frequency columns are present in the passed validation DataFrame
    # This requires that count_and_correlate either returns the df with counts
    # or that process_single_cluster passes the modified df here.
    # For simplicity, assume count_and_correlate added the necessary columns to val_df_with_counts

    plotted_count = 0
    for _, row in sig_df.iterrows():
        motif = row['motif']
        prop = row['property']
        # Use frequency for the plot x-axis as it's what was correlated
        freq_col = f"freq_{motif}"

        if freq_col not in val_df_with_counts.columns:
            logger.warning(f"[{cluster_name}] Frequency column '{freq_col}' for motif '{motif}' not found in plotting data. Skipping plot vs {prop}.")
            continue
        if prop not in val_df_with_counts.columns:
             logger.warning(f"[{cluster_name}] Property column '{prop}' not found in plotting data. Skipping plot for {motif}.")
             continue

        plt.figure(figsize=(8, 6))

        # Ensure property is numeric for plotting
        y_plot_numeric = pd.to_numeric(val_df_with_counts[prop], errors='coerce')

        # Filter NaNs for plotting
        plot_mask = val_df_with_counts[freq_col].notna() & y_plot_numeric.notna()
        x_plot = val_df_with_counts.loc[plot_mask, freq_col]
        y_plot = y_plot_numeric[plot_mask]

        if len(x_plot) < 3:
             logger.warning(f"[{cluster_name}] Skipping plot for {motif} vs {prop} due to insufficient data points ({len(x_plot)}) after NaN removal.")
             plt.close() # Close the figure even if not saved
             continue

        plt.scatter(x_plot, y_plot, alpha=0.6, s=15) # Smaller points might be better

        plt.xlabel(f'Normalized Frequency of motif "{motif}"')
        plt.ylabel(prop.replace('_', ' ').title())
        plt.title(f'[{cluster_name}] {prop.replace("_", " ").title()} vs. Motif "{motif}" Freq.\n'
                  f'(Pearson r={row["pearson_r"]:.3f}, p_adj={row["p_adj_pearson"]:.2e}, n={int(row["n_pairs"])})')
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.tight_layout()

        # Sanitize filenames
        safe_motif_name = re.sub(r'[^\w\-]+', '_', motif)
        safe_prop_name = re.sub(r'[^\w\-]+', '_', prop)
        safe_cluster_name = re.sub(r'[^\w\-]+', '_', cluster_name) # Sanitize cluster name too
        # Limit filename length
        max_len = 50
        safe_motif_name = safe_motif_name[:max_len]
        safe_prop_name = safe_prop_name[:max_len]

        plot_filename = os.path.join(output_plot_dir, f"corr_{safe_cluster_name}_{safe_prop_name}_vs_{safe_motif_name}.png")

        try:
            plt.savefig(plot_filename, dpi=150) # Slightly higher DPI
            plotted_count += 1
            # logger.debug(f"[{cluster_name}] Saved plot: {plot_filename}") # Log only if debugging
        except Exception as e:
            logger.error(f"[{cluster_name}] Failed to save plot {plot_filename}: {e}")
        finally:
            plt.close() # Ensure figure is closed to free memory

    logger.info(f"[{cluster_name}] Saved {plotted_count} plots to {output_plot_dir}")
# --- [End of unchanged functions] ---


# --- Worker Function for Parallel Processing ---

def process_single_cluster(fasta_file, validation_df, unique_output_dir, args):
    """Worker function to process a single cluster FASTA file using MEME."""
    logger = logging.getLogger()
    cluster_name = os.path.splitext(os.path.basename(fasta_file))[0]
    logger.info(f"--- [{cluster_name}] Starting processing (using MEME) ---")
    start_time_cluster = time.time()

    cluster_result_df = None # Initialize result DataFrame for this cluster

    try:
        # 1. Run MEME (or use existing if --skip_meme)
        # Note: --skip_streme flag reused, maybe rename later if needed.
        meme_output_file = None
        meme_output_base = os.path.join(unique_output_dir, f"meme_out_{cluster_name}")

        if not args.skip_meme: # Using skip_meme argument name now
            meme_output_file = run_meme(
                fasta_file, unique_output_dir, args.minw, args.maxw,
                args.meme_mod, args.meme_nmotifs, args.evalue_thresh, args.conda_env
            )
            if meme_output_file is None:
                logger.warning(f"[{cluster_name}] MEME failed or was skipped due to empty file. Aborting processing for this cluster.")
                return None # Indicate failure or skip
        else:
            # Try to find existing meme.txt if skipping run
            potential_meme_file = os.path.join(meme_output_base, 'meme.txt')
            if os.path.exists(potential_meme_file):
                meme_output_file = potential_meme_file
                logger.info(f"[{cluster_name}] Skipping MEME run, using existing file: {meme_output_file}")
            else:
                logger.warning(f"[{cluster_name}] --skip_meme flag set, but existing file not found: {potential_meme_file}. Cannot process this cluster.")
                return None # Cannot proceed without MEME results


        # Check again if MEME output exists (might have failed or been skipped)
        if not meme_output_file:
             logger.warning(f"[{cluster_name}] No MEME output file available. Skipping motif extraction and validation.")
             return None


        # 2. Extract Motifs
        motifs = extract_motifs(meme_output_file, cluster_name) # Should parse meme.txt
        if not motifs:
            logger.warning(f"[{cluster_name}] No motifs extracted from {meme_output_file}. Skipping validation.")
            return None # No motifs to validate


        # 3. Count Motifs and Correlate
        cluster_corr_df = count_and_correlate(validation_df, motifs, args.properties, cluster_name)


        # Check if correlation produced results before plotting
        if cluster_corr_df is not None and not cluster_corr_df.empty:
             cluster_result_df = cluster_corr_df # Store results

             # 4. Visualize Significant Correlations (if requested and results exist)
             if args.generate_plots:
                 logger.info(f"[{cluster_name}] Preparing data for plotting significant correlations...")
                 df_for_plotting = validation_df.copy()
                 significant_motifs = cluster_corr_df[cluster_corr_df['significant_pearson']]['motif'].unique()

                 # Recalculate frequencies for plotting if needed
                 for m in significant_motifs:
                     freq_col = f"freq_{m}"
                     if freq_col not in df_for_plotting.columns:
                        try:
                            esc_m = re.escape(m)
                            df_for_plotting[f"count_{m}"] = df_for_plotting["sequence"].apply(
                                lambda s: len(re.findall(f"(?={esc_m})", str(s))) if pd.notna(s) else 0
                            )
                            df_for_plotting[freq_col] = np.where(df_for_plotting["seq_length"] > 0,
                                                                    df_for_plotting[f"count_{m}"] / df_for_plotting["seq_length"], 0)
                        except Exception as e:
                            logger.error(f"[{cluster_name}] Error recalculating frequency for plotting motif '{m}': {e}")
                            df_for_plotting.drop(columns=[f"count_{m}", freq_col], errors='ignore', inplace=True)

                 # Define plot output dir based on MEME output base
                 plot_output_dir = os.path.join(unique_output_dir, f"plots_meme_{cluster_name}")
                 visualize_correlations(df_for_plotting, cluster_corr_df, plot_output_dir, cluster_name)
        else:
             logger.info(f"[{cluster_name}] No correlation results generated. Skipping plotting.")


        elapsed_time = time.time() - start_time_cluster
        logger.info(f"--- [{cluster_name}] Finished processing in {elapsed_time:.2f} seconds (using MEME) ---")
        return cluster_result_df # Return the correlation results DataFrame (or None if failed earlier)

    except Exception as e:
        # Catch any unexpected errors within the worker
        logger.error(f"!!! [{cluster_name}] Unhandled exception during processing: {e}", exc_info=True) # Log traceback
        return None # Indicate failure


# --- Helper function to count sequences (Unchanged) ---
def count_fasta_sequences(fasta_path):
    """Counts the number of sequences in a FASTA file."""
    count = 0
    try:
        with open(fasta_path, 'r') as f:
            for line in f:
                if line.startswith(">"):
                    count += 1
    except FileNotFoundError:
        return 0
    except Exception as e:
        logging.warning(f"Could not count sequences in {os.path.basename(fasta_path)}: {e}")
        return 0
    return count


# --- Main Execution Logic ---
def main():
    parser = argparse.ArgumentParser(
        description="Automated Motif Discovery (MEME) and Validation Pipeline (Parallelized, Top-K/Min-Size Selection).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
        )

    # --- Input/Output Arguments ---
    parser.add_argument("--fasta_dir", required=True, help="Directory containing input cluster FASTA files (e.g., cluster_*.fasta).")
    parser.add_argument("--validation_csv", required=True, help="Path to the validation dataset CSV file (must contain 'idv_id' and property columns).")
    parser.add_argument("--master_fasta", required=True, help="Path to the master protein FASTA file (headers like '>id|...' or '>id').")
    parser.add_argument("--output_dir", required=True, help="Base directory path for storing outputs. A timestamp will be appended.")
    parser.add_argument("--results_file", default="combined_correlation_results_meme.csv", help="Filename for the final combined correlation results table.")

    # --- Cluster Selection Arguments ---
    parser.add_argument("--min_cluster_size", type=int, default=None,
                        help="Process clusters with at least this many sequences (ignored if --top_k_clusters is set). Recommended >= 5 for MEME.")
    parser.add_argument("--top_k_clusters", type=int, default=None,
                        help="Process only the top K largest clusters (takes precedence over --min_cluster_size).")

    # --- MEME Arguments --- ### MODIFIED FOR MEME ###
    parser.add_argument("--conda_env", required=True, help="Name of the Conda environment containing MEME Suite.")
    parser.add_argument("--minw", type=int, default=3, help="Minimum motif width for MEME.")
    parser.add_argument("--maxw", type=int, default=15, help="Maximum motif width for MEME.")
    parser.add_argument("--evalue_thresh", type=float, default=0.05, help="E-value reporting threshold for MEME (-evt).")
    parser.add_argument("--meme_nmotifs", type=int, default=5, help="Maximum number of motifs for MEME to find (-nmotifs).")
    parser.add_argument("--meme_mod", type=str, default='anr', choices=['oops', 'zoops', 'anr'], help="Motif distribution model for MEME (-mod). 'anr' is generally recommended.")
    parser.add_argument("--skip_meme", action='store_true', help="Skip running MEME and use existing meme.txt files (expects files in output_dir/meme_out_*/meme.txt).")


    # --- Validation Arguments ---
    parser.add_argument("--properties", nargs='+', default=["toughness", "young's_modulus", "tensile_strength", "strain_at_break"],
                        help="List of property column names in validation_csv to correlate against motif frequency.")

    # --- Optional Flags ---
    parser.add_argument("--generate_plots", action='store_true', help="Generate scatter plots for significant correlations (Pearson p_adj <= 0.05).")

    # --- Parallelism Control ---
    parser.add_argument("--num_workers", type=int, default=None,
                        help="Number of parallel processes (workers) to use. Defaults to the number of CPU cores.")


    args = parser.parse_args()
    logger = logging.getLogger()

    # --- Preparations (in main process) ---
    pipeline_start_time = time.time()
    base_output_dir = args.output_dir
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    unique_output_dir = f"{base_output_dir.rstrip(os.sep)}_MEME_{timestamp}" # Indicate MEME in dir name

    try:
        logger.info(f"Creating unique output directory for this run: {unique_output_dir}")
        os.makedirs(unique_output_dir, exist_ok=True)
    except OSError as e:
         logger.error(f"Failed to create output directory {unique_output_dir}: {e}")
         sys.exit(1)

    # Check Conda environment (same as before)
    try:
        logger.info(f"Checking Conda environment '{args.conda_env}'...")
        result = subprocess.run(['conda', 'run', '-n', args.conda_env, 'conda', 'list'], check=True, capture_output=True, text=True, timeout=30)
        logger.info(f"Conda environment '{args.conda_env}' seems accessible.")
    except Exception as e:
        logger.error(f"Failed Conda check for environment '{args.conda_env}': {e}", exc_info=True)
        sys.exit(1)


    # Load validation data (same as before)
    try:
        logger.info("Loading shared validation data...")
        validation_df = load_validation_data(args.validation_csv, args.master_fasta)
        missing_props = [p for p in args.properties if p not in validation_df.columns]
        if missing_props:
             logger.error(f"Specified property columns not found in validation data: {missing_props}")
             sys.exit(1)
        logger.info("Shared validation data loaded and properties checked.")
    except Exception as e:
         logger.error(f"Failed to load or prepare validation data: {e}", exc_info=True)
         sys.exit(1)


    # --- Cluster File Selection Logic (same as before) ---
    all_fasta_files_paths = glob.glob(os.path.join(args.fasta_dir, "cluster_*.fasta"))
    # ... (rest of the selection logic using --top_k_clusters or --min_cluster_size) ...
    # [Copy the exact selection logic block from the previous response here]
    fasta_files_to_process = []
    if args.top_k_clusters is not None and args.top_k_clusters > 0:
        k = args.top_k_clusters
        logger.info(f"Selecting Top {k} largest clusters for processing.")
        cluster_sizes = []
        logger.info("Counting sequences in all found clusters...")
        count_start_time = time.time()
        for f_path in all_fasta_files_paths:
            size = count_fasta_sequences(f_path)
            if size > 0: # Only consider non-empty files with >0 sequences
                cluster_sizes.append((size, f_path))
        logger.info(f"Sequence counting took {time.time() - count_start_time:.2f} seconds.")

        if not cluster_sizes:
             logger.error("No non-empty cluster files found.")
             sys.exit(1)

        cluster_sizes.sort(key=operator.itemgetter(0), reverse=True)
        top_k_clusters = cluster_sizes[:k]
        fasta_files_to_process = [f_path for size, f_path in top_k_clusters]

        if not fasta_files_to_process:
             logger.error(f"Top K selection resulted in zero files, though {len(cluster_sizes)} non-empty files were found.")
             sys.exit(1)
        logger.info(f"Selected {len(fasta_files_to_process)} clusters for processing based on Top K={k}.")
        if len(top_k_clusters) < k:
             logger.warning(f"Found fewer than {k} non-empty clusters ({len(top_k_clusters)} found). Processing all of them.")
        if fasta_files_to_process:
             min_size = top_k_clusters[-1][0]
             max_size = top_k_clusters[0][0]
             logger.info(f"Size range of selected clusters: {min_size} to {max_size} sequences.")

    elif args.min_cluster_size is not None and args.min_cluster_size >= 1: # Ensure min_size is at least 1
        min_size_threshold = args.min_cluster_size
        logger.info(f"Filtering clusters smaller than {min_size_threshold} sequences...")
        filtered_out_count = 0
        count_start_time = time.time()
        for f_path in all_fasta_files_paths:
            seq_count = count_fasta_sequences(f_path)
            if seq_count >= min_size_threshold:
                fasta_files_to_process.append(f_path)
            else:
                filtered_out_count += 1
        logger.info(f"Sequence counting and filtering took {time.time() - count_start_time:.2f} seconds.")

        if not fasta_files_to_process:
            logger.error(f"No cluster files met the minimum size requirement of {min_size_threshold}. Check data or threshold.")
            sys.exit(1)
        logger.info(f"Filtered out {filtered_out_count} clusters. Processing {len(fasta_files_to_process)} clusters >= {min_size_threshold} sequences.")
    else:
        logger.warning("No filtering specified (--top_k_clusters or --min_cluster_size).")
        safety_limit = 5000
        if len(all_fasta_files_paths) > safety_limit:
             logger.error(f"Found {len(all_fasta_files_paths)} clusters, exceeding safety limit {safety_limit} without filtering.")
             logger.error("Please use --top_k_clusters or --min_cluster_size.")
             sys.exit(1)
        else:
             logger.info(f"Proceeding to process all {len(all_fasta_files_paths)} found clusters.")
             fasta_files_to_process = all_fasta_files_paths
    # --- [End of selection logic block] ---


    # --- Parallel Execution (same structure, but calls worker using MEME) ---
    all_cluster_results = []
    max_workers = args.num_workers if args.num_workers is not None else os.cpu_count()
    logger.info(f"Starting parallel processing for {len(fasta_files_to_process)} selected clusters using up to {max_workers} workers (MEME)...")

    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_to_fasta = {
            executor.submit(process_single_cluster, fasta_path, validation_df, unique_output_dir, args): fasta_path
            for fasta_path in fasta_files_to_process
        }
        # ... (rest of the result collection loop is identical to previous response) ...
        num_processed = 0
        num_successful = 0
        num_failed = 0
        log_interval = min(max(1, len(fasta_files_to_process) // 20), 100)

        for future in concurrent.futures.as_completed(future_to_fasta):
            fasta_path = future_to_fasta[future]
            cluster_name_simple = os.path.splitext(os.path.basename(fasta_path))[0]
            try:
                result_df = future.result()
                if result_df is not None and not result_df.empty:
                    all_cluster_results.append(result_df)
                    num_successful += 1
                elif result_df is not None and result_df.empty:
                     num_successful += 1
                else:
                    num_failed += 1
            except Exception as exc:
                logger.error(f"!!! Main loop caught exception from worker processing {cluster_name_simple}: {exc}", exc_info=True)
                num_failed += 1

            num_processed += 1
            if num_processed % log_interval == 0 or num_processed == len(fasta_files_to_process):
                 logger.info(f"Progress: {num_processed}/{len(fasta_files_to_process)} clusters processed ({num_successful} successful, {num_failed} failed/skipped).")
    # --- [End of result collection loop] ---

    logger.info(f"Parallel processing finished. Processed {num_processed} clusters: {num_successful} successful, {num_failed} failed/skipped.")

    # --- Final Aggregation and Output (same as before) ---
    if not all_cluster_results:
        logger.warning("No correlation results were generated across any clusters.")
        total_time = time.time() - pipeline_start_time
        print(f"\nPipeline finished in {total_time:.2f} seconds. No results to save.")
        sys.exit(0)

    logger.info(f"Aggregating results from {len(all_cluster_results)} successful clusters...")
    try:
        final_results_df = pd.concat(all_cluster_results, ignore_index=True)
        sort_cols = ["significant_pearson", "p_adj_pearson"]
        final_results_df.sort_values(by=sort_cols, ascending=[False, True], inplace=True)
    except Exception as e:
         logger.error(f"Failed during final results aggregation or sorting: {e}", exc_info=True)
         sys.exit(1)

    final_output_path = os.path.join(unique_output_dir, args.results_file)
    try:
        final_results_df.to_csv(final_output_path, index=False, float_format='%.5g')
        logger.info(f"Combined correlation results saved to: {final_output_path}")
        logger.info(f"Final results table shape: {final_results_df.shape}")
        num_sig_final = final_results_df['significant_pearson'].sum()
        logger.info(f"Total significant correlations found (after cluster-level FDR): {num_sig_final}")
        if num_sig_final > 0:
            logger.info("Top 5 significant results:")
            # Increase display width for pandas output
            pd.set_option('display.width', 1000)
            pd.set_option('display.max_columns', None)
            print(final_results_df[final_results_df['significant_pearson']].head().to_string())
            pd.reset_option('display.width')
            pd.reset_option('display.max_columns')

    except Exception as e:
        logger.error(f"Failed to save final results to {final_output_path}: {e}", exc_info=True)

    total_time = time.time() - pipeline_start_time
    logger.info(f"--- Pipeline Finished Successfully in {total_time:.2f} seconds ({total_time/60:.2f} minutes) (using MEME) ---")


if __name__ == "__main__":
    main()