# (Parallel Version) Combines Motif Extraction/Validation pipeline for batched processing.
# Combines Motif Extraction/Validation pipeline for batched processing - PARALLELIZED.
"""
Example Command:
python motif_parallel_pipeline.py \
    --fasta_dir ./Data/v0_cluster_windows_15mers \
    --validation_csv ./Data/validation_dataset.csv \
    --master_fasta ./Data/spider-silkome-database.v1.prot.fasta \
    --output_dir ./Results/v0_Parallel_Run \
    --conda_env memesuite \
    --generate_plots \
    --num_workers 4  # Optional: Specify number of parallel processes (if not specified uses max workers: os.cpu_count())
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
import concurrent.futures # <-- IMPORT FOR PARALLELISM
import time # <-- For potential timing/debugging

# --- Configuration ---
# Setup logging - Note: Logging from multiple processes might interleave messages.
# Consider more advanced logging handlers for cleaner parallel logs if needed.
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(processName)s - %(levelname)s - %(message)s', stream=sys.stdout)

# --- Function Definitions ---

# (run_streme, extract_motifs, load_validation_data, count_and_correlate, visualize_correlations functions remain unchanged)
# They will be called by the new worker function below.
# ... [Paste the unchanged functions here] ...
def run_streme(fasta_file, output_dir, minw, maxw, thresh, conda_env_name):
    """Runs STREME on a given FASTA file using a specified conda environment."""
    base_name = os.path.splitext(os.path.basename(fasta_file))[0]
    # NOTE: output_dir here is the UNIQUE run directory already
    streme_output_subdir = os.path.join(output_dir, f"streme_out_{base_name}")
    os.makedirs(streme_output_subdir, exist_ok=True)

    cmd = [
        'conda', 'run', '-n', conda_env_name, '--no-capture-output',
        'streme',
        '--protein',
        '--p', fasta_file,
        '--minw', str(minw),
        '--maxw', str(maxw),
        '--thresh', str(thresh),
        '--oc', streme_output_subdir
    ]

    # Use cluster name in logging messages from worker
    logger = logging.getLogger() # Get root logger
    logger.info(f"[{base_name}] Running STREME for {fasta_file}...")
    logger.info(f"[{base_name}] Command: {' '.join(cmd)}")

    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        # Limit potentially long stdout in parallel logging
        stdout_summary = (result.stdout[:500] + '...') if len(result.stdout) > 500 else result.stdout
        logger.info(f"[{base_name}] STREME stdout summary:\n{stdout_summary}")
        if result.stderr:
            stderr_summary = (result.stderr[:500] + '...') if len(result.stderr) > 500 else result.stderr
            logger.warning(f"[{base_name}] STREME stderr summary:\n{stderr_summary}")
        logger.info(f"[{base_name}] STREME completed successfully. Output in {streme_output_subdir}")
        return os.path.join(streme_output_subdir, 'streme.txt')
    except FileNotFoundError:
        logger.error(f"[{base_name}] Error: 'conda' command not found. Is Conda installed and in PATH?")
        # Reraise critical errors or return None depending on desired behavior
        raise
    except subprocess.CalledProcessError as e:
        logger.error(f"[{base_name}] STREME failed with exit code {e.returncode}.")
        logger.error(f"[{base_name}] STDOUT:\n{e.stdout}")
        logger.error(f"[{base_name}] STDERR:\n{e.stderr}")
        return None # Indicate failure for this cluster
    except Exception as e:
        logger.error(f"[{base_name}] An unexpected error occurred during STREME execution: {e}")
        raise # Reraise unexpected errors

def extract_motifs(streme_txt_file, cluster_name): # Add cluster_name for logging
    """Extracts unique motif consensus sequences from a streme.txt file."""
    logger = logging.getLogger()
    if not streme_txt_file or not os.path.exists(streme_txt_file):
        logger.warning(f"[{cluster_name}] STREME output file not found or not provided: {streme_txt_file}. Skipping motif extraction.")
        return []

    consensi = []
    try:
        with open(streme_txt_file, 'r') as fh:
            for line in fh:
                if line.startswith('MOTIF '):
                    parts = line.split()
                    if len(parts) > 1:
                        rank_dash_consensus = parts[1]
                        if '-' in rank_dash_consensus:
                            consensus = rank_dash_consensus.split('-', 1)[1]
                            consensi.append(consensus)
                        else:
                             logger.warning(f"[{cluster_name}] Could not parse motif consensus from line: {line.strip()}")
                    else:
                        logger.warning(f"[{cluster_name}] Unexpected MOTIF line format: {line.strip()}")

        seen = set()
        unique_motifs = [c for c in consensi if not (c in seen or seen.add(c))]

        motif_list_file = os.path.join(os.path.dirname(streme_txt_file), 'motif_candidates_export.txt')
        with open(motif_list_file, 'w') as out:
            for c in unique_motifs:
                out.write(c + '\n')
        logger.info(f"[{cluster_name}] Extracted {len(unique_motifs)} unique motif candidates from {streme_txt_file} and saved to {motif_list_file}")

        return unique_motifs
    except Exception as e:
        logger.error(f"[{cluster_name}] Failed to extract motifs from {streme_txt_file}: {e}")
        return []


def load_validation_data(validation_csv_path, master_fasta_path):
    """Loads validation data and merges sequences from the master FASTA."""
    # This runs only once in the main process, logging is fine here
    logger = logging.getLogger()
    logger.info(f"Loading validation data from {validation_csv_path} and {master_fasta_path}")

    seq_dict = {}
    try:
        with open(master_fasta_path) as f:
            current_id = None
            for line in f:
                line = line.strip()
                if line.startswith(">"):
                    try:
                        current_id = int(line[1:].split("|", 1)[0])
                        seq_dict[current_id] = ""
                    except (IndexError, ValueError):
                        logger.warning(f"Could not parse ID from header: {line}")
                        current_id = None
                elif current_id is not None:
                    seq_dict[current_id] += line
    except FileNotFoundError:
        logger.error(f"Master FASTA file not found: {master_fasta_path}")
        raise
    except Exception as e:
        logger.error(f"Error reading master FASTA {master_fasta_path}: {e}")
        raise

    if not seq_dict:
        logger.error(f"No sequences loaded from {master_fasta_path}. Check file format and headers.")
        raise ValueError("Failed to load sequences from master FASTA.")

    try:
        val_df = pd.read_csv(validation_csv_path)
    except FileNotFoundError:
        logger.error(f"Validation CSV file not found: {validation_csv_path}")
        raise
    except Exception as e:
        logger.error(f"Error reading validation CSV {validation_csv_path}: {e}")
        raise

    try:
        val_df["idv_id"] = val_df["idv_id"].astype(int)
    except KeyError:
        logger.error(f"'idv_id' column not found in {validation_csv_path}")
        raise
    except Exception as e:
         logger.error(f"Error converting 'idv_id' to integer: {e}")
         raise

    initial_rows = len(val_df)
    val_df["sequence"] = val_df["idv_id"].map(seq_dict)

    missing_seqs = val_df["sequence"].isna().sum()
    if missing_seqs > 0:
        logger.warning(f"{missing_seqs} entries in validation CSV did not have a matching sequence in the master FASTA. These rows will be dropped.")
        val_df.dropna(subset=["sequence"], inplace=True)
        logger.info(f"{len(val_df)} rows remaining in validation data after removing missing sequences.")

    if val_df.empty:
         logger.error("Validation DataFrame is empty after merging sequences. Check IDs in CSV and FASTA headers.")
         raise ValueError("Validation data became empty after sequence merge.")

    logger.info(f"Validation data loaded successfully. Shape: {val_df.shape}")
    return val_df

def count_and_correlate(val_df, motifs, properties, cluster_name):
    """Counts motifs, normalizes, calculates correlations, and applies FDR correction."""
    logger = logging.getLogger()
    if not motifs:
        logger.warning(f"[{cluster_name}] No motifs provided. Skipping correlation.")
        return pd.DataFrame()

    logger.info(f"[{cluster_name}] Calculating correlations for {len(motifs)} motifs...")

    df = val_df.copy() # Ensure worker uses a copy

    if "seq_length" not in df.columns:
         df["seq_length"] = df["sequence"].str.len()

    df = df[df["seq_length"] > 0].copy()
    if df["seq_length"].min() == 0:
        logger.warning(f"[{cluster_name}] Removed sequences with zero length.")

    for m in motifs:
        try:
            escaped_m = re.escape(m)
            # Ensure sequence column is string type before applying regex
            df[f"count_{m}"] = df["sequence"].apply(lambda s: len(re.findall(f"(?={escaped_m})", str(s))) if pd.notna(s) else 0)
            # Ensure seq_length is not zero before division
            df[f"freq_{m}"] = np.where(df["seq_length"] > 0, df[f"count_{m}"] / df["seq_length"], 0)
        except re.error as e:
            logger.error(f"[{cluster_name}] Regex error for motif '{m}': {e}. Skipping this motif.")
            if f"count_{m}" in df.columns: df.drop(columns=[f"count_{m}"], inplace=True)
            if f"freq_{m}" in df.columns: df.drop(columns=[f"freq_{m}"], inplace=True)
            continue
        except Exception as e:
             logger.error(f"[{cluster_name}] Error counting motif '{m}': {e}. Skipping this motif.")
             if f"count_{m}" in df.columns: df.drop(columns=[f"count_{m}"], inplace=True)
             if f"freq_{m}" in df.columns: df.drop(columns=[f"freq_{m}"], inplace=True)
             continue

    valid_motifs = [m for m in motifs if f"count_{m}" in df.columns]
    if not valid_motifs:
         logger.warning(f"[{cluster_name}] No motifs were successfully counted.")
         return pd.DataFrame()

    results = []
    for m in valid_motifs:
        count_col = f"count_{m}"
        for p in properties:
            if p not in df.columns:
                logger.warning(f"[{cluster_name}] Property '{p}' not found in validation data. Skipping correlation for {m} vs {p}.")
                continue

            x, y = df[count_col], df[p]
            mask = x.notna() & y.notna()

            if mask.sum() > 2:
                try:
                    # Ensure finite values for correlation
                    x_masked = x[mask]
                    y_masked = y[mask]
                    if np.all(np.isfinite(x_masked)) and np.all(np.isfinite(y_masked)):
                         # Check for zero variance which causes error in pearsonr/spearmanr
                        if np.std(x_masked) > 0 and np.std(y_masked) > 0:
                            r, pval = pearsonr(x_masked, y_masked)
                            rho, sval = spearmanr(x_masked, y_masked)
                            results.append((cluster_name, m, p, r, pval, rho, sval, mask.sum()))
                        else:
                            # Handle zero variance case
                            logger.warning(f"[{cluster_name}] Skipping correlation for {m} vs {p} due to zero variance in data.")
                            results.append((cluster_name, m, p, np.nan, np.nan, np.nan, np.nan, mask.sum()))
                    else:
                        logger.warning(f"[{cluster_name}] Skipping correlation for {m} vs {p} due to non-finite values.")
                        results.append((cluster_name, m, p, np.nan, np.nan, np.nan, np.nan, mask.sum()))

                except ValueError as ve:
                     logger.warning(f"[{cluster_name}] Could not calculate correlation for {m} vs {p}. Reason: {ve}. Skipping.")
                     results.append((cluster_name, m, p, np.nan, np.nan, np.nan, np.nan, mask.sum()))
                except Exception as e:
                     logger.error(f"[{cluster_name}] Unexpected error during correlation for {m} vs {p}: {e}. Skipping.")
                     results.append((cluster_name, m, p, np.nan, np.nan, np.nan, np.nan, mask.sum()))
            else:
                results.append((cluster_name, m, p, np.nan, np.nan, np.nan, np.nan, mask.sum()))

    if not results:
        logger.warning(f"[{cluster_name}] No correlation results generated.")
        return pd.DataFrame()

    corr_df = pd.DataFrame(results, columns=["cluster", "motif", "property", "pearson_r", "p_pearson", "spearman_rho", "p_spearman", "n_pairs"])

    valid_p = corr_df["p_pearson"].notna()
    if valid_p.sum() > 0:
        pvals_to_correct = corr_df.loc[valid_p, "p_pearson"].dropna() # Ensure no NaNs passed to multipletests
        if not pvals_to_correct.empty:
            reject, p_adj, _, _ = multipletests(pvals_to_correct, method="fdr_bh")
            corr_df["p_adj_pearson"] = np.nan
            corr_df["significant_pearson"] = False
            # Align results back using index
            corr_df.loc[pvals_to_correct.index, "p_adj_pearson"] = p_adj
            corr_df.loc[pvals_to_correct.index, "significant_pearson"] = reject
        else:
             corr_df["p_adj_pearson"] = np.nan
             corr_df["significant_pearson"] = False

    else:
        corr_df["p_adj_pearson"] = np.nan
        corr_df["significant_pearson"] = False

    logger.info(f"[{cluster_name}] Correlation complete. Found {corr_df['significant_pearson'].sum()} significant Pearson correlations (after FDR).")

    return corr_df

def visualize_correlations(val_df, corr_df, motifs, output_plot_dir, cluster_name): # Add cluster_name
    """Generates scatter plots for significant correlations."""
    logger = logging.getLogger()
    # Ensure corr_df is not None or empty before filtering
    if corr_df is None or corr_df.empty or 'significant_pearson' not in corr_df.columns:
         logger.info(f"[{cluster_name}] No correlation data available to generate plots.")
         return

    sig_df = corr_df[corr_df['significant_pearson']].copy()

    if sig_df.empty:
        logger.info(f"[{cluster_name}] No significant correlations found to plot.")
        return

    logger.info(f"[{cluster_name}] Generating {len(sig_df)} plots for significant correlations...")
    os.makedirs(output_plot_dir, exist_ok=True)

    df_plot = val_df.copy()
    motif_count_cols = {}
    for m in motifs:
         count_col = f"count_{m}"
         # Check if count column exists in df_plot AND was actually used (present in corr_df)
         if count_col in df_plot.columns and count_col in df_plot:
             motif_count_cols[m] = count_col
         else:
              # This motif might have failed counting earlier
              pass


    for _, row in sig_df.iterrows():
        motif = row['motif']
        prop = row['property']

        if motif not in motif_count_cols:
            logger.warning(f"[{cluster_name}] Count column for motif '{motif}' not found in plotting data. Skipping plot for {prop}.")
            continue

        count_col = motif_count_cols[motif]

        plt.figure(figsize=(8, 6))

        plot_mask = df_plot[count_col].notna() & df_plot[prop].notna()
        x_plot = df_plot.loc[plot_mask, count_col]
        y_plot = df_plot.loc[plot_mask, prop]

        if len(x_plot) < 3:
             logger.warning(f"[{cluster_name}] Skipping plot for {motif} vs {prop} due to insufficient data points after NaN removal.")
             plt.close()
             continue

        plt.scatter(x_plot, y_plot, alpha=0.7)

        plt.xlabel(f'Count of motif "{motif}"')
        plt.ylabel(prop.replace('_', ' ').title())
        plt.title(f'[{cluster_name}] {prop.replace("_", " ").title()} vs. Motif "{motif}"\n(Pearson r={row["pearson_r"]:.2f}, p_adj={row["p_adj_pearson"]:.2e})')
        plt.tight_layout()

        safe_motif_name = re.sub(r'[^\w\-]+', '_', motif)
        safe_prop_name = re.sub(r'[^\w\-]+', '_', prop)
        plot_filename = os.path.join(output_plot_dir, f"corr_{safe_prop_name}_vs_{safe_motif_name}.png")

        try:
            plt.savefig(plot_filename)
            # Reduce logging verbosity for plots in parallel mode
            # logger.info(f"[{cluster_name}] Saved plot: {plot_filename}")
        except Exception as e:
            logger.error(f"[{cluster_name}] Failed to save plot {plot_filename}: {e}")
        plt.close() # Close the figure to free memory

# --- Worker Function for Parallel Processing ---

def process_single_cluster(fasta_file, validation_df, unique_output_dir, args):
    """Worker function to process a single cluster FASTA file."""
    logger = logging.getLogger() # Get logger configured in main process
    cluster_name = os.path.splitext(os.path.basename(fasta_file))[0]
    logger.info(f"--- Starting processing for Cluster: {cluster_name} ---")

    try:
        streme_output_file = None
        streme_output_subdir = os.path.join(unique_output_dir, f"streme_out_{cluster_name}")

        if not args.skip_streme:
            streme_output_file = run_streme(
                fasta_file, unique_output_dir, args.minw, args.maxw, args.thresh, args.conda_env
            )
            if streme_output_file is None:
                logger.warning(f"[{cluster_name}] STREME failed, skipping further steps for this cluster.")
                return None # Return None to indicate failure
        else:
            potential_streme_file = os.path.join(streme_output_subdir, 'streme.txt')
            if os.path.exists(potential_streme_file):
                streme_output_file = potential_streme_file
                logger.info(f"[{cluster_name}] Skipping STREME run, using existing file: {streme_output_file}")
            else:
                logger.warning(f"[{cluster_name}] Skip STREME flag set, but existing file not found: {potential_streme_file}. Cannot process.")
                return None # Return None

        motifs = extract_motifs(streme_output_file, cluster_name) # Pass cluster_name for logging

        if not motifs:
            logger.warning(f"[{cluster_name}] No motifs extracted. Skipping validation.")
            return None # Return None

        cluster_corr_df = count_and_correlate(validation_df, motifs, args.properties, cluster_name)

        if args.generate_plots and cluster_corr_df is not None and not cluster_corr_df.empty:
            plot_output_dir = os.path.join(unique_output_dir, f"plots_{cluster_name}")
            visualize_correlations(validation_df, cluster_corr_df, motifs, plot_output_dir, cluster_name) # Pass cluster_name

        logger.info(f"--- Finished processing for Cluster: {cluster_name} ---")
        return cluster_corr_df # Return the results DataFrame

    except Exception as e:
        logger.error(f"!!! Unhandled exception processing cluster {cluster_name}: {e}", exc_info=True) # Log traceback
        return None # Indicate failure


# --- Main Execution Logic ---

def main():
    parser = argparse.ArgumentParser(description="Automated Motif Discovery and Validation Pipeline (Parallelized).")

    # Input/Output args (same as before)
    parser.add_argument("--fasta_dir", required=True, help="Directory containing input FASTA files (e.g., cluster_*.fasta).")
    parser.add_argument("--validation_csv", required=True, help="Path to the validation dataset CSV file.")
    parser.add_argument("--master_fasta", required=True, help="Path to the master protein FASTA file.")
    parser.add_argument("--output_dir", required=True, help="Base directory path for storing outputs. A timestamp will be appended.")
    parser.add_argument("--results_file", default="combined_correlation_results.csv", help="Filename for the final combined correlation results table.")

    # STREME args (same as before)
    parser.add_argument("--conda_env", required=True, help="Name of the Conda environment for MEME Suite.")
    parser.add_argument("--minw", type=int, default=3, help="Minimum motif width for STREME.")
    parser.add_argument("--maxw", type=int, default=15, help="Maximum motif width for STREME.")
    parser.add_argument("--thresh", type=float, default=0.05, help="E-value threshold for STREME.")

    # Validation args (same as before)
    parser.add_argument("--properties", nargs='+', default=["toughness", "young's_modulus", "tensile_strength", "strain_at_break"],
                        help="List of property columns in validation_csv to correlate against.")

    # Optional Flags (same as before)
    parser.add_argument("--generate_plots", action='store_true', help="Generate scatter plots for significant correlations.")
    parser.add_argument("--skip_streme", action='store_true', help="Skip running STREME and use existing streme.txt files.")

    # *** NEW: Parallelism Control ***
    parser.add_argument("--num_workers", type=int, default=None, # Default to None, will use os.cpu_count()
                        help="Number of parallel processes to use. Defaults to the number of CPU cores available.")


    args = parser.parse_args()
    logger = logging.getLogger()

    # --- Preparations (in main process) ---
    start_time = time.time()
    base_output_dir = args.output_dir
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    unique_output_dir = f"{base_output_dir.rstrip(os.sep)}_{timestamp}"

    logger.info(f"Creating unique output directory for this run: {unique_output_dir}")
    os.makedirs(unique_output_dir, exist_ok=True)

    try:
        subprocess.run(['conda', 'env', 'list'], check=True, capture_output=True, text=True)
    except FileNotFoundError:
        logger.error("Could not run 'conda'. Is Conda installed and in your PATH?")
        sys.exit(1)

    try:
        # Load validation data ONCE here, it will be passed to workers
        logger.info("Loading shared validation data...")
        validation_df = load_validation_data(args.validation_csv, args.master_fasta)
        logger.info("Shared validation data loaded.")
    except Exception as e:
         logger.error(f"Failed to load validation data. Exiting. Error: {e}")
         sys.exit(1)

    fasta_files = glob.glob(os.path.join(args.fasta_dir, "cluster_*.fasta"))
    if not fasta_files:
        logger.error(f"No 'cluster_*.fasta' files found in directory: {args.fasta_dir}")
        sys.exit(1)
    fasta_files.sort() # Ensure consistent order
    logger.info(f"Found {len(fasta_files)} cluster FASTA files to process.")


    # --- Parallel Execution ---
    all_cluster_results = []
    # Determine number of workers
    max_workers = args.num_workers if args.num_workers is not None else os.cpu_count()
    logger.info(f"Starting parallel processing with up to {max_workers} workers...")

    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_fasta = {
            executor.submit(process_single_cluster, fasta, validation_df, unique_output_dir, args): fasta
            for fasta in fasta_files
        }

        num_processed = 0
        for future in concurrent.futures.as_completed(future_to_fasta):
            fasta_path = future_to_fasta[future]
            cluster_name_simple = os.path.splitext(os.path.basename(fasta_path))[0]
            try:
                result_df = future.result() # Get result or raise exception from worker
                if result_df is not None and not result_df.empty:
                    all_cluster_results.append(result_df)
                    logger.info(f"Successfully completed processing for {cluster_name_simple}.")
                elif result_df is not None and result_df.empty:
                     logger.info(f"Processing completed for {cluster_name_simple}, but no correlation results were generated.")
                else:
                    # Failure was handled within worker and returned None
                    logger.warning(f"Processing failed or produced no results for {cluster_name_simple}.")

            except Exception as exc:
                logger.error(f"!!! Exception generated by worker processing {cluster_name_simple}: {exc}", exc_info=True) # Log exception traceback

            num_processed += 1
            logger.info(f"Progress: {num_processed}/{len(fasta_files)} clusters processed.")


    logger.info("Parallel processing finished.")

    # --- Final Aggregation and Output (in main process) ---
    if not all_cluster_results:
        logger.warning("No correlation results were generated across any clusters.")
        print(f"Pipeline finished in {time.time() - start_time:.2f} seconds. No results to save.")
        sys.exit(0)

    logger.info("Aggregating results...")
    final_results_df = pd.concat(all_cluster_results, ignore_index=True)
    final_results_df.sort_values(by=["significant_pearson", "p_adj_pearson"], ascending=[False, True], inplace=True)

    final_output_path = os.path.join(unique_output_dir, args.results_file)
    try:
        final_results_df.to_csv(final_output_path, index=False)
        logger.info(f"Combined correlation results saved to: {final_output_path}")
    except Exception as e:
        logger.error(f"Failed to save final results to {final_output_path}: {e}")

    total_time = time.time() - start_time
    logger.info(f"--- Pipeline Finished Successfully in {total_time:.2f} seconds ---")


if __name__ == "__main__":
    # This check is important for multiprocessing on some OS (like Windows)
    main()