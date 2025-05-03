# Combines Motif Extraction/Validation pipeline for batched processing.
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
from contextlib import redirect_stderr # To capture conda messages if needed

# --- Configuration ---
# Setup logging for progress and errors
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', stream=sys.stdout)

# --- Function Definitions ---

def run_streme(fasta_file, output_dir, minw, maxw, thresh, conda_env_name):
    """Runs STREME on a given FASTA file using a specified conda environment."""
    base_name = os.path.splitext(os.path.basename(fasta_file))[0]
    streme_output_subdir = os.path.join(output_dir, f"streme_out_{base_name}")
    os.makedirs(streme_output_subdir, exist_ok=True)
    
    # Construct the STREME command within the conda environment
    cmd = [
        'conda', 'run', '-n', conda_env_name, '--no-capture-output', # Added --no-capture-output for visibility
        'streme',
        '--protein',
        '--p', fasta_file,
        '--minw', str(minw),
        '--maxw', str(maxw),
        '--thresh', str(thresh),
        '--oc', streme_output_subdir
    ]
    
    logging.info(f"Running STREME for {fasta_file}...")
    logging.info(f"Command: {' '.join(cmd)}")
    
    try:
        # Run the command, stream stdout/stderr to log file or console
        # Using check=True will raise CalledProcessError if streme fails
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        logging.info(f"STREME stdout for {base_name}:\n{result.stdout}")
        if result.stderr:
            logging.warning(f"STREME stderr for {base_name}:\n{result.stderr}")
        logging.info(f"STREME completed successfully for {fasta_file}. Output in {streme_output_subdir}")
        return os.path.join(streme_output_subdir, 'streme.txt')
    except FileNotFoundError:
        logging.error(f"Error: 'conda' command not found. Is Conda installed and in your PATH?")
        raise
    except subprocess.CalledProcessError as e:
        logging.error(f"STREME failed for {fasta_file} with exit code {e.returncode}.")
        logging.error(f"STDOUT:\n{e.stdout}")
        logging.error(f"STDERR:\n{e.stderr}")
        # Optionally, decide if you want to stop the whole script or just skip this file
        # raise  # Re-raise the exception to stop the script
        return None # Return None to indicate failure for this file
    except Exception as e:
        logging.error(f"An unexpected error occurred during STREME execution for {fasta_file}: {e}")
        raise


def extract_motifs(streme_txt_file):
    """Extracts unique motif consensus sequences from a streme.txt file."""
    if not streme_txt_file or not os.path.exists(streme_txt_file):
        logging.warning(f"STREME output file not found or not provided: {streme_txt_file}. Skipping motif extraction.")
        return []
        
    consensi = []
    try:
        with open(streme_txt_file, 'r') as fh:
            for line in fh:
                if line.startswith('MOTIF '):
                    # line looks like: "MOTIF 1-VSAL STREME-1"
                    parts = line.split()
                    if len(parts) > 1:
                        rank_dash_consensus = parts[1] # e.g. "1-VSAL"
                        if '-' in rank_dash_consensus:
                            consensus = rank_dash_consensus.split('-', 1)[1]
                            consensi.append(consensus)
                        else:
                             logging.warning(f"Could not parse motif consensus from line: {line.strip()}")
                    else:
                        logging.warning(f"Unexpected MOTIF line format: {line.strip()}")


        # Keep only unique, in discovery order
        seen = set()
        unique_motifs = [c for c in consensi if not (c in seen or seen.add(c))]
        
        # Optionally save to a file (useful for debugging)
        motif_list_file = os.path.join(os.path.dirname(streme_txt_file), 'motif_candidates_export.txt')
        with open(motif_list_file, 'w') as out:
            for c in unique_motifs:
                out.write(c + '\n')
        logging.info(f"Extracted {len(unique_motifs)} unique motif candidates from {streme_txt_file} and saved to {motif_list_file}")
        
        return unique_motifs
    except Exception as e:
        logging.error(f"Failed to extract motifs from {streme_txt_file}: {e}")
        return []


def load_validation_data(validation_csv_path, master_fasta_path):
    """Loads validation data and merges sequences from the master FASTA."""
    logging.info(f"Loading validation data from {validation_csv_path} and {master_fasta_path}")
    
    # 1) Load sequences from master FASTA
    seq_dict = {}
    try:
        with open(master_fasta_path) as f:
            current_id = None
            for line in f:
                line = line.strip()
                if line.startswith(">"):
                    try:
                        # header example: ">123|7047|Oecobiidae|…"
                        current_id = int(line[1:].split("|", 1)[0])
                        seq_dict[current_id] = ""
                    except (IndexError, ValueError):
                        logging.warning(f"Could not parse ID from header: {line}")
                        current_id = None # Skip sequences with unparsable headers
                elif current_id is not None:
                     # Append sequence lines only if a valid ID was parsed
                    seq_dict[current_id] += line
    except FileNotFoundError:
        logging.error(f"Master FASTA file not found: {master_fasta_path}")
        raise
    except Exception as e:
        logging.error(f"Error reading master FASTA {master_fasta_path}: {e}")
        raise
        
    if not seq_dict:
        logging.error(f"No sequences loaded from {master_fasta_path}. Check file format and headers.")
        raise ValueError("Failed to load sequences from master FASTA.")

    # 2) Load validation CSV
    try:
        val_df = pd.read_csv(validation_csv_path)
    except FileNotFoundError:
        logging.error(f"Validation CSV file not found: {validation_csv_path}")
        raise
    except Exception as e:
        logging.error(f"Error reading validation CSV {validation_csv_path}: {e}")
        raise

    # 3) Ensure idv_id is integer
    try:
        val_df["idv_id"] = val_df["idv_id"].astype(int)
    except KeyError:
        logging.error(f"'idv_id' column not found in {validation_csv_path}")
        raise
    except Exception as e:
         logging.error(f"Error converting 'idv_id' to integer: {e}")
         raise

    # 4) Merge in the spidroin sequence
    initial_rows = len(val_df)
    val_df["sequence"] = val_df["idv_id"].map(seq_dict)
    
    # 5) Check for missing sequences and drop rows if necessary
    missing_seqs = val_df["sequence"].isna().sum()
    if missing_seqs > 0:
        logging.warning(f"{missing_seqs} entries in validation CSV did not have a matching sequence in the master FASTA. These rows will be dropped.")
        val_df.dropna(subset=["sequence"], inplace=True)
        logging.info(f"{len(val_df)} rows remaining in validation data after removing missing sequences.")
        
    if val_df.empty:
         logging.error("Validation DataFrame is empty after merging sequences. Check IDs in CSV and FASTA headers.")
         raise ValueError("Validation data became empty after sequence merge.")

    logging.info(f"Validation data loaded successfully. Shape: {val_df.shape}")
    # logging.debug(f"Validation data head:\n{val_df.head()}") # Use debug level for verbose output
    return val_df


def count_and_correlate(val_df, motifs, properties, cluster_name):
    """Counts motifs, normalizes, calculates correlations, and applies FDR correction."""
    if not motifs:
        logging.warning(f"No motifs provided for cluster {cluster_name}. Skipping correlation.")
        return pd.DataFrame() # Return empty DataFrame

    logging.info(f"Calculating correlations for {len(motifs)} motifs from cluster {cluster_name}...")
    
    # Create a copy to avoid modifying the original DataFrame passed between clusters
    df = val_df.copy()
    
    # --- Motif Counts & Frequencies ---
    # Calculate sequence length if not present
    if "seq_length" not in df.columns:
         df["seq_length"] = df["sequence"].str.len()
         
    # Handle potential zero length sequences if any (though unlikely for proteins)
    df = df[df["seq_length"] > 0].copy() 
    if df["seq_length"].min() == 0:
        logging.warning("Removed sequences with zero length.")

    for m in motifs:
        # Use positive lookahead `(?=...)` for overlapping matches
        try:
            # Escape special regex characters in motif if necessary (e.g., '.', '*', '+')
            escaped_m = re.escape(m)
            df[f"count_{m}"] = df["sequence"].apply(lambda s: len(re.findall(f"(?={escaped_m})", s)) if isinstance(s, str) else 0)
            df[f"freq_{m}"] = df[f"count_{m}"] / df["seq_length"]
        except re.error as e:
            logging.error(f"Regex error for motif '{m}': {e}. Skipping this motif.")
            # Optionally remove columns if created partially
            if f"count_{m}" in df.columns: df.drop(columns=[f"count_{m}"], inplace=True)
            if f"freq_{m}" in df.columns: df.drop(columns=[f"freq_{m}"], inplace=True)
            continue # Skip to the next motif
        except Exception as e:
             logging.error(f"Error counting motif '{m}': {e}. Skipping this motif.")
             if f"count_{m}" in df.columns: df.drop(columns=[f"count_{m}"], inplace=True)
             if f"freq_{m}" in df.columns: df.drop(columns=[f"freq_{m}"], inplace=True)
             continue

    # Filter motifs list to only include those successfully counted
    valid_motifs = [m for m in motifs if f"count_{m}" in df.columns]
    if not valid_motifs:
         logging.warning(f"No motifs were successfully counted for cluster {cluster_name}.")
         return pd.DataFrame()
            
    # --- Compute Correlation ---
    results = []
    for m in valid_motifs:
        count_col = f"count_{m}" # Or use freq_col = f"freq_{m}" if you prefer frequency
        for p in properties:
            if p not in df.columns:
                logging.warning(f"Property '{p}' not found in validation data. Skipping correlation for {m} vs {p}.")
                continue
                
            x, y = df[count_col], df[p]
            # Ensure we only correlate where both motif count and property are non-null
            mask = x.notna() & y.notna()
            
            # Need at least 3 valid data points for correlation
            if mask.sum() > 2:
                try:
                    r, pval = pearsonr(x[mask], y[mask])
                    rho, sval = spearmanr(x[mask], y[mask])
                    results.append((cluster_name, m, p, r, pval, rho, sval, mask.sum()))
                except ValueError as ve:
                    # Handle cases like constant input (zero variance)
                     logging.warning(f"Could not calculate correlation for {m} vs {p} (Cluster: {cluster_name}). Reason: {ve}. Skipping.")
                     results.append((cluster_name, m, p, np.nan, np.nan, np.nan, np.nan, mask.sum()))
                except Exception as e:
                     logging.error(f"Unexpected error during correlation for {m} vs {p} (Cluster: {cluster_name}): {e}. Skipping.")
                     results.append((cluster_name, m, p, np.nan, np.nan, np.nan, np.nan, mask.sum()))
            else:
                # Not enough data points for correlation
                results.append((cluster_name, m, p, np.nan, np.nan, np.nan, np.nan, mask.sum()))

    if not results:
        logging.warning(f"No correlation results generated for cluster {cluster_name}.")
        return pd.DataFrame()

    corr_df = pd.DataFrame(results, columns=["cluster", "motif", "property", "pearson_r", "p_pearson", "spearman_rho", "p_spearman", "n_pairs"])
    
    # --- FDR correction on Pearson p-values ---
    # Handle NaN p-values before correction
    valid_p = corr_df["p_pearson"].notna()
    if valid_p.sum() > 0:
        pvals_to_correct = corr_df.loc[valid_p, "p_pearson"]
        reject, p_adj, _, _ = multipletests(pvals_to_correct, method="fdr_bh")
        
        # Add adjusted p-values and significance back to the DataFrame
        corr_df["p_adj_pearson"] = np.nan
        corr_df["significant_pearson"] = False
        corr_df.loc[valid_p, "p_adj_pearson"] = p_adj
        corr_df.loc[valid_p, "significant_pearson"] = reject
    else:
        # No valid p-values to correct
        corr_df["p_adj_pearson"] = np.nan
        corr_df["significant_pearson"] = False

    logging.info(f"Correlation calculation complete for cluster {cluster_name}. Found {corr_df['significant_pearson'].sum()} significant Pearson correlations (after FDR).")
    # logging.debug(f"Correlation results for {cluster_name}:\n{corr_df.sort_values('p_adj_pearson')}")

    return corr_df


def visualize_correlations(val_df, corr_df, motifs, output_plot_dir):
    """Generates scatter plots for significant correlations."""
    sig_df = corr_df[corr_df['significant_pearson']].copy() # Work on a copy
    
    if sig_df.empty:
        logging.info("No significant correlations found to plot.")
        return
        
    logging.info(f"Generating {len(sig_df)} plots for significant correlations...")
    os.makedirs(output_plot_dir, exist_ok=True)

    # Ensure count columns exist from the correlation step (use the motifs list from that step)
    df_plot = val_df.copy()
    motif_count_cols = {}
    for m in motifs:
         count_col = f"count_{m}"
         if count_col in df_plot.columns:
             motif_count_cols[m] = count_col
         else:
              logging.warning(f"Count column {count_col} not found in DataFrame for plotting motif {m}. Skipping plots for this motif.")


    for _, row in sig_df.iterrows():
        motif = row['motif']
        prop = row['property']
        
        # Check if the necessary count column is available
        if motif not in motif_count_cols:
            continue # Skip if count column wasn't created or found
            
        count_col = motif_count_cols[motif]

        plt.figure(figsize=(8, 6))
        
        # Handle potential NaNs in plotting data
        plot_mask = df_plot[count_col].notna() & df_plot[prop].notna()
        x_plot = df_plot.loc[plot_mask, count_col]
        y_plot = df_plot.loc[plot_mask, prop]

        if len(x_plot) < 3: # Need points to plot
             logging.warning(f"Skipping plot for {motif} vs {prop} due to insufficient data points after NaN removal.")
             plt.close() # Close the empty figure
             continue

        plt.scatter(x_plot, y_plot, alpha=0.7)
        
        plt.xlabel(f'Count of motif "{motif}"')
        plt.ylabel(prop.replace('_', ' ').title())
        plt.title(f'{prop.replace("_", " ").title()} vs. Motif "{motif}"\n(Pearson r={row["pearson_r"]:.2f}, p_adj={row["p_adj_pearson"]:.2e})')
        plt.tight_layout()
        
        # Sanitize filename (replace spaces, slashes etc.)
        safe_motif_name = re.sub(r'[^\w\-]+', '_', motif)
        safe_prop_name = re.sub(r'[^\w\-]+', '_', prop)
        plot_filename = os.path.join(output_plot_dir, f"corr_{safe_prop_name}_vs_{safe_motif_name}.png")
        
        try:
            plt.savefig(plot_filename)
            logging.info(f"Saved plot: {plot_filename}")
        except Exception as e:
            logging.error(f"Failed to save plot {plot_filename}: {e}")
        plt.close() # Close the figure to free memory


# --- Main Execution Logic ---

def main():
    parser = argparse.ArgumentParser(description="Automated Motif Discovery and Validation Pipeline.")
    
    # Input Files/Dirs
    parser.add_argument("--fasta_dir", required=True, help="Directory containing input FASTA files (e.g., cluster_*.fasta).")
    parser.add_argument("--validation_csv", required=True, help="Path to the validation dataset CSV file.")
    parser.add_argument("--master_fasta", required=True, help="Path to the master protein FASTA file (e.g., spider-silkome-database.v1.prot.fasta).")
    
    # Output Files/Dirs
    parser.add_argument("--output_dir", required=True, help="Main directory to store all outputs (STREME results, plots, final correlation table).")
    parser.add_argument("--results_file", default="combined_correlation_results.csv", help="Filename for the final combined correlation results table (within output_dir).")
    
    # STREME Parameters
    parser.add_argument("--conda_env", required=True, help="Name of the Conda environment where MEME Suite (streme) is installed.")
    parser.add_argument("--minw", type=int, default=3, help="Minimum motif width for STREME.")
    parser.add_argument("--maxw", type=int, default=15, help="Maximum motif width for STREME.")
    parser.add_argument("--thresh", type=float, default=0.05, help="E-value threshold for STREME.")
    
    # Validation Parameters
    parser.add_argument("--properties", nargs='+', default=["toughness", "young's_modulus", "tensile_strength", "strain_at_break"], 
                        help="List of property columns in validation_csv to correlate against. Default includes main properties without '_sd'.")
    
    # Optional Flags
    parser.add_argument("--generate_plots", action='store_true', help="Generate scatter plots for significant correlations.")
    parser.add_argument("--skip_streme", action='store_true', help="Skip running STREME and try to use existing streme.txt files.")


    args = parser.parse_args()

    # --- Preparations ---
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Check if conda env exists (basic check)
    try:
        subprocess.run(['conda', 'env', 'list'], check=True, capture_output=True, text=True)
    except FileNotFoundError:
        logging.error("Could not run 'conda'. Is Conda installed and in your PATH?")
        sys.exit(1)
        
    # Load validation data ONCE
    try:
        validation_df = load_validation_data(args.validation_csv, args.master_fasta)
    except Exception as e:
         logging.error(f"Failed to load validation data. Exiting. Error: {e}")
         sys.exit(1) # Exit if validation data fails to load

    # Find cluster FASTA files
    fasta_files = glob.glob(os.path.join(args.fasta_dir, "cluster_*.fasta"))
    if not fasta_files:
        logging.error(f"No 'cluster_*.fasta' files found in directory: {args.fasta_dir}")
        sys.exit(1)
        
    logging.info(f"Found {len(fasta_files)} cluster FASTA files to process.")

    all_cluster_results = []

    # --- Main Loop ---
    for fasta_file in sorted(fasta_files): # Sort for consistent order
        cluster_name = os.path.splitext(os.path.basename(fasta_file))[0]
        logging.info(f"--- Processing Cluster: {cluster_name} ---")
        
        streme_output_file = None
        streme_output_subdir = os.path.join(args.output_dir, f"streme_out_{cluster_name}")


        if not args.skip_streme:
            # Run STREME
            streme_output_file = run_streme(
                fasta_file, 
                args.output_dir, # Pass main output dir, run_streme creates subdir
                args.minw, 
                args.maxw, 
                args.thresh, 
                args.conda_env
            )
            if streme_output_file is None:
                logging.warning(f"STREME failed for {cluster_name}, skipping validation for this cluster.")
                continue # Skip to the next cluster
        else:
             # Try to find existing streme.txt if skipping run
             potential_streme_file = os.path.join(streme_output_subdir, 'streme.txt')
             if os.path.exists(potential_streme_file):
                 streme_output_file = potential_streme_file
                 logging.info(f"Skipping STREME run, using existing file: {streme_output_file}")
             else:
                 logging.warning(f"Skip STREME flag set, but existing file not found: {potential_streme_file}. Cannot process {cluster_name}.")
                 continue # Skip to next cluster


        # Extract Motifs
        motifs = extract_motifs(streme_output_file)
        
        if not motifs:
            logging.warning(f"No motifs extracted for {cluster_name}. Skipping validation.")
            continue # Skip to the next cluster
            
        # Count, Correlate, and FDR Correct
        cluster_corr_df = count_and_correlate(validation_df, motifs, args.properties, cluster_name)
        
        if not cluster_corr_df.empty:
            all_cluster_results.append(cluster_corr_df)

            # Visualize (optional)
            if args.generate_plots:
                plot_output_dir = os.path.join(args.output_dir, f"plots_{cluster_name}")
                # Pass the list of motifs that were actually used in correlation (valid_motifs implicitly handled inside)
                visualize_correlations(validation_df, cluster_corr_df, motifs, plot_output_dir)
        else:
             logging.warning(f"No correlation results generated for {cluster_name}.")


    # --- Final Aggregation and Output ---
    if not all_cluster_results:
        logging.warning("No correlation results were generated across any clusters.")
        print("Pipeline finished. No results to save.")
        sys.exit(0)
        
    final_results_df = pd.concat(all_cluster_results, ignore_index=True)
    
    # Sort final results for better readability (e.g., by significance, then p-value)
    final_results_df.sort_values(by=["significant_pearson", "p_adj_pearson"], ascending=[False, True], inplace=True)
    
    final_output_path = os.path.join(args.output_dir, args.results_file)
    try:
        final_results_df.to_csv(final_output_path, index=False)
        logging.info(f"Combined correlation results saved to: {final_output_path}")
    except Exception as e:
        logging.error(f"Failed to save final results to {final_output_path}: {e}")

    logging.info("--- Pipeline Finished ---")


if __name__ == "__main__":
    main()