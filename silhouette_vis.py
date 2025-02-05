#!/usr/bin/env python
"""
Visualize Silhouette Scores from /workspace/silhouette_scores.txt.

This script reads silhouette scores from a whitespace-delimited text file,
creates a Pandas DataFrame, and produces two visualizations:
  1. A heatmap showing silhouette scores (without cell annotations) for each epoch (rows) 
     versus candidate cluster numbers (columns).
  2. A line plot where each epoch’s silhouette scores (across candidate clusters) are plotted.

Usage:
    python visualize_silhouette.py --file /workspace/silhouette_scores.txt
"""

import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def parse_args():
    parser = argparse.ArgumentParser(description="Visualize Silhouette Scores")
    parser.add_argument('--file', type=str, default='/workspace/silhouette_scores.txt',
                        help='Path to the silhouette scores text file')
    parser.add_argument('--plot_output', type=str, default='silhouette_heatmap.png',
                        help='Filename to save the heatmap figure (optional)')
    parser.add_argument('--line_output', type=str, default='silhouette_lineplot.png',
                        help='Filename to save the line plot figure (optional)')
    parser.add_argument('--vmin', type=float, default=0.0, help='Minimum value for heatmap color scale')
    parser.add_argument('--vmax', type=float, default=0.6, help='Maximum value for heatmap color scale')
    return parser.parse_args()

def load_data(file_path):
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    # Read the file using a whitespace delimiter, skipping problematic lines.
    df = pd.read_csv(file_path, sep=r'\s+', header=0, index_col=0, engine='c', on_bad_lines='skip')
    
    # Convert all entries to numeric (float) and drop any rows/columns with NaN values.
    df = df.apply(pd.to_numeric, errors='coerce')
    df.dropna(axis=0, inplace=True)
    df.dropna(axis=1, inplace=True)
    
    # Convert index and columns to numeric types.
    df.index = df.index.astype(float)
    df.columns = df.columns.astype(float)
    
    return df

def plot_heatmap(df, output_file=None, vmin=0.0, vmax=0.6):
    plt.figure(figsize=(14, 10))
    # Do not annotate the cells (to keep the plot uncluttered)
    ax = sns.heatmap(df, annot=False, fmt=".4f", cmap="YlOrRd", vmin=vmin, vmax=vmax,
                     cbar_kws={'label': 'Silhouette Score'})
    
    # Increase font sizes and adjust tick parameters for better readability.
    ax.set_xlabel("Candidate Number of Clusters", fontsize=14)
    ax.set_ylabel("Epoch", fontsize=14)
    ax.set_title("Silhouette Scores Heatmap", fontsize=16, pad=20)
    ax.tick_params(axis='both', which='major', labelsize=12)
    plt.xticks(rotation=45)
    plt.tight_layout()
    if output_file:
        plt.savefig(output_file)
        print("Heatmap saved to:", output_file)
    plt.show()

def plot_lineplot(df, output_file=None):
    plt.figure(figsize=(14, 10))
    # Plot one line per epoch
    for epoch in df.index:
        plt.plot(df.columns, df.loc[epoch], marker='o', linewidth=2, label=f"Epoch {int(epoch)}")
    plt.xlabel("Candidate Number of Clusters", fontsize=14)
    plt.ylabel("Silhouette Score", fontsize=14)
    plt.title("Silhouette Score vs. Candidate Cluster Number", fontsize=16, pad=20)
    plt.xticks(df.columns, rotation=45, fontsize=12)
    plt.yticks(fontsize=12)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small', ncol=1)
    plt.tight_layout()
    if output_file:
        plt.savefig(output_file)
        print("Line plot saved to:", output_file)
    plt.show()

def main():
    args = parse_args()
    df = load_data(args.file)
    print("Data loaded successfully. DataFrame shape:", df.shape)
    print("DataFrame head:")
    print(df.head())
    
    plot_heatmap(df, args.plot_output, vmin=args.vmin, vmax=args.vmax)
    plot_lineplot(df, args.line_output)

if __name__ == '__main__':
    main()
