# -*- coding: utf-8 -*-
"""
Advanced Compound Analysis
- Dynamic Z-score calculation based on group size and paper counts
- Group-specific and Metformin-based comparisons
- Leader compound identification
"""

import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import matplotlib as mpl

def calculate_dynamic_zscores(df):
    """Calculate Z-scores with dynamic reference based on group size and paper counts"""
    properties = ['antioxidant', 'antiinflammatory', 'antidiabetic', 'anticancer']
    min_group_size = 5  # Minimum size for group-based Z-scores
    
    # Store results
    all_results = pd.DataFrame()
    
    # Get Metformin scores for reference
    metformin = df[df['Group'] == 'Reference'].iloc[0]
    
    # First calculate Metformin's Z-scores against all compounds
    metformin_results = []
    for prop in properties:
        score_col = f'{prop}_final'
        papers_col = f'{prop}_total_papers'
        
        # Get all scores excluding Metformin
        all_scores = df[df['Group'] != 'Reference'][score_col]
        all_papers = df[df['Group'] != 'Reference'][papers_col]
        
        # Calculate Z-score for Metformin
        metformin_z = (metformin[score_col] - all_scores.mean()) / all_scores.std()
        
        metformin_results.append({
            'Compound': 'Metformin',
            'Group': 'Reference',
            'Property': prop,
            'Score': metformin[score_col],
            'Papers': metformin[papers_col],
            'Z_score': metformin_z
        })
    
    # Add Metformin results
    all_results = pd.concat([all_results, pd.DataFrame(metformin_results)])
    
    # Process each group
    for group in df['Group'].unique():
        if group == 'Reference':
            continue
            
        group_df = df[df['Group'] == group]
        group_size = len(group_df)
        
        for prop in properties:
            score_col = f'{prop}_final'
            papers_col = f'{prop}_total_papers'
            
            scores = group_df[score_col]
            papers = group_df[papers_col]
            
            # Normalize paper counts within group
            paper_weights = papers / papers.max()
            
            # Calculate weighted scores using paper counts
            weighted_scores = scores * (1 + paper_weights)
            
            # Calculate Z-scores differently based on group size
            if group_size >= min_group_size:
                # Use group-based Z-scores for larger groups
                z_scores = stats.zscore(weighted_scores)
                reference_score = weighted_scores.mean()
            else:
                # Use Metformin as reference for smaller groups
                reference_score = metformin[score_col]
                z_scores = (weighted_scores - reference_score) / weighted_scores.std()
            
            # Create temporary DataFrame
            temp_df = pd.DataFrame({
                'Compound': group_df['Compound'],
                'Group': group,
                'Property': prop,
                'Score': scores.values,
                'Papers': papers.values,
                'Paper_Weight': paper_weights.values,
                'Weighted_Score': weighted_scores.values,
                'Z_score': z_scores,
                'Reference_Score': reference_score,
                'Group_Size': group_size
            })
            
            all_results = pd.concat([all_results, temp_df])
    
    return all_results

def create_dynamic_plot(df, results_df):
    """Create 2x2 subplot with dynamic Z-score visualization"""
    properties = ['antioxidant', 'antiinflammatory', 'antidiabetic', 'anticancer']
    property_names = ['Antioxidant', 'Anti-inflammatory', 'Antidiabetic', 'Anticancer']
    
    # Setup the figure with Times New Roman font
    plt.rcParams['font.family'] = 'Times New Roman'
    
    # Create figure
    fig = plt.figure(figsize=(16, 12))
    gs = GridSpec(2, 2)
    
    # Define markers and colors with larger sizes for legend
    markers = {
        'Reference': '*',      # star
        'Flavonoids': 'o',    # circle
        'Terpenoids': 'D',    # diamond
        'Carotenoids': 's',   # square
        'Organosulfur': '^',  # triangle-up
        'Alkaloids': 'p'      # pentagon
    }
    
    colors = {
        'Reference': '#e41a1c',    # red
        'Flavonoids': '#377eb8',   # blue
        'Terpenoids': '#4daf4a',   # green
        'Carotenoids': '#ff7f00',  # orange
        'Organosulfur': '#984ea3', # purple
        'Alkaloids': '#a65628'     # brown
    }
    
    # Get Metformin reference values
    metformin_data = df[df['Group'] == 'Reference'].iloc[0]
    
    # Create subplots
    for idx, (prop, prop_name) in enumerate(zip(properties, property_names)):
        ax = fig.add_subplot(gs[idx // 2, idx % 2])
        
        # First plot Metformin as a red star
        metformin_score = metformin_data[f'{prop}_final']
        # Calculate Metformin's Z-score relative to all compounds
        all_scores = results_df[results_df['Property'] == prop]['Score']
        metformin_z = (metformin_score - all_scores.mean()) / all_scores.std()
        
        # Plot Metformin point
        ax.scatter(metformin_score, metformin_z,
                  c='red', marker='*', s=200, alpha=0.7,
                  label='Reference' if idx == 0 else "")  # Metformin yerine Reference
        
        # Add Metformin label
        ax.annotate('Metformin',  # Etiket hala Metformin
                   (metformin_score, metformin_z),
                   xytext=(5, 5),
                   textcoords='offset points',
                   alpha=0.7,
                   fontsize=14,
                   fontfamily='Times New Roman')
        
        # Plot each group
        for group in df['Group'].unique():
            if group == 'Reference':  # Skip Reference group as we already plotted Metformin
                continue
                
            group_data = results_df[
                (results_df['Group'] == group) & 
                (results_df['Property'] == prop)
            ]
            
            if len(group_data) == 0:
                continue
            
            # Plot points with size based on paper count
            max_papers = group_data['Papers'].max()
            sizes = 50 + (group_data['Papers'] / max_papers) * 150
            
            scatter = ax.scatter(
                group_data['Score'],
                group_data['Z_score'],
                c=colors[group],
                marker=markers[group],
                s=sizes,
                alpha=0.7,
                label=group if idx == 0 else ""
            )
            
            # Add labels for leaders
            for _, row in group_data.iterrows():
                if row['Z_score'] > 1:
                    ax.annotate(
                        row['Compound'],
                        (row['Score'], row['Z_score']),
                        xytext=(5, 5),
                        textcoords='offset points',
                        fontsize=14,
                        alpha=0.7,
                        fontfamily='Times New Roman'
                    )
        
        # Add Z=1 threshold line
        ax.axhline(y=1, color='red', linestyle='--', alpha=0.3, label='Z = 1' if idx == 0 else "")
        
        # Customize subplot
        ax.set_title(prop_name, fontsize=18, fontfamily='Times New Roman')
        ax.set_xlabel('Activity Score', fontsize=16, fontfamily='Times New Roman')
        ax.set_ylabel('Z-Score', fontsize=16, fontfamily='Times New Roman')
        ax.grid(True, linestyle='--', alpha=0.3)
        ax.tick_params(axis='both', labelsize=14)
        
        # Set axis limits
        ax.set_xlim(0, 1)
        y_max = max(3, results_df[results_df['Property'] == prop]['Z_score'].max() * 1.1)
        y_min = min(-2, results_df[results_df['Property'] == prop]['Z_score'].min() * 1.1)
        ax.set_ylim(y_min, y_max)
    
    # Adjust legend
    handles, labels = fig.axes[0].get_legend_handles_labels()
    
    # Increase legend marker sizes
    for handle in handles:
        if isinstance(handle, mpl.collections.PathCollection):
            handle.set_sizes([150])  # Büyük legend sembolleri
    
    # Legend'ı alta yatay olarak yerleştir
    leg = fig.legend(handles, labels, 
                    loc='center',  # Merkeze hizala
                    bbox_to_anchor=(0.5, 0.02),  # Alta yerleştir
                    fontsize=16,
                    frameon=True,
                    edgecolor='black',
                    facecolor='white',
                    bbox_transform=fig.transFigure,
                    prop={'family': 'Times New Roman', 'size': 16},
                    markerscale=2.0,  # Legend sembol boyutunu 2 kat artır
                    borderpad=1,  # Legend iç padding
                    labelspacing=1.5,  # Legend etiketleri arası boşluk
                    handletextpad=1.5,  # Sembol ve yazı arası boşluk
                    ncol=len(labels))  # Yatay düzen için tüm etiketleri tek satırda göster
    
    # Alt boşluğu artır
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15)  # Legend için alt boşluk
    
    return fig

def main():
    """Main function to run the analysis"""
    # Read CSV file
    csv_path = r"C:\Users\NAİL BESLİ\Desktop\BioBERT\scopus_sonuç\compound_analysis_results_20250409_090801.csv"
    print(f"\nReading CSV file: {csv_path}")
    df = pd.read_csv(csv_path)
    
    print("\nCalculating dynamic Z-scores...")
    results_df = calculate_dynamic_zscores(df)
    
    # Z-skorlarını orijinal CSV'ye ekle
    z_scores = {}
    properties = ['antioxidant', 'antiinflammatory', 'antidiabetic', 'anticancer']
    
    # Her bileşik için Z-skorlarını topla
    for _, row in results_df.iterrows():
        compound = row['Compound']
        property_name = row['Property']
        if compound not in z_scores:
            z_scores[compound] = {}
        z_scores[compound][f'{property_name}_zscore'] = row['Z_score']
    
    # Yeni sütunları DataFrame'e ekle
    for prop in properties:
        df[f'{prop}_zscore'] = df['Compound'].map(lambda x: z_scores[x][f'{prop}_zscore'])
    
    # Z-skorlu CSV'yi kaydet
    output_csv = csv_path.replace('.csv', '_with_zscores.csv')
    df.to_csv(output_csv, index=False)
    print(f"\nSaved CSV with Z-scores: {output_csv}")
    
    print("\nCreating dynamic visualization...")
    dynamic_fig = create_dynamic_plot(df, results_df)
    dynamic_png_path = csv_path.replace('.csv', '_dynamic_analysis.png')
    print(f"Saving dynamic visualization: {dynamic_png_path}")
    plt.savefig(dynamic_png_path, dpi=600, bbox_inches='tight', pad_inches=0.2)
    plt.close()

if __name__ == "__main__":
    main()
