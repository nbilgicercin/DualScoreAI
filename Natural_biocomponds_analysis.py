# -*- coding: utf-8 -*-
"""
Colab için Bileşik Analiz Kodu
"""

# Install required packages (Colab only)
import sys
import subprocess

def install_requirements():
    packages = [
        'biopython',
        'sentence-transformers',
        'transformers',
        'plotly',
        'tqdm',
        'kaleido'  # For static image export
    ]
    for package in packages:
        subprocess.check_call(['pip', 'install', package])

if 'google.colab' in sys.modules:
    install_requirements()

# Import required libraries
import os
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from Bio import Entrez
import datetime
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from sklearn.metrics.pairwise import cosine_similarity


# NCBI Entrez ayarları
Entrez.email = "beslinail@gmail.com"  # Entrez için gerekli
Entrez.api_key = "904f6767e648bdd1a3cdf0905622ac413508"  # Opsiyonel, varsa ekleyin

# Scopus API ayarları
SCOPUS_API_KEY = "cec280bbdd0256fef6dc2b07c4c71cca"
SCOPUS_INST_TOKEN = "7b83c7c133449c5b36af0f55d5a1a8d0"

PROPERTIES = {
    'antidiabetic': {
        'name': 'Antidiabetic',
        'keywords': ['antidiabetic', 'diabetes', 'glucose', 'insulin'],
        'reference_text': 'treatment of diabetes and blood glucose regulation'
    },
    'anticancer': {
        'name': 'Anticancer',
        'keywords': ['anticancer', 'antitumor', 'cytotoxic', 'antiproliferative'],
        'reference_text': 'treatment of cancer and tumor growth inhibition'
    },
    'antioxidant': {
        'name': 'Antioxidant',
        'keywords': ['antioxidant', 'oxidative stress', 'free radicals'],
        'reference_text': 'reduction of oxidative stress and free radicals'
    },
    'antiinflammatory': {
        'name': 'Antiinflammatory',
        'keywords': ['antiinflammatory', 'inflammation', 'cytokines'],
        'reference_text': 'reduction of inflammation and cytokine production'
    }
}

# Model yükleme
print("\nLoading models...")
sbert_model = SentenceTransformer('pritamdeka/S-PubMedBert-MS-MARCO')

# Load Biomedical NLI model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
stance_tokenizer = AutoTokenizer.from_pretrained("microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext")
stance_model = AutoModelForSequenceClassification.from_pretrained("microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext").to(device)

def calculate_semantic_similarity(compound, property_info):
    """Calculate semantic similarity between compound and property"""
    try:
        compound_text = f"{compound} {property_info['reference_text']}"
        property_text = " ".join(property_info['keywords'])
        
        embeddings = sbert_model.encode([compound_text, property_text])
        similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
        
        print(f"Semantic similarity score: {similarity:.3f}")
        return float(similarity)
    except Exception as e:
        print(f"Error calculating semantic similarity: {e}")
        return 0.0

def analyze_stance(compound, property_info):
    """Analyze stance between compound and property claim"""
    try:
        text = f"{compound} {property_info['reference_text']}"
        claim = f"{compound} has {property_info['name'].lower()} properties"
        
        inputs = stance_tokenizer(
            text,
            claim,
            return_tensors="pt",
            max_length=512,
            truncation=True,
            padding=True
        ).to(device)
        
        with torch.no_grad():
            outputs = stance_model(**inputs)
            probs = torch.softmax(outputs.logits, dim=1)
            prediction = torch.argmax(probs, dim=1).item()
        
        # BiomedNLP-PubMedBERT uses binary classification: 0 = contradiction, 1 = entailment
        probabilities = {
            'contradiction': float(probs[0][0]),
            'entailment': float(probs[0][1])
        }
        
        classes = ['contradiction', 'entailment']
        predicted_class = classes[prediction]
        
        # Convert binary classification to score
        score = float(probs[0][1])  # Entailment probability as score
        
        print(f"Stance analysis - Class: {predicted_class}, Score: {score:.3f}")
        return {
            'score': score,
            'class': predicted_class,
            'probabilities': probabilities
        }
    except Exception as e:
        print(f"Error analyzing stance: {e}")
        return {
            'score': 0.5,
            'class': 'neutral',
            'probabilities': {
                'contradiction': 0.5,
                'entailment': 0.5
            }
        }

def calculate_cooccurrence_score(compound, property_info):
    """Calculate co-occurrence score based on PubMed search"""
    try:
        # Search for compound
        compound_query = f"{compound}[Title/Abstract]"
        compound_handle = Entrez.esearch(db="pubmed", term=compound_query, retmax=0)
        compound_record = Entrez.read(compound_handle)
        compound_count = int(compound_record["Count"])
        
        # Search for compound + property keywords
        property_terms = " OR ".join([f"{kw}[Title/Abstract]" for kw in property_info['keywords']])
        combined_query = f"{compound_query} AND ({property_terms})"
        combined_handle = Entrez.esearch(db="pubmed", term=combined_query, retmax=0)
        combined_record = Entrez.read(combined_handle)
        combined_count = int(combined_record["Count"])
        
        # Calculate normalized score
        if compound_count == 0:
            score = 0.0
        else:
            score = min(combined_count / compound_count, 1.0)
        
        print(f"Co-occurrence analysis - Total papers: {compound_count}, Relevant papers: {combined_count}, Score: {score:.3f}")
        return {
            'score': float(score),
            'total_papers': compound_count,
            'relevant_papers': combined_count
        }
    except Exception as e:
        print(f"Error calculating co-occurrence score: {e}")
        return {
            'score': 0.0,
            'total_papers': 0,
            'relevant_papers': 0
        }

def calculate_clinical_trial_score(compound, property_info):
    """Calculate clinical trial score"""
    try:
        clinical_query = f"""({compound}[Title/Abstract]) AND 
            ({' OR '.join(property_info['keywords'])}[Title/Abstract]) AND 
            (Clinical Trial[Publication Type] OR Clinical Study[Publication Type])"""
        
        handle = Entrez.esearch(db="pubmed", term=clinical_query, retmax=1000)
        results = Entrez.read(handle)
        total_trials = int(results["Count"])
        
        current_year = datetime.datetime.now().year
        recent_query = f"{clinical_query} AND ({current_year-5}:{current_year}[Date - Publication])"
        handle = Entrez.esearch(db="pubmed", term=recent_query)
        recent_results = Entrez.read(handle)
        recent_trials = int(recent_results["Count"])
        
        if total_trials == 0:
            score = 0.0
        else:
            weighted_trials = (recent_trials * 1.5) + (total_trials - recent_trials)
            score = min(1.0, np.log1p(weighted_trials) / np.log1p(100))
        
        print(f"Clinical trials - Total: {total_trials}, Recent: {recent_trials}, Score: {score:.3f}")
        return float(score)
    except Exception as e:
        print(f"Error calculating clinical trial score: {e}")
        return 0.0

def analyze_compound_properties(compound):
    """Analyze compound for all properties"""
    print(f"\nAnalyzing {compound}...")
    results = {}
    
    for prop_id, prop_info in PROPERTIES.items():
        print(f"\nAnalyzing {prop_info['name']} property...")
        
        # Calculate individual scores
        semantic_score = calculate_semantic_similarity(compound, prop_info)
        cooccurrence_result = calculate_cooccurrence_score(compound, prop_info)
        clinical_score = calculate_clinical_trial_score(compound, prop_info)
        stance_result = analyze_stance(compound, prop_info)
        
        # Calculate final score
        final_score = (
            semantic_score * 0.3 +
            cooccurrence_result['score'] * 0.3 +
            clinical_score * 0.2 +
            stance_result['score'] * 0.2
        )
        
        print(f"\nFinal score for {prop_info['name']}: {final_score:.3f}")
        
        # Store results
        results[prop_id] = {
            'Final Score': float(final_score),
            'Semantic Similarity': float(semantic_score),
            'Co-occurrence Score': float(cooccurrence_result['score']),
            'Co-occurrence Total Papers': cooccurrence_result['total_papers'],
            'Co-occurrence Relevant Papers': cooccurrence_result['relevant_papers'],
            'Clinical Trial Score': float(clinical_score),
            'Stance Score': float(stance_result['score']),
            'Stance Class': stance_result['class'],
            'Stance Probabilities': stance_result['probabilities']
        }
    
    return results

def save_results_to_csv(results):
    """Save analysis results to CSV"""
    rows = []
    for result in results:
        compound_data = {
            'Compound': result['name'],
            'Group': result['group'],
            'antioxidant_final': result['scores']['antioxidant'],
            'antiinflammatory_final': result['scores']['antiinflammatory'],
            'antidiabetic_final': result['scores']['antidiabetic'],
            'anticancer_final': result['scores']['anticancer']
        }
        
        # Add detailed scores for each property
        for prop, details in result['details'].items():
            compound_data[f'{prop}_similarity'] = details['Semantic Similarity']
            compound_data[f'{prop}_pubmed_score'] = details['Clinical Trial Score']
            compound_data[f'{prop}_total_papers'] = details['Co-occurrence Total Papers']
            compound_data[f'{prop}_relevant_papers'] = details['Co-occurrence Relevant Papers']
        
        rows.append(compound_data)
    
    df = pd.DataFrame(rows)
    
    # Reorder columns for better readability
    column_order = [
        'Compound', 'Group',
        'antioxidant_final', 'antioxidant_similarity', 'antioxidant_pubmed_score', 'antioxidant_total_papers', 'antioxidant_relevant_papers',
        'antiinflammatory_final', 'antiinflammatory_similarity', 'antiinflammatory_pubmed_score', 'antiinflammatory_total_papers', 'antiinflammatory_relevant_papers',
        'antidiabetic_final', 'antidiabetic_similarity', 'antidiabetic_pubmed_score', 'antidiabetic_total_papers', 'antidiabetic_relevant_papers',
        'anticancer_final', 'anticancer_similarity', 'anticancer_pubmed_score', 'anticancer_total_papers', 'anticancer_relevant_papers'
    ]
    
    df = df[column_order]
    return df

def create_bubble_plot(df):
    """Create separate bubble plots for each compound group"""
    figs = []  # Store all figures
    
    # Define markers for each group
    group_markers = {
        'Reference': 'star',
        'Flavonoids': 'circle',
        'Terpenoids': 'diamond',
        'Carotenoids': 'square',
        'Organosulfur': 'triangle-up',
        'Alkaloids': 'pentagon'
    }
    
    # Create separate plot for each group
    for group in df['Group'].unique():
        fig = go.Figure()
        
        # Always add Metformin reference lines
        metformin_data = df[df['Compound'] == 'Metformin']
        metformin_x = metformin_data['antidiabetic_final'].iloc[0]
        metformin_y = metformin_data['anticancer_final'].iloc[0]
        
        # Add vertical reference line
        fig.add_shape(
            type="line",
            x0=metformin_x,
            x1=metformin_x,
            y0=0,
            y1=1,
            line=dict(
                color="red",
                width=2,
                dash="dash"
            )
        )
        
        # Add horizontal reference line
        fig.add_shape(
            type="line",
            x0=0,
            x1=1,
            y0=metformin_y,
            y1=metformin_y,
            line=dict(
                color="red",
                width=2,
                dash="dash"
            )
        )
        
        # Always show Metformin as reference
        fig.add_trace(go.Scatter(
            x=[metformin_x],
            y=[metformin_y],
            mode='markers+text',
            name='Metformin',
            text=['Metformin'],
            textposition='top center',
            marker=dict(
                size=50,  # Sabit boyut
                sizemode='area',
                color=metformin_data['antiinflammatory_final'].iloc[0],
                colorscale='RdYlBu',
                symbol='star',
                line=dict(color='black', width=1)
            ),
            legendgroup='Reference',
            customdata=[[
                metformin_data['antioxidant_final'].iloc[0],
                metformin_data['antiinflammatory_final'].iloc[0]
            ]],
            hovertemplate="<b>Metformin</b><br>" +
                         "Antidiabetic Score: %{x:.3f}<br>" +
                         "Anticancer Score: %{y:.3f}<br>" +
                         "Antioxidant Score: %{customdata[0]:.3f}<br>" +
                         "Anti-inflammatory Score: %{customdata[1]:.3f}<br>" +
                         "<extra></extra>"
        ))
        
        # Add compounds for current group
        group_data = df[df['Group'] == group]
        if not group_data.empty and group != 'Reference':
            # Add each compound separately for individual legend entries
            for idx, compound_data in group_data.iterrows():
                fig.add_trace(go.Scatter(
                    x=[compound_data['antidiabetic_final']],
                    y=[compound_data['anticancer_final']],
                    mode='markers+text',
                    name=compound_data['Compound'],
                    text=[compound_data['Compound']],
                    textposition='top center',
                    marker=dict(
                        size=50,  # Sabit boyut
                        sizemode='area',
                        color=compound_data['antiinflammatory_final'],
                        colorscale='RdYlBu',
                        showscale=True if idx == group_data.index[0] else False,
                        colorbar=dict(
                            title='Anti-inflammatory Score',
                            tickformat='.3f',
                            x=1.1
                        ) if idx == group_data.index[0] else None,
                        symbol=group_markers[group],
                        line=dict(color='black', width=1)
                    ),
                    legendgroup=group,
                    customdata=[[
                        compound_data['antioxidant_final'],
                        compound_data['antiinflammatory_final']
                    ]],
                    hovertemplate="<b>%{text}</b><br>" +
                                 "Antidiabetic Score: %{x:.3f}<br>" +
                                 "Anticancer Score: %{y:.3f}<br>" +
                                 "Antioxidant Score: %{customdata[0]:.3f}<br>" +
                                 "Anti-inflammatory Score: %{customdata[1]:.3f}<br>" +
                                 "<extra></extra>"
                ))
        
        # Update layout for each plot
        fig.update_layout(
            title=dict(
                text=f'Analysis of {group} Compounds vs Metformin',
                font=dict(size=24),
                x=0.5,
                xanchor='center'
            ),
            xaxis=dict(
                title=dict(
                    text='Antidiabetic Activity Score',
                    font=dict(size=16)
                ),
                range=[-0.1, 1.1],
                showgrid=True,
                gridwidth=1,
                gridcolor='LightGray',
                zeroline=True,
                zerolinewidth=2
            ),
            yaxis=dict(
                title=dict(
                    text='Anticancer Activity Score',
                    font=dict(size=16)
                ),
                range=[-0.1, 1.1],
                showgrid=True,
                gridwidth=1,
                gridcolor='LightGray',
                zeroline=True,
                zerolinewidth=2
            ),
            height=800,
            width=1000,
            showlegend=True,
            plot_bgcolor='white',
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01,
                bgcolor='rgba(255, 255, 255, 0.8)',
                groupclick="toggleitem"
            ),
            margin=dict(r=120, t=80, l=80, b=80)
        )
        
        figs.append(fig)
    
    return figs

# Test analysis with all compound groups
print("\nStarting analysis test...")

# Define compound groups
COMPOUND_GROUPS = {
    'Flavonoids': [
        'Quercetin', 'Kaempferol', 'Myricetin', 'Luteolin', 'Apigenin',
        'Naringenin', 'Hesperidin', 'Rutin', 'Baicalein', 'Chrysin',
        'Fisetin', 'Acacetin', 'EGCG', 'Curcumin', 'Resveratrol',
        'Catechin', 'Gallic acid', 'Caffeic acid', 'Ferulic acid',
        'Ellagic acid', 'Magnolol'
    ],
    'Terpenoids': [
        'Artemisinin', 'Ginsenosides', 'Thymoquinone', 'Citral',
        'Carvacrol', 'Geraniol', 'Thymol', 'Eugenol'
    ],
    'Carotenoids': [
        'Astaxanthin', 'Beta-carotene', 'Lycopene', 'Zeaxanthin',
        'Lutein', 'Fucoxanthin'
    ],
    'Organosulfur': ['Sulforaphane', 'Allicin'],
    'Alkaloids': ['Berberine', 'Piperine']
}

# First analyze Metformin (reference)
print("\nAnalyzing Metformin (reference compound)...")
metformin_results = analyze_compound_properties('Metformin')
all_results = [{
    'name': 'Metformin',
    'scores': {prop: details['Final Score'] for prop, details in metformin_results.items()},
    'details': metformin_results,
    'group': 'Reference'
}]

# Analyze each group
for group_name, compounds in COMPOUND_GROUPS.items():
    print(f"\nAnalyzing {group_name} group...")
    for compound in compounds:
        print(f"\nAnalyzing {compound}...")
        try:
            results = analyze_compound_properties(compound)
            all_results.append({
                'name': compound,
                'scores': {prop: details['Final Score'] for prop, details in results.items()},
                'details': results,
                'group': group_name
            })
        except Exception as e:
            print(f"Error analyzing {compound}: {e}")
            continue

# Create DataFrame and visualize
print("\nCreating visualizations...")
results_df = save_results_to_csv(all_results)

# Get current date for file naming
current_date = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

# Create output directory
if 'google.colab' in sys.modules:
    # Colab için
    from google.colab import files
    output_dir = "/content/analysis_results"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
else:
    # Yerel bilgisayar için
    output_dir = os.path.join(os.getcwd(), "analysis_results")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

# Save results to CSV with date
results_csv = os.path.join(output_dir, f"compound_analysis_results_{current_date}.csv")
results_df.to_csv(results_csv, index=False)
print(f"\nResults saved to: {results_csv}")

# Create and save bubble plots
figs = create_bubble_plot(results_df)
saved_files = []  # Kaydedilen dosyaları takip et

for idx, fig in enumerate(figs):
    group = results_df['Group'].unique()[idx]
    # Save as HTML for interactive viewing
    html_file = os.path.join(output_dir, f"{group}_analysis_{current_date}.html")
    fig.write_html(html_file)
    saved_files.append(html_file)
    print(f"Plot saved for {group}:")
    print(f"- Interactive HTML: {html_file}")
    # Display in notebook
    fig.show()

print("\nAnalysis completed successfully!")
print(f"\nAll results are saved in: {output_dir}")

# Colab'da çalışıyorsa dosyaları indir
if 'google.colab' in sys.modules:
    print("\nDownloading files to your computer...")
    # CSV dosyasını indir
    files.download(results_csv)
    # HTML dosyalarını indir
    for file in saved_files:
        files.download(file)
    print("\nAll files have been downloaded to your computer's Downloads folder.")
else:
    print("\nFiles saved:")
    print(f"1. CSV Results: {os.path.basename(results_csv)}")
    print("2. Interactive HTML plots:")
    for group in results_df['Group'].unique():
        print(f"   - {group}_analysis_{current_date}.html")
