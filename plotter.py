import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import os

# --- Configuration ---
OUTPUT_DIR = 'plot'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Consistent Color Map (For Parser-based plots)
PARSER_COLORS = {
    'CYK': '#1f77b4',       # Blue
    'Valiant': '#ff7f0e',   # Orange
    'Earley': '#2ca02c',    # Green
    'GLL': '#d62728',       # Red
    'RNGLR': '#9467bd',     # Purple
    'BRNGLR': '#8c564b',    # Brown
    'LL': '#e377c2',        # Pink
    'LR': '#7f7f7f',        # Gray
}

# Grammar Colors (For Grammar-based plots - Small Input)
GRAMMAR_COLORS = {
    'sexp': '#1f77b4', 'calc': '#ff7f0e', 'tinyc': '#2ca02c', 
    'json': '#d62728', 'ansi_c': '#9467bd', 'cpp': '#8c564b', 
    'java': '#e377c2', 'pascal': '#7f7f7f', 'ambiguous': '#bcbd22'
}
GRAMMAR_NAMES = {
    'sexp': 'S-Expression', 'calc': 'Calculator', 'tinyc': 'TinyC',
    'json': 'JSON', 'ansi_c': 'ANSI C', 'cpp': 'C++',
    'java': 'Java', 'pascal': 'Pascal', 'ambiguous': 'Stress Test'
}

# Standard 8 Grammars
GRAMMARS_GENERAL = [
    ('sexp', 'S-Expression'), ('calc', 'Calculator'), ('tinyc', 'TinyC'), 
    ('json', 'JSON'), ('ansi_c', 'ANSI C'), ('cpp', 'C++'), 
    ('java', 'Java'), ('pascal', 'Pascal')
]

GRAMMARS_LL_LR = [
    ('calc', 'Calculator (LL(1))'), ('json', 'JSON (LL(1))'), ('sexp', 'S-Expression (LL(1))'), ('tinypascal', 'SmallPascal (LL(1))')
]

GRAMMARS_LR = [
    ('tinyc', 'TinyC (LR(1))'), ('json', 'JSON (LR(1))')
]


def get_data_filepath(grammar, category, special_mappings=None):
    if special_mappings and grammar in special_mappings:
        return f"results/{special_mappings[grammar]}"
    if category == 'small':
        return f"results/benchmark_{grammar}_small.csv"
    elif category == 'large':
        return f"results/benchmark_{grammar}_large.csv"
    elif category == 'll_lr':
        return f"results/benchmark_{grammar}_with_lllr_large.csv"
    elif category == 'lr':
        return f"results/benchmark_lr_{grammar}_large.csv"
    return None

def plot_set(title_main, output_base, grammars, category, parsers_to_include, metric='time', special_json_earley=False, group_by_parser=False):
    """
    Generates a figure.
    group_by_parser: If True, creates subplots for each PARSER (showing all grammars). 
                     If False, creates subplots for each GRAMMAR (showing all parsers).
    """
    y_label = "Time (ms)" if metric == 'time' else "Peak Memory (MB)"
    
    if group_by_parser:
        # Layout: 1 row, N parsers columns
        cols = len(parsers_to_include)
        rows = 1
        figsize = (8 * cols, 6)
        unique_items = grammars # We iterate grammars inside
        
        fig, axes = plt.subplots(rows, cols, figsize=figsize, dpi=120)
        if rows * cols == 1: axes = [axes]
        axes = np.array(axes).flatten()
    else:
        # Mode: Subplots = Grammars, Lines = Parsers (Standard)
        # Dynamic cols: Use 4 or the number of grammars if less than 4
        cols = min(len(grammars), 4)
        rows = (len(grammars) + cols - 1) // cols
        
        # Adjust figsize width based on actual cols
        figsize = (6 * cols, 5 * rows)
        
        fig, axes = plt.subplots(rows, cols, figsize=figsize, dpi=120)
        if rows * cols == 1: axes = [axes]
        axes = np.array(axes).flatten()
    
    # Track min/max for memory zooming
    global_min_y = float('inf')
    global_max_y = float('-inf')

    # Data Collection Phase (to determine limits first, or just plot on fly)
    # We'll plot on fly and adjust if needed, or rely on autoscaling but with min-cut
    
    legend_handles = {}

    if group_by_parser:
        # Mode: Subplots = Parsers, Lines = Grammars
        for i, parser in enumerate(parsers_to_include):
            ax = axes[i]
            ax.set_title(f"Parser: {parser}", fontsize=16, fontweight='bold')
            
            has_data = False
            for grammar_code, grammar_name in grammars:
                fpath = get_data_filepath(grammar_code, category, None)
                if not os.path.exists(fpath): continue
                
                try:
                    df = pd.read_csv(fpath)
                    p_data = df[df['parser'] == parser]
                    p_data = p_data[p_data['success'] == True]
                    p_data = p_data[p_data['success'] != 'false']
                    
                    if p_data.empty: continue
                    
                    x = p_data['input_length']
                    if metric == 'time':
                        y = p_data['median_time_ns'] / 1e6
                    else:
                        y = p_data['peak_memory_bytes'] / 1e6
                        
                    has_data = True
                    global_min_y = min(global_min_y, y.min())
                    
                    color = GRAMMAR_COLORS.get(grammar_code, '#333')
                    label = grammar_name
                    
                    # Scatter
                    ax.scatter(x, y, color=color, alpha=0.5, s=20)
                    
                    # Polynomial Trend Line (Degree 2)
                    if len(x) > 1:
                        sorted_idx = x.argsort()
                        x_s, y_s = x.iloc[sorted_idx], y.iloc[sorted_idx]
                        z = np.polyfit(x_s, y_s, 2)
                        p = np.poly1d(z)
                        x_trend = np.linspace(x_s.min(), x_s.max(), 100)

                        # Store handle for legend
                        if label not in legend_handles:
                            line, = ax.plot(x_trend, p(x_trend), color=color, alpha=0.8, linewidth=2, label=label)
                            legend_handles[label] = line
                        else:
                            ax.plot(x_trend, p(x_trend), color=color, alpha=0.8, linewidth=2)
                        
                except Exception as e:
                    print(f"Skipping {grammar_code}/{parser}: {e}")

            ax.set_xlabel("Input Size (Bytes)")
            ax.set_ylabel(y_label)
            if metric == 'time': ax.set_ylim(bottom=0)
            
            # Memory Zoom
            if metric == 'memory' and has_data:
                ax.relim()
                ax.autoscale_view()
                ymin, ymax = ax.get_ylim()
                if ymin > 0:
                     ax.set_ylim(bottom=ymin * 0.95)


    else:
        # Mode: Subplots = Grammars, Lines = Parsers (Standard)
        # Mode: Subplots = Grammars, Lines = Parsers (Standard)
        # Dynamic cols: Use 4 or the number of grammars if less than 4
        cols = min(len(grammars), 4)
        rows = (len(grammars) + cols - 1) // cols
        
        is_large_plot = (cols == 4)
        subtitle_fs = 18 if is_large_plot else 12
        
        # Adjust figsize width based on actual cols
        figsize = (6 * cols, 5 * rows)
        
        fig, axes = plt.subplots(rows, cols, figsize=figsize, dpi=120)
        if rows * cols == 1: axes = [axes]
        axes = np.array(axes).flatten()

        for i, (grammar_code, grammar_name) in enumerate(grammars):
            ax = axes[i]
            
            mapping = None
            if category == 'll_lr' and grammar_code == 'tinypascal':
                mapping = {'tinypascal': 'benchmark_tinypascal_with_lllr_large.csv'}
            
            files = [get_data_filepath(grammar_code, category, mapping)]
            
            # Special JSON Earley handling
            if special_json_earley and grammar_code == 'json':
                files.append('results/benchmark_json_earley.csv')

            # Special LL/LR JSON Earley handling
            if category == 'll_lr' and grammar_code == 'json':
                files.append('results/benchmark_json_with_lllr_earley.csv')
            
            has_data = False
            subplot_min_y = float('inf')
            
            for fpath in files:
                if not fpath or not os.path.exists(fpath): continue
                try:
                    df = pd.read_csv(fpath)
                    subset = df[df['parser'].isin(parsers_to_include)]
                    
                    for parser in subset['parser'].unique():
                        p_data = subset[(subset['parser'] == parser) & (subset['success'] == True) & (subset['success'] != 'false')]
                        if p_data.empty: continue
                        
                        x = p_data['input_length']
                        if metric == 'time': y = p_data['median_time_ns'] / 1e6
                        else: y = p_data['peak_memory_bytes'] / 1e6
                        
                        has_data = True
                        subplot_min_y = min(subplot_min_y, y.min())
                        
                        color = PARSER_COLORS.get(parser, '#333')
                        
                        ax.scatter(x, y, color=color, alpha=0.5, s=20)
                        
                        # Trend line
                        if len(x) > 1:
                            sorted_idx = x.argsort()
                            x_s, y_s = x.iloc[sorted_idx], y.iloc[sorted_idx]
                            z = np.polyfit(x_s, y_s, 2)
                            p = np.poly1d(z)
                            x_trend = np.linspace(x_s.min(), x_s.max(), 100)
                            
                            if parser not in legend_handles:
                                line, = ax.plot(x_trend, p(x_trend), color=color, linewidth=2, label=parser)
                                legend_handles[parser] = line
                            else:
                                ax.plot(x_trend, p(x_trend), color=color, linewidth=2)
                                
                except Exception: pass
            
            ax.set_title(grammar_name, fontsize=subtitle_fs, fontweight='bold')
            ax.set_xlabel("Size (Bytes)")
            ax.set_ylabel(y_label)
            if metric == 'time': ax.set_ylim(bottom=0)
            if metric == 'memory' and has_data:
                # Heuristic: if min memory > 1MB, don't start at 0
                if subplot_min_y > 1:
                    ax.set_ylim(bottom=subplot_min_y * 0.9)
    
        # Hide unused subplots
        for j in range(len(grammars), len(axes)):
            axes[j].axis('off')

    # Unified Legend
    if legend_handles:
        # Create custom thicker handles for legend
        custom_handles = [Line2D([0], [0], color=h.get_color(), lw=4, label=l) for l, h in legend_handles.items()]
        
        is_large_plot = (cols == 4)
        if is_large_plot:
            fig.legend(handles=custom_handles, labels=legend_handles.keys(), 
                       loc='lower center', bbox_to_anchor=(0.5, 0.02), fontsize=22, 
                       frameon=True, fancybox=True, shadow=True, ncol=len(legend_handles))
            plt.tight_layout(rect=[0, 0.12, 1, 1])
        else:
            fig.legend(handles=custom_handles, labels=legend_handles.keys(), 
                       loc='center left', bbox_to_anchor=(0.86, 0.5), fontsize=14, 
                       frameon=True, fancybox=True, shadow=True)
            plt.tight_layout(rect=[0, 0, 0.85, 1])
    else:
        plt.tight_layout()
    
    save_path = os.path.join(OUTPUT_DIR, f"{output_base}_{metric}.pdf")
    plt.savefig(save_path, bbox_inches='tight')
    print(f"Saved {save_path}")
    plt.close(fig)

# Filter out simple grammars for Small Input plots to reduce clutter
GRAMMARS_SMALL_SIMPLE = [
    ('sexp', 'S-Expression'), ('calc', 'Calculator'), 
    ('tinyc', 'TinyC'), ('json', 'JSON')
]

def plot_memory_range_max_bar(grammars, category, parsers_to_include, output_base, special_json_earley=False):
    """
    Generates a grouped bar chart for Max Peak Memory for inputs < 5000 tokens.
    Shows the worst-case memory requirement for small-medium inputs.
    """
    TOKEN_LIMIT = 5000
    data = []
    
    for grammar_code, grammar_name in grammars:
        files = [get_data_filepath(grammar_code, category)]
        
        # Special JSON Earley handling
        if special_json_earley and grammar_code == 'json':
            files.append('results/benchmark_json_earley.csv')

        combined_df = pd.DataFrame()
        
        for fpath in files:
            if not fpath or not os.path.exists(fpath): continue
            try:
                df = pd.read_csv(fpath)
                # Filter for successful runs
                df = df[(df['success'] == True) & (df['success'] != 'false')]
                combined_df = pd.concat([combined_df, df], ignore_index=True)
            except Exception as e:
                print(f"Error reading {fpath}: {e}")
        
        if combined_df.empty: continue
            
        try:
            # Check if token_count exists
            if 'token_count' not in combined_df.columns:
                print(f"Skipping {grammar_code}: No token_count column")
                continue

            # Filtering: Runs <= TOKEN_LIMIT
            subset = combined_df[(combined_df['parser'].isin(parsers_to_include)) & 
                                 (combined_df['token_count'] <= TOKEN_LIMIT)]
            
            for parser in parsers_to_include:
                p_data = subset[subset['parser'] == parser]
                
                if p_data.empty: continue
                
                # Metric: Maximum Peak Memory observed in this range
                max_memory = p_data['peak_memory_bytes'].max() / 1e6 # MB
                
                data.append({
                    'Grammar': grammar_name,
                    'Parser': parser,
                    'Max Memory': max_memory
                })
                
        except Exception as e:
            print(f"Error processing {grammar_code}: {e}")

    if not data:
        print("No data found for memory plot.")
        return

    df_plot = pd.DataFrame(data)
    
    # Setup plot
    grammars_present = df_plot['Grammar'].unique()
    parsers_present = [p for p in parsers_to_include if p in df_plot['Parser'].unique()]
    
    n_grammars = len(grammars_present)
    n_parsers = len(parsers_present)
    
    x = np.arange(n_grammars) * 0.6  # Compact
    width = 0.3 / n_parsers  # Thin columns
    
    fig, ax = plt.subplots(figsize=(max(8, n_grammars * 1.0), 6), dpi=120)
    
    for i, parser in enumerate(parsers_present):
        parser_data = df_plot[df_plot['Parser'] == parser]
        
        y_values = []
        for grammar in grammars_present:
            val = parser_data[parser_data['Grammar'] == grammar]['Max Memory'].values
            y_values.append(val[0] if len(val) > 0 else 0)
            
        offset = (i - n_parsers / 2) * width + width / 2
        ax.bar(x + offset, y_values, width, label=parser, color=PARSER_COLORS.get(parser, '#333'))

    # ax.set_xlabel('Grammar', fontsize=12)
    ax.set_ylabel(f'Peak Memory', fontsize=12)
    # Linear Scale (Log scale removed per request)
    # ax.set_yscale('log')
    
    ax.set_xticks(x)
    ax.set_xticklabels(grammars_present, rotation=45, ha='right')
    
    # Custom Thicker Legend
    custom_handles = [Line2D([0], [0], color=PARSER_COLORS.get(p, '#333'), lw=4, label=p) for p in parsers_present]
    ax.legend(handles=custom_handles, loc='upper right', bbox_to_anchor=(1.15, 1))
    
    ax.grid(axis='y', linestyle='--', alpha=0.3)
    
    plt.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, f"{output_base}_max_limit_{TOKEN_LIMIT}_bar.pdf")
    plt.savefig(save_path, bbox_inches='tight')
    print(f"Saved {save_path}")
    plt.close(fig)

# --- Execution ---

# 1. Small Input (Grouped by Parser, Filtered Grammars)
plot_set(
    title_main="Small Input Benchmarks",
    output_base="small_cyk_valiant",
    grammars=GRAMMARS_SMALL_SIMPLE,
    category='small',
    parsers_to_include=['CYK', 'Valiant'],
    metric='time',
    group_by_parser=True
)
plot_set(
    title_main="Small Input Benchmarks",
    output_base="small_cyk_valiant",
    grammars=GRAMMARS_SMALL_SIMPLE,
    category='small',
    parsers_to_include=['CYK', 'Valiant'],
    metric='memory',
    group_by_parser=True
)


# 2. Large Input
plot_set(
    title_main="Large Input Benchmarks",
    output_base="large_general",
    grammars=GRAMMARS_GENERAL,
    category='large',
    parsers_to_include=['Earley', 'GLL', 'RNGLR', 'BRNGLR'],
    metric='time',
    special_json_earley=True
)
plot_set(
    title_main="Large Input Benchmarks",
    output_base="large_general",
    grammars=GRAMMARS_GENERAL,
    category='large',
    parsers_to_include=['Earley', 'GLL', 'RNGLR', 'BRNGLR'],
    metric='memory',
    special_json_earley=True
)


# 3. LL/LR Comparison
plot_set(
    title_main="LL/LR vs General",
    output_base="ll_lr_baseline",
    grammars=GRAMMARS_LL_LR,
    category='ll_lr',
    parsers_to_include=['Earley', 'GLL', 'RNGLR', 'BRNGLR', 'LL', 'LR'],
    metric='time'
)
plot_set(
    title_main="LL/LR vs General",
    output_base="ll_lr_baseline",
    grammars=GRAMMARS_LL_LR,
    category='ll_lr',
    parsers_to_include=['Earley', 'GLL', 'RNGLR', 'BRNGLR', 'LL', 'LR'],
    metric='memory'
)


# 4. LR Comparison
plot_set(
    title_main="LR Comparison",
    output_base="lr_comparison",
    grammars=GRAMMARS_LR,
    category='lr',
    parsers_to_include=['Earley', 'GLL', 'RNGLR', 'BRNGLR', 'LR', 'LL'],
    metric='time'
)
plot_set(
    title_main="LR Comparison",
    output_base="lr_comparison",
    grammars=GRAMMARS_LR,
    category='lr',
    parsers_to_include=['Earley', 'GLL', 'RNGLR', 'BRNGLR', 'LR', 'LL'],
    metric='memory'
)

# 5. Max Memory Limited Bar Chart
plot_memory_range_max_bar(
    grammars=GRAMMARS_GENERAL,
    category='large',
    parsers_to_include=['Earley', 'GLL', 'RNGLR', 'BRNGLR'],
    output_base="memory_max_limit",
    special_json_earley=True
)