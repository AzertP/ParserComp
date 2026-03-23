import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import os

# --- Configuration ---
OUTPUT_DIR = 'plot'
os.makedirs(OUTPUT_DIR, exist_ok=True)

PARSER_COLORS = {
    'Earley': '#2ca02c',    # Green
    'GLL': '#d62728',       # Red
    'RNGLR': '#9467bd',     # Purple
    'BRNGLR': '#8c564b',    # Brown
    'LL':     '#1f77b4',    # Blue
    'LR':     '#ff7f0e',    # Orange
    'CYK': '#ff00ff', #some different color, maybe pink
}

PARSERS = ['Earley', 'GLL', 'RNGLR', 'BRNGLR']

GRAMMARS = [
    ('ansi_c', 'ANSI C'),
    ('calc', 'Calculator'),
    ('cpp', 'C++'),
    ('css', 'CSS'),
    ('html', 'HTML'),
    ('java', 'Java'),
    ('json', 'JSON'),
    ('pascal', 'Pascal'),
    ('tinyc', 'TinyC'),
    ('sexp', 'S-Expression'),
    ('shell', 'Shell'),
    ('sql', 'SQL'),
]

# Map grammar codes to their CSV file(s).
# JSON is special: Earley data lives in a separate file.
GRAMMAR_FILES = {
    'ansi_c': ['results/benchmark_ansi_c_large.csv'],
    'calc':   ['results/benchmark_calc_large.csv'],
    'cpp':    ['results/benchmark_cpp_large.csv'],
    'css':    ['results/benchmark_css_large.csv'],
    'html':   ['results/benchmark_html.csv'],
    'java':   ['results/benchmark_java_large.csv'],
    'json':   ['results/benchmark_json_large.csv', 'results/benchmark_json_earley.csv'],
    'pascal': ['results/benchmark_pascal_large.csv'],
    'tinyc':  ['results/benchmark_tinyc_large.csv'],
    'sexp':   ['results/benchmark_sexp_large.csv'],
    'shell':  ['results/benchmark_shell.csv'],
    'sql':    ['results/benchmark_sql.csv'],
}

# LL(1)/LLLR grammars – files contain GLL, RNGLR, BRNGLR, LL, LR.
# JSON needs a separate Earley file stitched in.
LL_GRAMMARS = [
    ('json',       'JSON'),
    ('calc',       'Calculator'),
    ('sexp',       'S-Expression'),
    ('tinypascal', 'SmallPascal'),
]

LL_GRAMMAR_FILES = {
    'json':       ['results/benchmark_json_with_lllr_large.csv',
                   'results/benchmark_json_with_lllr_earley.csv'],
    'calc':       ['results/benchmark_calc_with_lllr_large.csv'],
    'sexp':       ['results/benchmark_sexp_with_lllr_large.csv'],
    'tinypascal': ['results/benchmark_tinypascal_with_lllr_large.csv'],
}

# LR grammars – files contain Earley, GLL, RNGLR, BRNGLR, LR.
LR_GRAMMARS = [
    ('json',  'JSON'),
    ('tinyc', 'TinyC'),
]

LR_GRAMMAR_FILES = {
    'json':  ['results/benchmark_lr_json_large.csv'],
    'tinyc': ['results/benchmark_lr_tinyc_large.csv'],
}

# Ambiguous / unambiguous expression grammars for comparison.
# CYK and Valiant are excluded from the plot.
AMBI_GRAMMARS = [
    ('expr',       'Expr (unambiguous)', 'results/benchmark_expr.csv'),
    ('expr_ambi',  'Expr (ambiguous)',   'results/benchmark_expr_ambi.csv'),
    ('bool',       'Bool (ambiguous)',   'results/benchmark_bool.csv'),
]
# Parsers to render – CYK and Valiant are included for the relative MAD table.
AMBI_PARSERS = ['Earley', 'GLL', 'RNGLR', 'BRNGLR', 'LR', 'CYK']


def load_grammar_data(grammar_code):
    """Load and concatenate all CSV files for a grammar, filtering for successful runs."""
    files = GRAMMAR_FILES.get(grammar_code, [])
    frames = []
    for fpath in files:
        if os.path.exists(fpath):
            try:
                df = pd.read_csv(fpath)
                df = df[df['success'] == True]
                df = df[df['success'] != 'false']
                frames.append(df)
            except Exception as e:
                print(f"  Warning: could not read {fpath}: {e}")
    if frames:
        return pd.concat(frames, ignore_index=True)
    return pd.DataFrame()


def plot_trend(ax, x, y, color, label, legend_handles, marker='o', degree=2, linestyle='-'):
    """
    Scatter plot + polynomial trend line constrained to pass through origin.
    """
    # Scatter
    ax.scatter(x, y, color=color, alpha=0.5, s=30, marker=marker)

    if len(x) < 3:
        return

    sorted_idx = x.argsort()
    x_s = x.iloc[sorted_idx].values
    y_s = y.iloc[sorted_idx].values

    # Start trend from 0 to visually emphasize the pass-through-origin requirement
    x_trend = np.linspace(0, x_s.max(), 200)

    # Extract Vandermonde matrix without the constant column [x^degree, ..., x^1]
    V = np.vander(x_s, degree + 1)[:, :-1]
    w, _, _, _ = np.linalg.lstsq(V, y_s, rcond=None)

    V_trend = np.vander(x_trend, degree + 1)[:, :-1]
    y_trend = V_trend @ w

    # Ensure the curve doesn't dip below zero algebraically
    y_trend = np.maximum(y_trend, 0)

    # Draw trend line and track legend handle
    if label not in legend_handles:
        line, = ax.plot(x_trend, y_trend, color=color, linewidth=2, linestyle=linestyle, label=label)
        legend_handles[label] = line
    else:
        ax.plot(x_trend, y_trend, color=color, linewidth=2, linestyle=linestyle)


def plot_all():
    cols = 4
    rows = (len(GRAMMARS) + cols - 1) // cols  # 3 rows for 12 grammars
    figsize = (6 * cols, 5 * rows)

    fig, axes = plt.subplots(rows, cols, figsize=figsize, dpi=120)
    axes = np.array(axes).flatten()

    legend_handles = {}

    for i, (grammar_code, grammar_name) in enumerate(GRAMMARS):
        ax = axes[i]
        df = load_grammar_data(grammar_code)

        if df.empty:
            ax.set_title(grammar_name, fontsize=18, fontweight='bold')
            ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
            continue

        subset = df[df['parser'].isin(PARSERS)]

        for parser in PARSERS:
            p_data = subset[subset['parser'] == parser]
            if p_data.empty:
                continue

            x = p_data['input_length']
            y = p_data['median_time_ns'] / 1e6  # ms

            color = PARSER_COLORS.get(parser, '#333')
            plot_trend(ax, x, y, color, parser, legend_handles)

        ax.set_title(grammar_name, fontsize=18, fontweight='bold')
        ax.set_xlabel("Size (Bytes)")
        ax.set_ylabel("Time (ms)")
        ax.set_ylim(bottom=0)
        if grammar_code == 'json':
            ax.set_ylim(top=500)

    # Hide unused subplots
    for j in range(len(GRAMMARS), len(axes)):
        axes[j].axis('off')

    # Unified legend at bottom
    if legend_handles:
        custom_handles = [
            Line2D([0], [0], color=h.get_color(), lw=4, label=l)
            for l, h in legend_handles.items()
        ]
        fig.legend(
            handles=custom_handles,
            labels=legend_handles.keys(),
            loc='lower center',
            bbox_to_anchor=(0.5, 0.02),
            fontsize=22,
            frameon=True,
            fancybox=True,
            shadow=True,
            ncol=len(legend_handles),
        )
        plt.tight_layout(rect=[0, 0.08, 1, 1])
    else:
        plt.tight_layout()

    save_path = os.path.join(OUTPUT_DIR, 'general_parsers_time.png')
    plt.savefig(save_path, bbox_inches='tight')
    print(f"Saved {save_path}")
    plt.close(fig)


def plot_ll_comparison():
    """One subplot per LL-parseable grammar comparing all parsers including LL."""
    cols = len(LL_GRAMMARS)
    fig, axes = plt.subplots(1, cols, figsize=(6 * cols, 5), dpi=120)
    axes = np.array(axes).flatten()

    # Parsers present in these files (no Earley for non-JSON grammars)
    ll_parsers = ['GLL', 'RNGLR', 'BRNGLR', 'LL', 'LR']
    legend_handles = {}

    for i, (grammar_code, grammar_name) in enumerate(LL_GRAMMARS):
        ax = axes[i]
        files = LL_GRAMMAR_FILES.get(grammar_code, [])
        frames = []
        for fpath in files:
            if os.path.exists(fpath):
                try:
                    df = pd.read_csv(fpath)
                    df = df[df['success'] == True]
                    frames.append(df)
                except Exception as e:
                    print(f"  Warning: could not read {fpath}: {e}")

        if not frames:
            ax.set_title(grammar_name, fontsize=18, fontweight='bold')
            ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
            continue

        df = pd.concat(frames, ignore_index=True)
        available = [p for p in ll_parsers if p in df['parser'].values]

        for parser in available:
            p_data = df[df['parser'] == parser]
            x = p_data['input_length']
            y = p_data['median_time_ns'] / 1e6
            color = PARSER_COLORS.get(parser, '#333')
            plot_trend(ax, x, y, color, parser, legend_handles)

        ax.set_title(grammar_name, fontsize=18, fontweight='bold')
        ax.set_xlabel("Size (Bytes)")
        ax.set_ylabel("Time (ms)")
        ax.set_ylim(bottom=0)

    if legend_handles:
        custom_handles = [
            Line2D([0], [0], color=h.get_color(), lw=4, label=l)
            for l, h in legend_handles.items()
        ]
        fig.legend(
            handles=custom_handles,
            labels=legend_handles.keys(),
            loc='lower center',
            bbox_to_anchor=(0.5, 0.02),
            fontsize=22,
            frameon=True,
            fancybox=True,
            shadow=True,
            ncol=len(legend_handles),
        )
        plt.tight_layout(rect=[0, 0.10, 1, 1])
    else:
        plt.tight_layout()

    save_path = os.path.join(OUTPUT_DIR, 'll_comparison.png')
    plt.savefig(save_path, bbox_inches='tight')
    print(f"Saved {save_path}")
    plt.close(fig)


def plot_lr_comparison():
    """One subplot per LR-parseable grammar comparing all parsers including LR."""
    cols = len(LR_GRAMMARS)
    fig, axes = plt.subplots(1, cols, figsize=(6 * cols, 5), dpi=120)
    axes = np.array(axes).flatten()

    lr_parsers = ['Earley', 'GLL', 'RNGLR', 'BRNGLR', 'LR']
    legend_handles = {}

    for i, (grammar_code, grammar_name) in enumerate(LR_GRAMMARS):
        ax = axes[i]
        files = LR_GRAMMAR_FILES.get(grammar_code, [])
        frames = []
        for fpath in files:
            if os.path.exists(fpath):
                try:
                    df = pd.read_csv(fpath)
                    df = df[df['success'] == True]
                    frames.append(df)
                except Exception as e:
                    print(f"  Warning: could not read {fpath}: {e}")

        if not frames:
            ax.set_title(grammar_name, fontsize=18, fontweight='bold')
            ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
            continue

        df = pd.concat(frames, ignore_index=True)
        available = [p for p in lr_parsers if p in df['parser'].values]

        for parser in available:
            p_data = df[df['parser'] == parser]
            x = p_data['input_length']
            y = p_data['median_time_ns'] / 1e6
            color = PARSER_COLORS.get(parser, '#333')
            plot_trend(ax, x, y, color, parser, legend_handles)

        ax.set_title(grammar_name, fontsize=18, fontweight='bold')
        ax.set_xlabel("Size (Bytes)")
        ax.set_ylabel("Time (ms)")
        ax.set_ylim(bottom=0)

    if legend_handles:
        custom_handles = [
            Line2D([0], [0], color=h.get_color(), lw=4, label=l)
            for l, h in legend_handles.items()
        ]
        fig.legend(
            handles=custom_handles,
            labels=legend_handles.keys(),
            loc='lower center',
            bbox_to_anchor=(0.5, 0.02),
            fontsize=22,
            frameon=True,
            fancybox=True,
            shadow=True,
            ncol=len(legend_handles),
        )
        plt.tight_layout(rect=[0, 0.10, 1, 1])
    else:
        plt.tight_layout()

    save_path = os.path.join(OUTPUT_DIR, 'lr_comparison.png')
    plt.savefig(save_path, bbox_inches='tight')
    print(f"Saved {save_path}")
    plt.close(fig)


def plot_ambiguous_comparison():
    """Side-by-side comparison of expr (unambiguous) vs expr_ambi and bool (ambiguous).

    CYK and Valiant are excluded.  LR appears only in the unambiguous subplot,
    making the contrast between grammar classes immediately visible.
    """
    cols = len(AMBI_GRAMMARS)
    fig, axes = plt.subplots(1, cols, figsize=(6 * cols, 5), dpi=120)
    axes = np.array(axes).flatten()

    legend_handles = {}

    for i, (code, name, fpath) in enumerate(AMBI_GRAMMARS):
        ax = axes[i]
        if not os.path.exists(fpath):
            ax.set_title(name, fontsize=18, fontweight='bold')
            ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
            continue

        try:
            df = pd.read_csv(fpath)
            df = df[df['success'] == True]
        except Exception as e:
            print(f"  Warning: could not read {fpath}: {e}")
            continue

        # Only keep the parsers we care about (drops CYK & Valiant automatically)
        available = [p for p in AMBI_PARSERS if p in df['parser'].values]

        for parser in available:
            p_data = df[df['parser'] == parser]
            x = p_data['input_length']
            y = p_data['median_time_ns'] / 1e6
            color = PARSER_COLORS.get(parser, '#333')
            plot_trend(ax, x, y, color, parser, legend_handles)

        ax.set_title(name, fontsize=18, fontweight='bold')
        ax.set_xlabel("Size (Bytes)")
        ax.set_ylabel("Time (ms)")
        ax.set_ylim(bottom=0)
        if code == 'expr_ambi':
            ax.set_ylim(top=400)

    if legend_handles:
        custom_handles = [
            Line2D([0], [0], color=h.get_color(), lw=4, label=l)
            for l, h in legend_handles.items()
        ]
        fig.legend(
            handles=custom_handles,
            labels=legend_handles.keys(),
            loc='lower center',
            bbox_to_anchor=(0.5, 0.02),
            fontsize=22,
            frameon=True,
            fancybox=True,
            shadow=True,
            ncol=len(legend_handles),
        )
        plt.tight_layout(rect=[0, 0.10, 1, 1])
    else:
        plt.tight_layout()

    save_path = os.path.join(OUTPUT_DIR, 'ambiguous_comparison.png')
    plt.savefig(save_path, bbox_inches='tight')
    print(f"Saved {save_path}")
    plt.close(fig)


def plot_cyk_valiant_comparison():
    """Side-by-side comparison of CYK and Valiant across multiple grammars."""
    parsers = ['CYK', 'Valiant']
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=120)
    
    cv_grammars = [
        ('bool',      'Bool (ambiguous)',   'results/benchmark_bool.csv'),
        ('expr_ambi', 'Expr (ambiguous)',   'results/benchmark_expr_ambi.csv'),
        ('calc',      'Calculator',         'results/benchmark_calc_small.csv'),
        ('json',      'JSON',               'results/benchmark_json_small.csv'),
        ('sexp',      'S-Expression',       'results/benchmark_sexp_small.csv'),
        ('tinyc',     'TinyC',              'results/benchmark_tinyc_small.csv'),
    ]
    
    import matplotlib.cm as cm
    colors = cm.tab10.colors
    markers = ['o', 's', '^', 'D', 'v', 'P']
    global_legend_handles = {}

    for i, parser in enumerate(parsers):
        ax = axes[i]
        legend_handles = {}
        for j, (code, name, fpath) in enumerate(cv_grammars):
            if not os.path.exists(fpath):
                continue
            try:
                df = pd.read_csv(fpath)
                df = df[df['success'] == True]
            except Exception:
                continue

            p_data = df[df['parser'] == parser]
            if p_data.empty:
                continue

            x = p_data['input_length']
            y = p_data['median_time_ns'] / 1e6
            
            color = colors[j % len(colors)]
            marker = markers[j % len(markers)]
            # CYK and Valiant are theoretically cubic/almost-cubic
            # Use dashed lines to distinguish this graph from parser-based ones
            plot_trend(ax, x, y, color, name, legend_handles, marker=marker, degree=3, linestyle='--')
            global_legend_handles.update(legend_handles)

        # Make the plot visually distinct
        ax.set_facecolor('#f4f6f9')  # Subtle light blue/grey background
        ax.grid(True, linestyle=':', alpha=0.7, color='gray')
        
        ax.set_title(f"{parser}", fontsize=18, fontweight='bold')
        ax.set_xlabel("Size (Bytes)")
        ax.set_ylabel("Time (ms)")
        ax.set_ylim(bottom=0)
        ax.set_xlim(left=0)
        
        if parser == 'CYK':
            ax.set_xlim(right=160)
            ax.set_ylim(top=120)
        elif parser == 'Valiant':
            ax.set_xlim(right=40)
            ax.set_ylim(top=3100)

    if global_legend_handles:
        custom_handles = [
            Line2D([0], [0], color=h.get_color(), lw=4, label=l)
            for l, h in global_legend_handles.items()
        ]
        fig.legend(
            handles=custom_handles,
            labels=global_legend_handles.keys(),
            loc='lower center',
            bbox_to_anchor=(0.5, -0.05),
            fontsize=14,
            frameon=True,
            fancybox=True,
            shadow=True,
            ncol=3,
            title="Grammars",
            title_fontsize=15
        )
        plt.tight_layout(rect=[0, 0.15, 1, 1])
    else:
        plt.tight_layout()

    save_path = os.path.join(OUTPUT_DIR, 'cyk_valiant_comparison.png')
    plt.savefig(save_path, bbox_inches='tight')
    print(f"Saved {save_path}")
    plt.close(fig)


TABLE_DIR = 'plot'
os.makedirs(TABLE_DIR, exist_ok=True)

def _format_cell(df, parser):
    """Return average relative MAD (%) for a given parser across all input sizes, or '—' if absent."""
    p_data = df[df['parser'] == parser]
    if p_data.empty:
        return '—'

    # Filter out median times exactly 0 to avoid division by zero
    valid = p_data[p_data['median_time_ns'] > 0]
    if valid.empty:
        return '0.00%'

    # Compute relative MAD per row, then average
    rel_mad = (valid['mad_ns'] / valid['median_time_ns']) * 100
    avg_rel_mad = rel_mad.mean()
    return f'{avg_rel_mad:.2f}%'


def _write_group_table(f, group_title, row_data_list, parsers):
    """
    Write one markdown table block to file f.
    row_data_list is a list of tuples: (grammar_name, dataframe)
    """
    f.write(f'### {group_title}\n\n')

    # Header
    col_headers = ' | '.join(parsers)
    separator   = ' | '.join(['---'] * len(parsers))
    f.write(f'| Grammar | {col_headers} |\n')
    f.write(f'|---------|{separator}|\n')

    # Rows (Grammars)
    for name, df in row_data_list:
        cells = ' | '.join(_format_cell(df, parser) for parser in parsers)
        f.write(f'| {name} | {cells} |\n')
    f.write('\n')


def generate_tables():
    """Write a markdown file with average relative MAD (%) summary tables for every grammar group."""
    out_path = os.path.join(TABLE_DIR, 'summary.md')

    with open(out_path, 'w') as f:
        f.write('# Parser Benchmark Summary: Average Relative MAD\n\n')
        f.write('Values shown are the average of (MAD / Median Time) * 100% across all successful input sizes.\n\n')

        # --- General parsers ---
        gen_rows = []
        for code, name in GRAMMARS:
            df = load_grammar_data(code)
            if not df.empty:
                gen_rows.append((name, df))
        if gen_rows:
            _write_group_table(f, 'General Parsers', gen_rows, PARSERS)

        # --- LL grammars ---
        ll_parsers = ['GLL', 'RNGLR', 'BRNGLR', 'LL', 'LR']
        ll_rows = []
        for code, name in LL_GRAMMARS:
            frames = []
            for fpath in LL_GRAMMAR_FILES.get(code, []):
                if os.path.exists(fpath):
                    try:
                        frames.append(pd.read_csv(fpath))
                    except Exception:
                        pass
            if frames:
                df = pd.concat(frames, ignore_index=True)
                df = df[df['success'] == True]
                if not df.empty:
                    ll_rows.append((name, df))
        if ll_rows:
            _write_group_table(f, 'LL(1) Grammars', ll_rows, ll_parsers)

        # --- LR grammars ---
        lr_parsers = ['Earley', 'GLL', 'RNGLR', 'BRNGLR', 'LR']
        lr_rows = []
        for code, name in LR_GRAMMARS:
            frames = []
            for fpath in LR_GRAMMAR_FILES.get(code, []):
                if os.path.exists(fpath):
                    try:
                        frames.append(pd.read_csv(fpath))
                    except Exception:
                        pass
            if frames:
                df = pd.concat(frames, ignore_index=True)
                df = df[df['success'] == True]
                if not df.empty:
                    lr_rows.append((name, df))
        if lr_rows:
            _write_group_table(f, 'LR Grammars', lr_rows, lr_parsers)

        # --- Ambiguous grammars ---
        ambi_rows = []
        for code, name, fpath in AMBI_GRAMMARS:
            if not os.path.exists(fpath):
                continue
            try:
                df = pd.read_csv(fpath)
                df = df[df['success'] == True]
                if not df.empty:
                    ambi_rows.append((name, df))
            except Exception:
                continue
        if ambi_rows:
            _write_group_table(f, 'Ambiguous Grammars', ambi_rows, AMBI_PARSERS)

    print(f'Saved {out_path}')


if __name__ == '__main__':
    plot_all()
    plot_ll_comparison()
    plot_lr_comparison()
    plot_ambiguous_comparison()
    plot_cyk_valiant_comparison()
    generate_tables()
