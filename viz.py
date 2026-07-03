import plotly.graph_objects as go

# Sequential blue ramp (magnitude, light->dark) -- validated default palette,
# not a rainbow/jet colormap (see dataviz skill: sequential = one hue).
BLUE_RAMP = [
    [0.0, "#f7fafd"],
    [0.15, "#cde2fb"],
    [0.35, "#9ec5f4"],
    [0.55, "#6da7ec"],
    [0.72, "#3987e5"],
    [0.86, "#256abf"],
    [1.0, "#0d366b"],
]

MAX_ANNOTATED_CELLS = 144  # annotate numeric weights only on small grids (<=12x12)


def build_attention_heatmap(src_tokens, tgt_tokens, attn_matrix, dark_mode=False):
    """attn_matrix: [tgt_len, src_len] tensor/array of head-averaged
    cross-attention weights. Rows are the generated (target) tokens, columns
    are the source tokens."""
    z = attn_matrix.tolist() if hasattr(attn_matrix, "tolist") else attn_matrix

    n_rows, n_cols = len(z), len(z[0]) if z else 0
    annotate = n_rows * n_cols <= MAX_ANNOTATED_CELLS

    text = [[f"{v:.2f}" for v in row] for row in z] if annotate else None

    heatmap = go.Heatmap(
        z=z,
        x=src_tokens,
        y=tgt_tokens,
        colorscale=BLUE_RAMP,
        zmin=0,
        zmax=1,
        colorbar=dict(title="weight"),
        text=text,
        texttemplate="%{text}" if annotate else None,
        textfont={"size": 11},
        hovertemplate="source: %{x}<br>generated: %{y}<br>weight: %{z:.3f}<extra></extra>",
    )

    surface = "#1c1c1c" if dark_mode else "#fcfcfb"
    ink = "#e6e6e3" if dark_mode else "#1c1c1c"

    fig = go.Figure(data=[heatmap])
    fig.update_layout(
        paper_bgcolor=surface,
        plot_bgcolor=surface,
        font=dict(color=ink),
        xaxis=dict(title="Source (English)", side="bottom", tickangle=-45),
        yaxis=dict(title="Generated (French)", autorange="reversed"),
        margin=dict(l=10, r=10, t=30, b=10),
        height=max(300, 40 * n_rows),
    )
    return fig
