"""Publication defaults for matplotlib (sizes in points at final print size)."""
import matplotlib as mpl


def set_pub_style(serif=True):
    """Apply publication defaults. Call once before plotting."""
    family = (["STIXGeneral", "Times New Roman", "DejaVu Serif"]
              if serif else ["Helvetica", "Arial", "DejaVu Sans"])
    mpl.rcParams.update({
        # typography
        "font.family": "serif" if serif else "sans-serif",
        ("font.serif" if serif else "font.sans-serif"): family,
        "mathtext.fontset": "stix" if serif else "dejavusans",
        "font.size":        8,
        "axes.titlesize":   9,
        "axes.labelsize":   9,
        "xtick.labelsize":  7.5,
        "ytick.labelsize":  7.5,
        "legend.fontsize":  7.5,
        "figure.titlesize": 10,
        # lines and markers
        "lines.linewidth":  1.2,
        "lines.markersize": 3.5,
        "axes.linewidth":   0.6,
        # spines, ticks, grid
        "axes.spines.top":   False,
        "axes.spines.right": False,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "axes.grid": False,
        "grid.linewidth": 0.4,
        "grid.alpha": 0.3,
        "legend.frameon": False,
        # export: vector first, high-dpi raster fallback, embedded fonts
        "savefig.format": "pdf",
        "savefig.dpi": 600,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })
