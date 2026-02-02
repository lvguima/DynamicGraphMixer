import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

def make_demo_series(T=140, seed=0):
    rng = np.random.default_rng(seed)
    x = np.arange(T)
    y_tr = 0.018 * x + 0.35 * np.sin(x / 6) + 0.10 * rng.normal(size=T)
    y_se = 0.014 * x + 0.45 * np.sin(x / 4 + 0.8) + 0.12 * rng.normal(size=T) + 0.35
    return x, y_tr, y_se

def rolling_band(y, win=21, min_hw=0.06):
    y = np.asarray(y)
    hw = np.empty_like(y, dtype=float)
    half = win // 2
    for t in range(len(y)):
        l = max(0, t - half)
        r = min(len(y), t + half + 1)
        hw[t] = np.std(y[l:r])
    return np.maximum(hw, min_hw)

def plot_decomp_panel(
    x, y_trend, y_season,
    band_trend=None, band_season=None,
    trend_color="#3A7E5A",
    season_color="#B86A2B",
    save_path="decomp_panel.pdf"
):
    fig = plt.figure(figsize=(8.0, 3.2), dpi=200)
    ax_main = fig.add_subplot(111)
    fig.patch.set_alpha(0.0)
    ax_main.set_facecolor("none")

    x = np.asarray(x)
    y_trend = np.asarray(y_trend)
    y_season = np.asarray(y_season)

    if band_trend is None:
        band_trend = rolling_band(y_trend, win=25, min_hw=0.05)
    if band_season is None:
        band_season = rolling_band(y_season, win=25, min_hw=0.05)

    ax_main.fill_between(x, y_trend - band_trend, y_trend + band_trend,
                         color=trend_color, alpha=0.18, linewidth=0, zorder=1)
    ax_main.fill_between(x, y_season - band_season, y_season + band_season,
                         color=season_color, alpha=0.18, linewidth=0, zorder=1)

    ax_main.plot(x, y_trend, color=trend_color, lw=2.6, solid_capstyle="round", zorder=4)
    ax_main.plot(x, y_season, color=season_color, lw=2.6, solid_capstyle="round", zorder=4)

    ax_main.grid(True, axis="y", linestyle=(0, (3, 4)), linewidth=1.0, alpha=0.25)
    ax_main.grid(True, axis="x", linestyle=(0, (3, 4)), linewidth=1.0, alpha=0.18)

    ax_main.set_axis_off()

    y_min = min((y_trend - band_trend).min(), (y_season - band_season).min())
    y_max = max((y_trend + band_trend).max(), (y_season + band_season).max())
    ax_main.set_xlim(x.min(), x.max() + (x.max()-x.min())*0.03)
    ax_main.set_ylim(y_min - 0.08*(y_max-y_min), y_max + 0.08*(y_max-y_min))

    fig.savefig(save_path, bbox_inches="tight", dpi=500, transparent=True)
    print(f"Saved to: {save_path}")

if __name__ == "__main__":
    x, y_tr, y_se = make_demo_series(T=150, seed=1)
    plot_decomp_panel(
        x, y_tr, y_se,
        save_path="decomp_panel.png"
    )
