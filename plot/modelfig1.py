import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba

def plot_series_with_band(
    x,
    ys,
    band=None,
    band_scale=1.0,
    line_width=2.2,
    band_alpha=0.14,
    line_alpha=1.0,
    zorder_start=10,
    ax=None,
    show_tick_labels=True,
):
    """
    画多条时间序列，并为每条序列加同色系浅阴影带。

    参数
    ----
    x : (T,) array-like
        时间轴
    ys : list of (T,) array-like
        多条序列，长度决定曲线条数
    band : None | list | array-like | callable
        阴影带“半宽度”(half-width) 的来源：
        - None: 不画阴影带
        - array-like (T,): 所有曲线共用同一个半宽度
        - list: 每条曲线一个半宽度（每个元素可为 (T,) 或标量）
        - callable: 接收 y -> 返回 (T,) 的半宽度（每条线单独算）
    band_scale : float
        半宽度缩放系数
    band_alpha : float
        阴影透明度（越小越浅）
    """

    if ax is None:
        fig, ax = plt.subplots(figsize=(7.2, 2.8), dpi=160)
    else:
        fig = ax.figure

    x = np.asarray(x)
    ys = [np.asarray(y) for y in ys]

    for i, y in enumerate(ys):
        # 先画线，让 matplotlib 自动分配颜色（也可以之后手动传 colors）
        ln, = ax.plot(
            x, y,
            lw=line_width,
            alpha=line_alpha,
            solid_capstyle="round",
            zorder=zorder_start + i
        )
        c = ln.get_color()
        rgba = to_rgba(c, 1.0)

        # 决定该条线的 band 半宽度
        if band is None:
            continue
        elif callable(band):
            hw = np.asarray(band(y))
        elif isinstance(band, (list, tuple)):
            b_i = band[i]
            hw = np.full_like(y, float(b_i)) if np.isscalar(b_i) else np.asarray(b_i)
        else:
            hw = np.full_like(y, float(band)) if np.isscalar(band) else np.asarray(band)

        hw = band_scale * hw
        y_lo, y_hi = y - hw, y + hw

        # 阴影：用同一个颜色，但更透明（看起来就是“浅浅的版本”）
        ax.fill_between(
            x, y_lo, y_hi,
            color=rgba, alpha=band_alpha,
            linewidth=0,
            zorder=zorder_start + i - 1
        )

    ax.set_xlim(x.min(), x.max())
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()

    # 浅浅的渐变背景：自上而下由浅绿到浅黄（更淡，接近白）
    n_grad = 128
    grad = np.ones((n_grad, 1, 4))
    # 图像 row 0=底部(y0)，row -1=顶部(y1)：顶部极浅绿，底部极浅黄
    grad[:, 0, 0] = np.linspace(1.0, 0.96, n_grad)   # R: 黄→绿
    grad[:, 0, 1] = np.linspace(1.0, 0.99, n_grad)   # G
    grad[:, 0, 2] = np.linspace(0.96, 0.96, n_grad)  # B
    grad[:, 0, 3] = 1.0
    grad_2d = np.repeat(grad, 2, axis=1)  # 至少 2 列供 imshow 用
    ax.imshow(
        grad_2d,
        extent=[x0, x1, y0, y1],
        aspect="auto",
        zorder=0,
        interpolation="bilinear",
    )

    ax.grid(False)  # 去掉背景网格
    ax.tick_params(labelsize=9)
    if not show_tick_labels:
        ax.tick_params(axis="both", which="both", labelbottom=False, labelleft=False, bottom=False, left=False)
        for spine in ax.spines.values():
            spine.set_visible(False)
    else:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    return fig, ax


# =========================
# 示例：曲线条数可调，支持命令行参数
# =========================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="画多条时间序列+阴影带并保存")
    parser.add_argument("--axis", action="store_true", help="显示 X/Y 轴刻度（默认不显示）")
    parser.add_argument("--band-scale", type=float, default=1.35, help="阴影带宽度系数，越大越宽（默认 1.35）")
    parser.add_argument("--band-alpha", type=float, default=0.16, help="阴影透明度（默认 0.16）")
    parser.add_argument("--dpi", type=int, default=450, help="保存图片 DPI（默认 450）")
    parser.add_argument("-o", "--out", default="modelfig1.jpg", help="输出文件路径（默认 modelfig1.jpg）")
    parser.add_argument("--show", action="store_true", help="同时弹窗显示")
    args = parser.parse_args()

    rng = np.random.default_rng(0)
    T = 220
    x = np.arange(T)

    n_series = 3
    ys = []
    for k in range(n_series):
        y = np.cumsum(rng.normal(0, 0.22 + 0.05*k, size=T)) + 0.03*k*x
        ys.append(y)

    def rolling_std_band(y, win=21, min_hw=0.10):
        y = np.asarray(y)
        hw = np.empty_like(y, dtype=float)
        half = win // 2
        for t in range(len(y)):
            l = max(0, t - half)
            r = min(len(y), t + half + 1)
            hw[t] = np.std(y[l:r])
        return np.maximum(hw, min_hw)

    fig, ax = plot_series_with_band(
        x, ys,
        band=lambda y: rolling_std_band(y, win=25, min_hw=0.12),
        band_scale=args.band_scale,
        band_alpha=args.band_alpha,
        line_width=2.6,
        show_tick_labels=args.axis,
    )

    # 无图例、无标题
    plt.tight_layout()
    plt.savefig(args.out, dpi=args.dpi, format="jpg")
    print(f"已保存: {args.out} (dpi={args.dpi}, 带宽系数={args.band_scale}, 显示坐标={args.axis})")
    if args.show:
        plt.show()
