"""Deterministic reproduction of the three figures distributed with NNWS."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import numpy as np

from .metrics import (
    normalize_max_abs,
    reported_snr,
    rmsr_selective_stacking,
)
from .stacking import weighted_stack


def _plot_dependencies():
    try:
        import matplotlib.pyplot as plt
        from matplotlib import rcParams
        from matplotlib.ticker import NullFormatter
        from obspy.signal.filter import bandpass
        from scipy.fft import fft, fftfreq, ifft
        from scipy.interpolate import interp1d
        from scipy.signal import hilbert
    except ImportError as exc:
        raise RuntimeError(
            'Figure dependencies are missing. Install with: pip install -e ".[figures]"'
        ) from exc
    return plt, rcParams, NullFormatter, bandpass, fft, fftfreq, ifft, interp1d, hilbert


def _style(base_size: int = 12, title_size: int = 14) -> None:
    plt, rc_params, *_ = _plot_dependencies()
    plt.style.use("default")
    rc_params.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman"],
            "font.size": base_size,
            "axes.labelsize": base_size,
            "axes.titlesize": title_size,
            "xtick.labelsize": base_size,
            "ytick.labelsize": base_size,
            "legend.fontsize": base_size,
            "figure.dpi": 300,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.1,
        }
    )


def _save_figure(figure, output_dir: Path, basename: str) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    svg = output_dir / f"{basename}.svg"
    pdf = output_dir / f"{basename}.pdf"
    figure.savefig(svg, dpi=300, bbox_inches="tight")
    figure.savefig(pdf, dpi=300, bbox_inches="tight")
    return svg, pdf


def _ricker(time: np.ndarray, shift: float = 20.0) -> np.ndarray:
    shifted = time - shift
    phase = np.pi * 0.1 * shifted
    return (1 - 2 * phase**2) * np.exp(-(phase**2))


def _ricker_sigma2(time: np.ndarray, shift: float = 20.0) -> np.ndarray:
    shifted = time - shift
    sigma = 2.0
    result = (
        2
        * (1 - (shifted / sigma) ** 2)
        * np.exp(-(shifted**2) / (2 * sigma**2))
        / (np.sqrt(3 * sigma) * np.pi**0.25)
    )
    return result / np.max(np.abs(result))


def ground_truth(
    time: np.ndarray,
    station_distance: float = 200.0,
    speed: float = 3.0,
    shift: float = 20.0,
    wavelet: str = "pi",
) -> np.ndarray:
    """Synthetic two-sided empirical Green's function used by Figs. 2 and 3."""
    arrival = station_distance / speed
    symmetric_time = np.linspace(-time[-1], time[-1], 2 * len(time) - 1)
    if wavelet == "pi":
        source = _ricker(time, shift)
    elif wavelet == "sigma2":
        source = _ricker_sigma2(time, shift)
    else:
        raise ValueError("wavelet must be 'pi' or 'sigma2'.")
    autocorrelation = np.correlate(source, source, mode="full")
    autocorrelation /= np.max(np.abs(autocorrelation))
    theoretical = np.zeros_like(symmetric_time)
    for sign in (-1, 1):
        centered = symmetric_time - sign * arrival
        mask = np.abs(centered) < 50
        theoretical[mask] += 2 * (1 - (centered[mask] / 2) ** 2) * np.exp(
            -(centered[mask] ** 2) / 8
        )
    theoretical /= np.max(np.abs(theoretical))
    result = np.convolve(autocorrelation, theoretical, mode="same")
    return result / np.max(np.abs(result))


def ftan_group_velocity(
    time: np.ndarray,
    data: np.ndarray,
    distance: float,
    fmin: float = 1 / 16,
    fmax: float = 1 / 5,
    nfreq: int = 23,
    alpha: float = 2.5,
    vmin: float = 1.0,
    vmax: float = 5.0,
    vdelta: float = 0.002,
) -> tuple[np.ndarray, np.ndarray]:
    """Measure group velocity with the MATLAB-style interpolation workflow."""
    _, _, _, _, fft, fftfreq, ifft, interp1d, hilbert = _plot_dependencies()
    midpoint = len(time) // 2
    right = data[midpoint:]
    left = data[: midpoint + 1][::-1]
    size = min(len(right), len(left))
    symmetric = (right[:size] + left[:size]) / 2
    positive_time = time[midpoint : midpoint + size]
    periods = np.linspace(1 / fmax, 1 / fmin, nfreq)
    frequencies = 1 / periods
    frequency_axis = fftfreq(len(symmetric), np.mean(np.diff(positive_time)))
    velocities = np.full(nfreq, np.nan)
    velocity_grid = np.arange(vmin, vmax + vdelta, vdelta)
    time_mask = (positive_time >= distance / vmax) & (
        positive_time <= distance / vmin
    )

    spectrum = fft(symmetric)
    for index, center_frequency in enumerate(frequencies):
        sigma = center_frequency / alpha
        gaussian = np.exp(
            -((frequency_axis - center_frequency) ** 2) / (2 * sigma**2)
        )
        gaussian += np.exp(
            -((frequency_axis + center_frequency) ** 2) / (2 * sigma**2)
        )
        envelope = np.abs(hilbert(ifft(spectrum * gaussian).real))
        if not np.any(time_mask):
            continue
        window_envelope = envelope[time_mask]
        window_time = positive_time[time_mask]
        window_velocity = distance / window_time
        order = np.argsort(window_velocity)[::-1]
        velocity_sorted = window_velocity[order]
        envelope_sorted = window_envelope[order]
        unique = np.diff(velocity_sorted) < 0
        velocity_unique = velocity_sorted[:-1][unique]
        envelope_unique = envelope_sorted[:-1][unique]
        if len(velocity_unique) < 4:
            continue
        try:
            interpolation = interp1d(
                velocity_unique,
                envelope_unique,
                kind="cubic",
                bounds_error=False,
                fill_value=0,
            )
            envelope_grid = interpolation(velocity_grid)
            peak = int(np.argmax(envelope_grid))
            group_velocity = velocity_grid[peak]
            if 2 <= peak <= len(velocity_grid) - 3:
                local = slice(peak - 2, peak + 3)
                coefficients = np.polyfit(
                    velocity_grid[local], envelope_grid[local], 2
                )
                precise = -coefficients[1] / (2 * coefficients[0])
                if velocity_grid[local][0] <= precise <= velocity_grid[local][-1]:
                    group_velocity = precise
            velocities[index] = group_velocity
        except (TypeError, ValueError):
            peak = int(np.argmax(window_envelope))
            if window_time[peak] > 0:
                velocities[index] = distance / window_time[peak]
    return periods, velocities


def _plot_dispersion_error(
    axis,
    methods: Iterable[np.ndarray],
    *,
    xlabel: bool = True,
    label_size: int = 14,
) -> None:
    time = np.linspace(-500, 500, 1001)
    styles = (("g", "v", 4), ("b", "^", 4), ("r", "o", 6))
    names = ("Linear", "RMSR_SS", "NNWS")
    for data, name, (color, marker, size) in zip(methods, names, styles):
        periods, velocities = ftan_group_velocity(time, data, distance=200)
        error = (velocities - 3.0) / 3.0 * 100
        axis.plot(
            periods,
            error,
            marker=marker,
            color=color,
            linewidth=1.5,
            markersize=size,
            label=name,
            markerfacecolor="none",
        )
    axis.axhline(
        0, color="k", linestyle="--", linewidth=1.5, label="Theoretical value"
    )
    axis.set_xlabel("Period (s)" if xlabel else "", fontsize=label_size)
    axis.set_ylabel("Relative Error (%)", fontsize=label_size)
    axis.grid(True, linestyle="--", alpha=0.5)
    axis.set_ylim(-3.5, 5.5)
    axis.set_xlim(4, 17)
    axis.legend(fontsize=12 if label_size == 14 else 11, loc="upper left")


def _synthetic_paths(root: Path, name: str) -> tuple[Path, Path, Path]:
    return (
        root / "syn_datasets" / f"info_{name}.npy",
        root / "syn_datasets" / f"dataset_{name}.npy",
        root / "syn_results" / f"{name}_nn_weight.npy",
    )


def _load_synthetic(root: Path, name: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    paths = _synthetic_paths(root, name)
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing figure input(s): " + ", ".join(missing))
    return tuple(np.load(path) for path in paths)  # type: ignore[return-value]


def _filtered_ground_truth(wavelet: str = "pi") -> np.ndarray:
    *_, bandpass, _, _, _, _, _ = _plot_dependencies()
    waveform = ground_truth(np.linspace(0, 500, 501), wavelet=wavelet)
    filtered = bandpass(
        waveform, freqmin=1 / 16, freqmax=1 / 5, df=1, corners=4, zerophase=True
    )
    return filtered / np.max(filtered)


def make_test0_figure(root: str | Path = "."):
    """Reproduce ``Figs/test0_analysis_figure.{svg,pdf}``."""
    plt, _, NullFormatter, *_ = _plot_dependencies()
    _style(12, 14)
    root = Path(root).resolve()
    info, dataset, weights = _load_synthetic(root, "test_uni")
    tau, tmin, tmax, delta = 8, 64, 69, 1
    figure, axes = plt.subplots(1, 3, figsize=(15, 4.35))

    radius, theta = info[:, 0], info[:, 1]
    x, y = radius * np.cos(theta), radius * np.sin(theta)
    scatter = axes[0].scatter(
        x, y, c=info[:, 2], s=3, cmap="plasma", edgecolors="none", alpha=0.8
    )
    for station_x, label in ((-100, "A"), (100, "B")):
        axes[0].scatter(
            station_x,
            0,
            marker="v",
            color="#d62728",
            s=100,
            edgecolor="k",
            linewidth=0.5,
        )
        axes[0].text(
            station_x, 50, label, ha="center", va="center", fontsize=10,
            fontweight="bold",
        )
    axes[0].set_xlabel("Distance X (km)", fontsize=14)
    axes[0].set_ylabel("Distance Y (km)", fontsize=14)
    axes[0].set_xlim(x.min() - 20, x.max() + 20)
    axes[0].set_ylim(y.min() - 20, y.max() + 20)
    colorbar = figure.colorbar(
        scatter, ax=axes[0], orientation="vertical", shrink=0.8, aspect=15, pad=0.05
    )
    colorbar.set_label("Time (s)", fontsize=14)
    colorbar.ax.tick_params(labelsize=10)

    linear, selective, _ = rmsr_selective_stacking(
        dataset, tau, tmin, tmax, delta, threshold=5
    )
    neural = weighted_stack(dataset, weights)
    truth = _filtered_ground_truth()
    normalized = [
        normalize_max_abs(value) for value in (linear, selective, neural)
    ]
    snrs = [
        reported_snr(value, tau, tmin, tmax, delta)
        for value in normalized
    ]
    correlations = [np.corrcoef(truth, value)[0, 1] for value in normalized]
    time_axis = np.linspace(-200, 200, 401)
    axes[1].plot(
        time_axis, normalized[0][300:701] + 4, lw=1.5, color="g", label="Linear"
    )
    axes[1].plot(
        time_axis, normalized[1][300:701] + 2, lw=1.5, color="b", label="RMSR_SS"
    )
    axes[1].plot(
        time_axis, normalized[2][300:701], lw=1.5, color="r", label="NNWS"
    )
    axes[1].plot(
        time_axis, truth[300:701] - 2, lw=2, color="k", label="Ground Truth"
    )
    for ymin, ymax in ((-0.4, 0.4), (1.6, 2.4), (3.6, 4.4)):
        for lag in (
            tmin - 2 * tau,
            tmax + 2 * tau,
            -tmin + 2 * tau,
            -tmax - 2 * tau,
        ):
            axes[1].vlines(
                lag, ymin, ymax, color="gray", linestyles="--", linewidth=0.8
            )
        for offset in (6, 10):
            for lag in (-tmax - offset * tau, tmax + offset * tau):
                axes[1].vlines(
                    lag, ymin, ymax, color="orange", linestyles="--", linewidth=0.8
                )
    for y_text, snr, correlation in zip((4.6, 2.6, 0.6), snrs, correlations):
        axes[1].text(-190, y_text, f"SNR: {snr:.2f}", ha="left", fontsize=14)
        axes[1].text(
            190, y_text, f"CC: {np.round(correlation, 2)}", ha="right", fontsize=14
        )
    axes[1].yaxis.set_major_formatter(NullFormatter())
    axes[1].yaxis.set_minor_formatter(NullFormatter())
    axes[1].set_xlim(-200, 200)
    axes[1].set_ylim(-3, 5.2)
    axes[1].set_xlabel("Lag time (s)", fontsize=14)
    axes[1].legend(fontsize=8, loc="lower left")

    # The published dispersion panel used G=1+1/M for RMSR_SS.
    linear_disp, selective_disp, _ = rmsr_selective_stacking(
        dataset, tau, tmin, tmax, delta, threshold=1
    )
    _plot_dispersion_error(
        axes[2], (linear_disp, selective_disp, neural), label_size=14
    )
    for axis, label in zip(axes, ("(a)", "(b)", "(c)")):
        axis.text(
            -0.08,
            1.08,
            label,
            transform=axis.transAxes,
            fontsize=16,
            fontweight="bold",
            va="bottom",
            ha="right",
        )
    figure.tight_layout()
    paths = _save_figure(figure, root / "Figs", "test0_analysis_figure")
    return figure, paths


def make_comprehensive_figure(root: str | Path = "."):
    """Reproduce ``Figs/comprehensive_analysis_figure.{svg,pdf}``."""
    plt, _, NullFormatter, *_ = _plot_dependencies()
    _style(18, 20)
    root = Path(root).resolve()
    names = ("test_azi40", "test_azi65", "test_azi90")
    loaded = [_load_synthetic(root, name) for name in names]
    tau, tmin, tmax, delta = 8, 64, 69, 1
    # This panel follows the reference fig3_m.py source-wavelet convention.
    truth = _filtered_ground_truth(wavelet="sigma2")
    figure, axes = plt.subplots(3, 3, figsize=(14.8, 12))
    panel_labels = (("(a)", "(b)", "(c)"), ("(d)", "(e)", "(f)"), ("(g)", "(h)", "(i)"))
    titles = ("Source Distribution", "Waveform Comparison", "Dispersion Error")

    for row, (name, (info, dataset, weights)) in enumerate(zip(names, loaded)):
        radius, theta = info[:, 0], info[:, 1]
        x, y = radius * np.cos(theta), radius * np.sin(theta)
        source_axis = axes[row, 0]
        scatter = source_axis.scatter(
            x,
            y,
            c=info[:, 2],
            s=3,
            cmap="plasma",
            edgecolors="none",
            alpha=0.8,
        )
        for station_x, label in ((-100, "A"), (100, "B")):
            source_axis.scatter(
                station_x,
                0,
                marker="v",
                color="#d62728",
                s=100,
                edgecolor="k",
                linewidth=0.5,
            )
            source_axis.text(
                station_x, 50, label, ha="center", va="center", fontsize=10,
                fontweight="bold",
            )
        source_axis.set_xlabel(
            "Distance X (km)" if row == 2 else "", fontsize=18
        )
        source_axis.set_ylabel("Distance Y (km)", fontsize=18)
        source_axis.set_xlim(x.min() - 20, x.max() + 20)
        source_axis.set_ylim(y.min() - 20, y.max() + 20)
        colorbar = figure.colorbar(
            scatter,
            ax=source_axis,
            orientation="vertical",
            shrink=0.8,
            aspect=15,
            pad=0.05,
        )
        colorbar.set_label("Time (s)", fontsize=18)
        colorbar.ax.tick_params(labelsize=18)
        source_axis.text(
            -0.45,
            0.5,
            name,
            transform=source_axis.transAxes,
            fontsize=20,
            va="center",
            ha="center",
            rotation=90,
        )

        linear, selective, _ = rmsr_selective_stacking(
            dataset, tau, tmin, tmax, delta, threshold=5
        )
        neural = weighted_stack(dataset, weights)
        normalized = [
            normalize_max_abs(value) for value in (linear, selective, neural)
        ]
        snrs = [
            reported_snr(value, tau, tmin, tmax, delta)
            for value in normalized
        ]
        correlations = [np.corrcoef(truth, value)[0, 1] for value in normalized]
        waveform_axis = axes[row, 1]
        time_axis = np.linspace(-200, 200, 401)
        waveform_axis.plot(
            time_axis, normalized[2][300:701], "r", lw=1.5, label="NNWS"
        )
        waveform_axis.plot(
            time_axis,
            normalized[1][300:701] + 2,
            "b",
            lw=1.5,
            label="RMSR_SS",
        )
        waveform_axis.plot(
            time_axis,
            normalized[0][300:701] + 4,
            "g",
            lw=1.5,
            label="Linear",
        )
        waveform_axis.plot(
            time_axis, truth[300:701] - 2, "k", lw=2, label="Ground Truth"
        )
        for ymin, ymax in ((-0.4, 0.4), (1.6, 2.4), (3.6, 4.4)):
            for lag in (
                tmin - 2 * tau,
                tmax + 2 * tau,
                -tmin + 2 * tau,
                -tmax - 2 * tau,
            ):
                waveform_axis.vlines(
                    lag, ymin, ymax, color="r", linestyles="--", linewidth=0.8
                )
            for offset in (6, 10):
                for lag in (-tmax - offset * tau, tmax + offset * tau):
                    waveform_axis.vlines(
                        lag,
                        ymin,
                        ymax,
                        color="orange",
                        linestyles="--",
                        linewidth=0.8,
                    )
        for y_text, snr, correlation in zip((4.6, 2.6, 0.6), snrs, correlations):
            waveform_axis.text(
                -190,
                y_text,
                f"SNR: {np.round(snr, 2)}",
                ha="left",
                fontsize=15,
            )
            waveform_axis.text(
                190,
                y_text,
                f"CC: {np.round(correlation, 2)}",
                ha="right",
                fontsize=15,
            )
        waveform_axis.yaxis.set_major_formatter(NullFormatter())
        waveform_axis.yaxis.set_minor_formatter(NullFormatter())
        waveform_axis.set_xlim(-200, 200)
        waveform_axis.set_ylim(-3, 8)
        waveform_axis.set_xlabel("Lag time (s)" if row == 2 else "", fontsize=18)
        waveform_axis.legend(
            fontsize=11,
            loc="upper center",
            ncol=2,
            columnspacing=0.9,
            handlelength=2,
            bbox_to_anchor=(0.5, 0.99),
        )

        dispersion_axis = axes[row, 2]
        _plot_dispersion_error(
            dispersion_axis,
            (linear, selective, neural),
            xlabel=row == 2,
            label_size=18,
        )
        for column in range(3):
            axes[row, column].text(
                -0.08,
                1.08,
                panel_labels[row][column],
                transform=axes[row, column].transAxes,
                fontsize=18,
                fontweight="bold",
                va="bottom",
                ha="right",
            )
            if row == 0:
                axes[row, column].set_title(
                    titles[column], fontsize=20, pad=15 if column == 1 else 10
                )

    figure.tight_layout()
    figure.subplots_adjust(wspace=0.26, hspace=0.22)
    paths = _save_figure(
        figure, root / "Figs", "comprehensive_analysis_figure"
    )
    return figure, paths


def make_mechanism_figure(root: str | Path = "."):
    """Reproduce ``Figs/mechanism.{svg,pdf}``."""
    plt, *_ = _plot_dependencies()
    try:
        import pandas as pd
        import seaborn as sns
        from matplotlib.cm import ScalarMappable
        from matplotlib.colors import PowerNorm
        from scipy.stats import gaussian_kde
    except ImportError as exc:
        raise RuntimeError(
            'Mechanism-figure dependencies are missing; install ".[figures]".'
        ) from exc
    _style(12, 14)
    root = Path(root).resolve()
    info, dataset, weights = _load_synthetic(root, "test_azi40")
    normalized_weights = weights / np.max(weights)
    figure = plt.figure(figsize=(15, 12))
    top_grid = figure.add_gridspec(
        1, 2, width_ratios=(7, 1), wspace=0.08, top=0.95, bottom=0.58
    )
    heatmap_axis = figure.add_subplot(top_grid[0])
    weight_axis = figure.add_subplot(top_grid[1])

    frame = pd.DataFrame(dataset[:, 300:701] / np.max(dataset[:, 300:701]))
    frame["Label"] = normalized_weights
    sorted_frame = frame.sort_values("Label", ascending=False).reset_index(drop=True)
    heatmap_data = sorted_frame.iloc[:, :-1] + 1
    color_norm = PowerNorm(
        gamma=0.8,
        vmin=heatmap_data.min().min(),
        vmax=heatmap_data.max().max(),
    )
    heatmap_axis.imshow(
        heatmap_data,
        aspect="auto",
        cmap="seismic",
        norm=color_norm,
        extent=(-200, 200, len(sorted_frame), 0),
    )
    heatmap_axis.set_xlabel("Lag Time (s)", fontsize=16)
    heatmap_axis.set_ylabel("CCF Index", fontsize=16)
    heatmap_axis.set_xticks(np.arange(-200, 201, 100))
    heatmap_axis.set_yticks(np.arange(0, 601, 100))
    heatmap_axis.tick_params(axis="both", which="major", length=6, width=1.2, labelsize=14)
    for spine in heatmap_axis.spines.values():
        spine.set_linewidth(1.5)

    indices = np.arange(len(sorted_frame))
    sorted_weights = sorted_frame["Label"].values
    weight_axis.plot(sorted_weights, indices, color="#2E86AB", linewidth=2, alpha=0.9)
    weight_axis.fill_betweenx(indices, 0, sorted_weights, alpha=0.3, color="#2E86AB")
    weight_axis.set_xlabel("Weight", fontsize=16)
    weight_axis.set_yticks(np.arange(0, 601, 100))
    weight_axis.yaxis.tick_right()
    weight_axis.set_ylim(heatmap_axis.get_ylim())
    weight_axis.grid(True, linestyle=":", alpha=0.6, color="gray", linewidth=0.8)
    weight_axis.set_axisbelow(True)
    weight_axis.set_xlim(0, 1)
    weight_axis.set_xticks((0, 0.5, 1))
    for spine in weight_axis.spines.values():
        spine.set_linewidth(1.5)

    bottom_grid = figure.add_gridspec(
        1, 2, wspace=0.15, top=0.52, bottom=0.08
    )
    unweighted_axis = figure.add_subplot(bottom_grid[0])
    weighted_axis = figure.add_subplot(bottom_grid[1])
    cluster: list[int] = []
    source_weights: list[np.ndarray] = []
    for segment in range(dataset.shape[0]):
        mask = (info[:, 2] > 1001 * segment) & (
            info[:, 2] < 1001 * (segment + 1)
        )
        selected_indices = np.flatnonzero(mask)
        cluster.extend(selected_indices.tolist())
        source_weights.extend([normalized_weights[segment]] * len(selected_indices))
    coordinate = info[cluster, :2]
    radius, theta = coordinate[:, 0], coordinate[:, 1]
    x, y = radius * np.cos(theta), radius * np.sin(theta)
    source_weight_array = np.asarray(source_weights)[:, 0]

    def density_range(kde_weights=None):
        kde = gaussian_kde(np.vstack((x, y)), weights=kde_weights)
        xx, yy = np.mgrid[x.min() : x.max() : 100j, y.min() : y.max() : 100j]
        density = kde(np.vstack((xx.ravel(), yy.ravel()))).reshape(xx.shape)
        return density.min(), density.max()

    unweighted_range = density_range()
    weighted_range = density_range(source_weight_array)
    density_norm = plt.Normalize(
        vmin=min(unweighted_range[0], weighted_range[0]),
        vmax=max(unweighted_range[1], weighted_range[1]),
    )
    sns.kdeplot(
        x=x,
        y=y,
        cmap="plasma",
        fill=True,
        thresh=0.01,
        ax=unweighted_axis,
        norm=density_norm,
    )
    sns.kdeplot(
        x=x,
        y=y,
        weights=source_weight_array,
        cmap="plasma",
        fill=True,
        thresh=0.01,
        ax=weighted_axis,
        norm=density_norm,
    )
    for axis, title in ((unweighted_axis, "Unweighted"), (weighted_axis, "Weighted")):
        axis.set_xlabel("Distance X (km)", fontsize=16)
        axis.set_ylabel("Distance Y (km)", fontsize=16)
        axis.set_title(title, fontsize=20)
        for label, (point_x, point_y) in {
            "P": (400, 300),
            "Q": (-400, 0),
            "R": (400, 0),
        }.items():
            axis.plot(point_x, point_y, "o", markersize=6, color="white")
            axis.text(
                point_x,
                point_y - 40,
                label,
                fontsize=20,
                fontweight="bold",
                color="white",
                ha="center",
                va="top",
            )
    for axis, label, x_offset in (
        (heatmap_axis, "(a)", -0.04),
        (weight_axis, "(b)", -0.08),
        (unweighted_axis, "(c)", -0.08),
        (weighted_axis, "(d)", -0.02),
    ):
        axis.text(
            x_offset,
            1.04 if axis in (heatmap_axis, weight_axis) else 1.02,
            label,
            transform=axis.transAxes,
            fontsize=20,
            fontweight="bold",
            va="bottom",
            ha="right",
        )
    scalar = ScalarMappable(cmap="plasma", norm=density_norm)
    scalar.set_array([])
    position = weighted_axis.get_position()
    colorbar_axis = figure.add_axes(
        [position.x1 + 0.02, position.y0, 0.015, position.height]
    )
    colorbar = figure.colorbar(scalar, cax=colorbar_axis)
    colorbar.set_label("Probability Density", fontsize=16)
    colorbar.ax.tick_params(labelsize=12)
    figure.tight_layout()
    paths = _save_figure(figure, root / "Figs", "mechanism")
    return figure, paths


def reproduce_figures(
    root: str | Path = ".", figures: Iterable[str] = ("all",)
) -> list[Path]:
    """Generate selected publication figures and return all output paths."""
    selected = set(figures)
    if "all" in selected:
        selected = {"test0", "comprehensive", "mechanism"}
    unknown = selected - {"test0", "comprehensive", "mechanism"}
    if unknown:
        raise ValueError(f"Unknown figure name(s): {', '.join(sorted(unknown))}")
    functions = {
        "test0": make_test0_figure,
        "comprehensive": make_comprehensive_figure,
        "mechanism": make_mechanism_figure,
    }
    plt, *_ = _plot_dependencies()
    outputs: list[Path] = []
    for name in ("test0", "comprehensive", "mechanism"):
        if name in selected:
            figure, paths = functions[name](root)
            outputs.extend(paths)
            plt.close(figure)
    return outputs


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path.cwd(),
        help="Repository root containing syn_datasets and syn_results.",
    )
    parser.add_argument(
        "--figures",
        nargs="+",
        choices=("all", "test0", "comprehensive", "mechanism"),
        default=("all",),
    )
    args = parser.parse_args(argv)
    for path in reproduce_figures(args.root, args.figures):
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
