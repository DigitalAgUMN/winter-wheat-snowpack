import os

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error

from winter_wheat.util import get_project_root


def draw_lineplot(output_filename, df_filename):
    sns.set(style="white", palette="muted", color_codes=True)
    params = {
        'legend.fontsize': 12,
        'axes.labelsize': 14,
        'axes.titlesize': 14,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
    }
    plt.rcParams.update(params)

    pane_row = 7
    pane_col = 3
    # Set up the matplotlib figure
    fig, axes = plt.subplots(pane_row, pane_col, figsize=(12, 12), sharex=True, sharey="row", dpi=300, constrained_layout=False)
    plt.subplots_adjust(left=0.05, bottom=0.04, right=0.98, top=0.98, wspace=0.05, hspace=0.05)

    df = pd.read_csv(df_filename)

    df_cdl = pd.read_csv(get_project_root() / "input/GPS/gps_cdl.csv")
    df = df.merge(df_cdl, how="left", left_on="station", right_on="station")
    df = df[df["weight"] >= 0.125].copy()
    df["date"] = pd.to_datetime(df["date_str"], format="%Y%m%d")

    stations = df["station"].unique()

    for idx, station in enumerate(stations):
        r = idx // pane_col
        c = idx % pane_col
        ax = axes[r, c]

        df_month = df[df["station"] == station]

        date_rng = pd.date_range(start='1/1/2012', end='5/31/2017', freq='D')
        df_line = pd.DataFrame(date_rng, columns=['date'])
        df_month = df_line.merge(df_month, on="date", how="left")

        df_month_clean = df_month.dropna()

        r2 = pearsonr(df_month_clean['snow_depth_GPS_cm'], df_month_clean['snow_CMC'])[0] ** 2

        rmse = np.sqrt(mean_squared_error(df_month_clean['snow_CMC'], df_month_clean['snow_depth_GPS_cm']))
        true_values = df_month_clean['snow_depth_GPS_cm']
        target_values = df_month_clean['snow_CMC']

        label_lat = df_month_clean["lat"].iloc[0]
        label_lon = df_month_clean["lon"].iloc[0]
        label_elevation = df_month_clean["elevation"].iloc[0]


        df_line = pd.DataFrame(date_rng, columns=['date'])
        df_month_clean = df_line.merge(df_month_clean, on="date", how="left")

        ax.plot(df_month_clean["date"], df_month_clean['snow_CMC'], color='#6699cc', label='CMC (cm)', linewidth=1)  # tab:blue
        ax.plot(df_month_clean["date"], df_month_clean['snow_depth_GPS_cm'], color='#F39B7F', label='GPS (cm)', linewidth=1)  # tab:orange

        ax.annotate("site: {}".format(station, r, c),
                    xy=(0.5, 0.87), xycoords='axes fraction',
                    xytext=(0.5, 0.87), textcoords='axes fraction', fontsize=14,
                    ha='center',
                    arrowprops=dict(arrowstyle="-",
                                    connectionstyle="arc3, rad=0"),
                    )

        ax.annotate("$R^2$ = {:.3f}".format(r2),
                    xy=(0.05, 0.75), xycoords='axes fraction',
                    xytext=(0.05, 0.75), textcoords='axes fraction', fontsize=12,
                    arrowprops=dict(arrowstyle="-",
                                    connectionstyle="arc3, rad=0"),
                    )
        ax.annotate("RMSE = {:.3f}".format(rmse),
                    xy=(0.05, 0.65),  xycoords='axes fraction',
                    xytext=(0.05, 0.65), textcoords='axes fraction', fontsize=12,
                    arrowprops=dict(arrowstyle="-",
                                    connectionstyle="arc3, rad=0"),
                    )

        ax.annotate("Latitude = {:.3f}".format(label_lat),
                    xy=(0.5, 0.75), xycoords='axes fraction',
                    xytext=(0.5, 0.75), textcoords='axes fraction', fontsize=12,
                    arrowprops=dict(arrowstyle="-",
                                    connectionstyle="arc3, rad=0"),
                    )
        ax.annotate("Longitude = {:.3f}".format(label_lon),
                    xy=(0.5, 0.65),  xycoords='axes fraction',
                    xytext=(0.5, 0.65), textcoords='axes fraction', fontsize=12,
                    arrowprops=dict(arrowstyle="-",
                                    connectionstyle="arc3, rad=0"),
                    )
        ax.annotate("Elevation = {:.1f}".format(label_elevation),
                    xy=(0.5, 0.55),  xycoords='axes fraction',
                    xytext=(0.5, 0.55), textcoords='axes fraction', fontsize=12,
                    arrowprops=dict(arrowstyle="-",
                                    connectionstyle="arc3, rad=0"),
                    )

        ax.yaxis.set_ticks_position('both')
        if c == 0:
            ax.yaxis.set_tick_params(labelleft=True, direction='in')
        else:
            ax.yaxis.set_tick_params(labelleft=False, direction='in')
        if r == 5 and c != 0:
            ax.xaxis.set_tick_params(labelleft=True)
        print(r2, rmse)

    fig.delaxes(axes[6, 1])
    fig.delaxes(axes[6, 2])

    lines_labels = [ax.get_legend_handles_labels() for ax in fig.axes]
    lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
    fig.align_ylabels(axes[:])
    fig.legend(lines[:2], ["Snow depth (CMC, cm)", "Snow depth (GPS, cm)"], loc='lower right', ncol=1, bbox_to_anchor=(0.98, 0.05))

    fig.text(0.01, 0.5, "Snow depth (cm)", va='center', rotation='vertical')

    plt.savefig(output_filename)
    plt.close()


def main():
    output_dir = "N:/WinterWheat/output/manuscript_20220215_9GCMs"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    df_sd_prod_filename = "N:/WinterWheat/GPS/clean_CMC_all_append_GPS.csv"

    output_filename = os.path.join(output_dir, "FigureS08_lineplot-SD-GPS-CMC.pdf")
    draw_lineplot(output_filename, df_sd_prod_filename)


if __name__ == "__main__":
    main()
