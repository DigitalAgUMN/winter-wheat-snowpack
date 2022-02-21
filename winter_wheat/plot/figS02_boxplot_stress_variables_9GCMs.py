import csv
import os

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from winter_wheat.util import get_project_root, make_dirs

scenarios = ["rcp45", "rcp85"]
scenarios_title = {"rcp45": "RCP4.5", "rcp85": "RCP8.5"}
periods = ["2080-2100", ]
field_names = {
    "fdd1": ("fdd1", "FDD\n(degree days)",),
    "fdd1_sc2_sctf": ("fdd1_sc2_sctf", "SCF\n(%)",),
    "gdd3_fall": ("gdd_high_fall", "GDD3_F",),
    "gdd3_winter": ("gdd_high_winter", "GDD3_W",),
    "gdd3_spring": ("gdd_high_spring", "GDD$_{high}^{spring}$\n(degree days)",),
    "gdd3_all": ("gdd3_all", "GDD3",),
    "gdd_all": ("gdd_all", "GDD",),
    "prcp_fall": ("prcp_fall", "Rainfall$_{fall}$\n(mm)",),
    "prcp_winter": ("prcp_winter", "Rainfall$_{winter}$\n(mm)",),
    "prcp_spring": ("prcp_spring", "Rainfall$_{spring}$\n(mm)",),
    "prcp_all": ("prcp_all", "Rainfall\n(mm)",),
    "snowfall_fall": ("snowfall_fall", "Snowfall$_{fall}$\n(mm)",),
    "snowfall_winter": ("snowfall_winter", "Snowfall$_{winter}$\n(mm)",),
    "snowfall_spring": ("snowfall_spring", "Snowfall$_{spring}$\n(mm)",),
    "snowfall_all": ("snowfall_all", "Snowfall\n(mm)",),
}
rcp_names = ["RCP4.5", "RCP8.5"]


def draw_box_plot(base_dir, output_filename, models, variables, filename_pattern, box_whisker=None):
    sns.set(style="white", palette="muted", color_codes=True)
    params = {
        'legend.fontsize': 12,
        'axes.labelsize': 14,
        'axes.titlesize': 14,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'axes.linewidth': 1.2,
        'lines.linewidth': 1.2,
        'patch.linewidth': 1.2,
    }


    plt.rcParams.update(params)

    # Set up the matplotlib figure
    fig, axes = plt.subplots(len(variables), 2, figsize=(10, 8), gridspec_kw={"width_ratios": [1, 10], "height_ratios": [1.2, 1., 1, 1, 1], "wspace": 0}, sharey="row", dpi=300, constrained_layout=False)
    plt.subplots_adjust(left=0.11, bottom=0.11, right=0.94, top=0.95, wspace=0.13, hspace=0.1)


    color_pallete = [
        "#66A4C6",  # blue  hsl (142, 116, 164) in pickpick
        "#FC9630",  # orange  hsl (21, 247, 167) in pickpick
    ]

    sns.set_palette(sns.color_palette(color_pallete))

    for i, var_name in enumerate(variables):
        df_hist = pd.read_csv(os.path.join(base_dir, filename_pattern.format(var_name)))
        df = None
        for j, scenario in enumerate(scenarios):
            for model_name in models:
                if df is None:
                    df_hist_copy = df_hist.copy()
                    df_hist_copy["GCMs"] = "historical"
                    df_hist_copy["period"] = "historical"
                    df_hist_copy["scenario"] = "Historical"
                    df_hist_copy["scenario"] = "Historical"
                    df = df_hist_copy
                for period_name in periods:
                    output_all_dir = os.path.join(base_dir, model_name)
                    output_all_dir_each = os.path.join(output_all_dir, scenario)
                    output_all_dir_each = os.path.join(output_all_dir_each, period_name)
                    df_model = pd.read_csv(os.path.join(output_all_dir_each, filename_pattern.format(var_name)))
                    if model_name == "all_models_avg_9":
                        df_model["GCMs"] = "Avg."
                    else:
                        df_model["GCMs"] = model_name
                    df_model["period"] = period_name
                    df_model["scenario"] = scenarios_title[scenario]

                    df = pd.concat([df, df_model], ignore_index=True)

        if field_names[var_name][0] == "fdd1_sc2_sctf":
            df["fdd1_sc2_sctf"] = df["fdd1_sc2_sctf"] * 100  # SCF unit convert to %

        df = df.rename(columns={field_names[var_name][0]: field_names[var_name][1]})
        df["GCMs_lower"] = df['GCMs'].str.lower()
        df = df.sort_values(by="GCMs_lower")

        if box_whisker is None:
            sns.set_palette(sns.color_palette([
                "#66A4C6",  # blue  hsl (142, 116, 164) in pickpick
                "#FC9630",  # orange  hsl (21, 247, 167) in pickpick
                ]))
            g = sns.boxplot(x='GCMs', y=field_names[var_name][1], data=df[df["scenario"]!="historical"], hue='scenario', ax=axes[i, 1],
                            showfliers=False, showmeans=True,
                            width=0.4,
                            meanprops={"marker": "o",
                                       "markerfacecolor": "white",
                                       "markeredgecolor": "black",
                                       "markersize": "5"})
            sns.set_palette(sns.color_palette([
                "#96C#468",
            ]))
            g = sns.boxplot(x='GCMs', y=field_names[var_name][1], data=df[df["scenario"]=="historical"], hue='scenario', ax=axes[i, 0],
                            showfliers=False, showmeans=True,
                            width=0.4,
                            meanprops={"marker": "o",
                                       "markerfacecolor": "white",
                                       "markeredgecolor": "black",
                                       "markersize": "5"})
        else:
            sns.set_palette(sns.color_palette([
                "#66A4C6",  # blue  hsl (142, 116, 164) in pickpick
                "#FC9630",  # orange  hsl (21, 247, 167) in pickpick
                ]))
            g = sns.boxplot(x='GCMs', y=field_names[var_name][1], data=df[df["scenario"]!="Historical"], hue='scenario', ax=axes[i, 1],
                            showfliers=False, showmeans=True, whis=box_whisker,
                            saturation=1.0,
                            width=0.4,
                            meanprops={"marker": "o",
                                       "markerfacecolor": "white",
                                       "markeredgecolor": "black",
                                       "markersize": "5"})
            sns.set_palette(sns.color_palette([
                "#96C468",
            ]))
            g = sns.boxplot(x='GCMs', y=field_names[var_name][1], data=df[df["scenario"]=="Historical"], hue='scenario', ax=axes[i, 0],
                            showfliers=False, showmeans=True, whis=box_whisker,
                            saturation=1.0,
                            width=0.2,
                            meanprops={"marker": "o",
                                       "markerfacecolor": "white",
                                       "markeredgecolor": "black",
                                       "markersize": "5"})

        for j in range(2):  # historical/future
            if i < len(variables) - 1:
                axes[i, j].set_xlabel('')
                axes[i, j].set_xticklabels([])
            else:
                axes[i, j].xaxis.set_tick_params(direction="out")
                axes[i, j].xaxis.set_ticks_position('bottom')
                for label in axes[i, j].get_xticklabels():
                    label.set_rotation(30)
                    label.set_ha('right')
            if j == 0:
                axes[i, j].set(xlabel="")
                axes[i, j].yaxis.set_tick_params(labelleft=True)
                axes[i, j].yaxis.set_label_position("left")
                axes[i, j].yaxis.set_ticks_position('left')
            else:
                axes[i, j].yaxis.set_tick_params(labelright=True)
                axes[i, j].yaxis.set_ticks_position('right')
                axes[i, j].set_ylabel('')

            axes[i, j].legend_.remove()
            axes[i, j].yaxis.set_tick_params(direction="out")

            axes[i, 0].spines['right'].set_visible(False)
            axes[i, 1].spines['left'].set_visible(False)

        if i == 0:
            for j in range(2):  # historical/future
                y_lim_top = axes[0, j].get_ylim()
                print(y_lim_top)
                axes[0, j].set(ylim=(y_lim_top[0], y_lim_top[1] * 1.1))

        axes[i, 1].axvline(0.5, color='gray', ls="--")
        axes[i, 1].axvline(-0.45, color='gray', ls="--")


    lines_labels = [ax.get_legend_handles_labels() for ax in fig.axes]
    lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
    fig.align_ylabels(axes[:])

    fig.legend(lines[:3], labels[:3], loc='upper right', ncol=3, bbox_to_anchor=(0.9, 0.94))
    plt.savefig(output_filename)

    plt.close()


def draw_box_plot_season(base_dir, output_filename, models, variables, filename_pattern, box_whisker=None):
    sns.set(style="white", palette="muted", color_codes=True)
    params = {
        'legend.fontsize': 12,
        'axes.labelsize': 14,
        'axes.titlesize': 14,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
    }
    plt.rcParams.update(params)

    # Set up the matplotlib figure
    fig, axes = plt.subplots(len(variables), 2, figsize=(10, 10), gridspec_kw={"width_ratios": [1, 10], "height_ratios": [1.2, 1., 1, 1, 1, 1], "wspace": 0}, sharey="row", dpi=300, constrained_layout=False)
    plt.subplots_adjust(left=0.11, bottom=0.11, right=0.94, top=0.95, wspace=0.13, hspace=0.1)


    color_pallete = [
        "#66A4C6",  # blue  hsl (142, 116, 164) in pickpick
        "#FC9630",  # orange  hsl (21, 247, 167) in pickpick
    ]

    sns.set_palette(sns.color_palette(color_pallete))

    for i, var_name in enumerate(variables):
        df_hist = pd.read_csv(os.path.join(base_dir, filename_pattern.format(var_name)))
        df = None
        for j, scenario in enumerate(scenarios):
            for model_name in models:
                if df is None:
                    df_hist_copy = df_hist.copy()
                    df_hist_copy["GCMs"] = "historical"
                    df_hist_copy["period"] = "historical"
                    df_hist_copy["scenario"] = "Historical"
                    df_hist_copy["scenario"] = "Historical"
                    df = df_hist_copy
                for period_name in periods:
                    output_all_dir = os.path.join(base_dir, model_name)
                    output_all_dir_each = os.path.join(output_all_dir, scenario)
                    output_all_dir_each = os.path.join(output_all_dir_each, period_name)
                    df_model = pd.read_csv(os.path.join(output_all_dir_each, filename_pattern.format(var_name)))
                    if model_name == "all_models_avg_9":
                        df_model["GCMs"] = "Avg."
                    else:
                        df_model["GCMs"] = model_name
                    df_model["period"] = period_name
                    df_model["scenario"] = scenarios_title[scenario]

                    df = pd.concat([df, df_model], ignore_index=True)

        df = df.rename(columns={field_names[var_name][0]: field_names[var_name][1]})
        df["GCMs_lower"] = df['GCMs'].str.lower()
        df = df.sort_values(by="GCMs_lower")

        if box_whisker is None:
            sns.set_palette(sns.color_palette([
                "#66A4C6",  # blue  hsl (142, 116, 164) in pickpick
                "#FC9630",  # orange  hsl (21, 247, 167) in pickpick
                ]))
            g = sns.boxplot(x='GCMs', y=field_names[var_name][1], data=df[df["scenario"]!="historical"], hue='scenario', ax=axes[i, 1],
                            showfliers=False, showmeans=True,
                            width=0.4,
                            meanprops={"marker": "o",
                                       "markerfacecolor": "white",
                                       "markeredgecolor": "black",
                                       "markersize": "5"})
            sns.set_palette(sns.color_palette([
                "#96C#468",
            ]))
            g = sns.boxplot(x='GCMs', y=field_names[var_name][1], data=df[df["scenario"]=="historical"], hue='scenario', ax=axes[i, 0],
                            showfliers=False, showmeans=True,
                            width=0.4,
                            meanprops={"marker": "o",
                                       "markerfacecolor": "white",
                                       "markeredgecolor": "black",
                                       "markersize": "5"})
        else:
            sns.set_palette(sns.color_palette([
                "#66A4C6",  # blue  hsl (142, 116, 164) in pickpick
                "#FC9630",  # orange  hsl (21, 247, 167) in pickpick
                ]))
            g = sns.boxplot(x='GCMs', y=field_names[var_name][1], data=df[df["scenario"]!="Historical"], hue='scenario', ax=axes[i, 1],
                            showfliers=False, showmeans=True, whis=box_whisker,
                            saturation=1.0,
                            width=0.4,
                            meanprops={"marker": "o",
                                       "markerfacecolor": "white",
                                       "markeredgecolor": "black",
                                       "markersize": "5"})
            sns.set_palette(sns.color_palette([
                "#96C468",
            ]))
            g = sns.boxplot(x='GCMs', y=field_names[var_name][1], data=df[df["scenario"]=="Historical"], hue='scenario', ax=axes[i, 0],
                            showfliers=False, showmeans=True, whis=box_whisker,
                            saturation=1.0,
                            width=0.2,
                            meanprops={"marker": "o",
                                       "markerfacecolor": "white",
                                       "markeredgecolor": "black",
                                       "markersize": "5"})

        for j in range(2):  # historical/future
            if i < len(variables) - 1:
                axes[i, j].set_xlabel('')
                axes[i, j].set_xticklabels([])
            else:
                axes[i, j].xaxis.set_tick_params(direction="out")
                axes[i, j].xaxis.set_ticks_position('bottom')
                for label in axes[i, j].get_xticklabels():
                    label.set_rotation(30)
                    label.set_ha('right')
            if j == 0:
                axes[i, j].set(xlabel="")
                axes[i, j].yaxis.set_tick_params(labelleft=True)
                axes[i, j].yaxis.set_label_position("left")
                axes[i, j].yaxis.set_ticks_position('left')
            else:
                axes[i, j].yaxis.set_tick_params(labelright=True)
                axes[i, j].yaxis.set_ticks_position('right')
                axes[i, j].set_ylabel('')

            axes[i, j].legend_.remove()
            axes[i, j].yaxis.set_tick_params(direction="out")

            axes[i, 0].spines['right'].set_visible(False)
            axes[i, 1].spines['left'].set_visible(False)

        if i == 0:
            for j in range(2):  # historical/future
                y_lim_top = axes[0, j].get_ylim()
                print(y_lim_top)
                axes[0, j].set(ylim=(y_lim_top[0], y_lim_top[1] * 1.1))

        axes[i, 1].axvline(0.5, color='gray', ls="--")
        axes[i, 1].axvline(-0.45, color='gray', ls="--")


    lines_labels = [ax.get_legend_handles_labels() for ax in fig.axes]
    lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
    fig.align_ylabels(axes[:])

    fig.legend(lines[:3], labels[:3], loc='upper right', ncol=3, bbox_to_anchor=(0.9, 0.94))
    plt.savefig(output_filename)

    plt.close()


def draw_bar_plot(base_dir, output_filename, models, variables, filename_pattern):
    sns.set(style="white", palette="muted", color_codes=True)
    params = {
        'legend.fontsize': 12,
        'axes.labelsize': 14,
        'axes.titlesize': 14,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
    }
    plt.rcParams.update(params)

    # Set up the matplotlib figure
    fig, axes = plt.subplots(len(variables), 2, figsize=(12, 8), sharex=True, sharey="row", dpi=300)
    sns.despine()

    color_pallete = [
        "#00A087", "#3C5488", "#E64B35",
    ]

    sns.set_palette(sns.color_palette(color_pallete))

    for i, var_name in enumerate(variables):
        df_hist = pd.read_csv(os.path.join(base_dir, filename_pattern.format(var_name)))
        for j, scenario in enumerate(scenarios):
            df = None
            for model_name in models:
                df_hist_copy = df_hist.copy()
                df_hist_copy["period"] = "historical"
                if model_name == "all_models_avg_9":
                    df_hist_copy["GCMs"] = "Avg."
                else:
                    df_hist_copy["GCMs"] = model_name
                if df is None:
                    df = df_hist_copy
                else:
                    df = pd.concat([df, df_hist_copy], ignore_index=True)
                for period_name in periods:
                    output_all_dir = os.path.join(base_dir, model_name)
                    output_all_dir_each = os.path.join(output_all_dir, scenario)
                    output_all_dir_each = os.path.join(output_all_dir_each, period_name)
                    df_model = pd.read_csv(os.path.join(output_all_dir_each, filename_pattern.format(var_name)))
                    if model_name == "all_models_avg_9":
                        df_model["GCMs"] = "Avg."
                    else:
                        df_model["GCMs"] = model_name
                    df_model["period"] = period_name
                    df = pd.concat([df, df_model], ignore_index=True)

            df = df.rename(columns={field_names[var_name][0]: field_names[var_name][1]})
            g = sns.barplot(x='GCMs', y=field_names[var_name][1], data=df, hue='period', ax=axes[i, j], capsize=.1, ci=None)

            if i == 0:
                axes[i, j].set_title(rcp_names[j])

            if i > 0 or j == 0:
                axes[i, j].legend_.remove()
            else:
                g.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

    plt.tight_layout()
    plt.savefig(output_filename)
    plt.close()


def write_summary(base_dir, output_filename, models, variables, filename_pattern):
    df_all = None
    for i, var_name in enumerate(variables):
        df_hist = pd.read_csv(os.path.join(base_dir, filename_pattern.format(var_name)))
        for j, scenario in enumerate(scenarios):
            df = None
            for model_name in models:
                df_hist_copy = df_hist.copy()
                df_hist_copy["period"] = "historical"
                df_hist_copy["scenario"] = scenario
                if model_name == "all_models_avg_9":
                    df_hist_copy["GCMs"] = "Avg."
                else:
                    df_hist_copy["GCMs"] = model_name
                if df is None:
                    df = df_hist_copy
                else:
                    df = pd.concat([df, df_hist_copy], ignore_index=True)
                for period_name in periods:
                    output_all_dir = os.path.join(base_dir, model_name)
                    output_all_dir_each = os.path.join(output_all_dir, scenario)
                    output_all_dir_each = os.path.join(output_all_dir_each, period_name)
                    df_model = pd.read_csv(os.path.join(output_all_dir_each, filename_pattern.format(var_name)))
                    if model_name == "all_models_avg_9":
                        df_model["GCMs"] = "Avg."
                    else:
                        df_model["GCMs"] = model_name
                    df_model["period"] = period_name
                    df_model["scenario"] = scenario
                    df = pd.concat([df, df_model], ignore_index=True)

            df = df.rename(columns={field_names[var_name][0]: field_names[var_name][1]})
            # g = sns.barplot(x='GCMs', y=field_names[var_name][1], data=df, hue='period', ax=axes[i, j], capsize=.1, ci=None)

            if df_all is None:
                df_all = df
            else:
                df_all = pd.concat([df_all, df], ignore_index=True)
    print(df_all[["scenario", "GCMs", "period", ]].head())
    column_list = ["scenario", "GCMs", "period", ]
    for var_name in variables:
        column_list.append(field_names[var_name][1])
    df_all = df_all[column_list]
    df_summary = df_all.groupby(["scenario", "GCMs", "period"]).describe().stack(level=0)[['25%', '50%', '75%', 'mean']]
    df_summary.to_csv(output_filename)


def main():
    models = [
        # Phase 1: Using 3 GCMs in first step
        "CanESM2",
        "CSIRO-Mk3-6-0",
        "inmcm4",

        # Phase 2: Add remained GCMs
        "CNRM-CM5",
        "GFDL-ESM2G",
        "MIROC-ESM",
        "MPI-ESM-MR",
        "MRI-CGCM3",
        "NorESM1-M",
        "all_models_avg_9",
    ]

    variables = [
        "fdd1",
        "fdd1_sc2_sctf",
        "gdd3_spring",
        "prcp_all",
        "snowfall_all",
    ]

    base_dir = "N:/WinterWheat/county_map/v20210111-for-Peng-rawdata"  # add more GCMs

    output_dir = "N:/WinterWheat/output/manuscript_20220215_9GCMs"
    make_dirs(output_dir)

    output_file = os.path.join(output_dir, "FigureS02_boxplot-stress-variables-ci95.pdf")
    draw_box_plot(base_dir, output_file, models, variables, "gridMET_{}_avg.csv", box_whisker=[5, 95])

    output_file = os.path.join(output_dir, "FigureS02_boxplot-stress-variables-ci95.csv")
    write_summary(base_dir, output_file, models, variables, "gridMET_{}_avg.csv")

    variables = [
        "snowfall_fall",
        "snowfall_winter",
        "snowfall_spring",
        "prcp_fall",
        "prcp_winter",
        "prcp_spring",
    ]
    output_file = os.path.join(output_dir, "FigureS03_boxplot-stress-variables-ci95.pdf")
    draw_box_plot_season(base_dir, output_file, models, variables, "gridMET_{}_avg.csv", box_whisker=[5, 95])
    output_file = os.path.join(output_dir, "FigureS03_boxplot-stress-variables-ci95.csv")
    write_summary(base_dir, output_file, models, variables, "gridMET_{}_avg.csv")


if __name__ == "__main__":
    main()
