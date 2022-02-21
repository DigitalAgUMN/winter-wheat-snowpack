import os
from typing import NamedTuple

import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from seaborn.categorical import _BarPlotter
from seaborn.utils import remove_na

from winter_wheat.plot.manuscript.fig03_barplot_effect_avg_error_bar import draw_confints_custom
from winter_wheat.util import get_project_root, make_dirs


class Coefficient(NamedTuple):
    name: str
    value: float
    SE: float


scenarios = ["rcp45", "rcp85"]
periods = ["2080-2100"]
field_names = {
    "fdd1": ("effect", "FDD\nEffect (%)",),
    "fdd1_sc2_sctf": ("effect", "SCF\nEffect (%)",),
    "gdd3_fall": ("effect", "GDD3_F",),
    "gdd3_winter": ("effect", "GDD3_W",),
    "gdd3_spring": ("effect", "GDD3_S",),
    "gdd1_all": ("effect", "GDD1",),
    "gdd2_all": ("effect", "GDD2",),
    "gdd3_all": ("effect", "GDD3",),
    "gdd_fall_all": ("effect", "GDD$_{fall}$\nEffect (%)",),
    "gdd_winter_all": ("effect", "GDD$_{winter}$\nEffect (%)",),
    "gdd_spring_all": ("effect", "GDD$_{spring}$\nEffect (%)",),
    "gdd_all": ("effect", "GDD\nEffect (%)",),
    "prcp_all": ("effect", "Rainfall\nEffect (%)",),
    "snowfall_all": ("effect", "Snowfall\nEffect (%)",),
}
rcp_names = ["RCP4.5", "RCP8.5"]


coeff = {
    "fdd1": Coefficient("fdd1", -0.00050632, 3.0832e-5),
    "gdd1_spring": Coefficient("gdd_low_spring", 2.1181e-5, 2.9334e-5),
    "gdd1_winter": Coefficient("gdd_low_winter", 0.00033099, 0.0001576),
    "gdd1_fall": Coefficient("gdd_low_fall", 0.00048502, 8.2739e-5),
    "gdd2_spring": Coefficient("gdd_med_spring", -0.00091571, 8.1836e-5),
    "gdd2_winter": Coefficient("gdd_med_winter", 0.00053237, 0.00023883),
    "gdd2_fall": Coefficient("gdd_med_fall", -0.00212, 0.00012421),
    "gdd3_spring": Coefficient("gdd_high_spring", -0.0029815, 0.00038717),
    "gdd3_winter": Coefficient("gdd_high_winter", -0.00076423, 0.00011462),
    "gdd3_fall": Coefficient("gdd_high_fall", 0.00093829, 5.724e-5),
    "prcp_fall": Coefficient("prcp_fall", 0.00026098, 2.1384e-5),
    "prcp_winter": Coefficient("prcp_winter", 0.00037849, 3.532e-5),
    "prcp_spring": Coefficient("prcp_spring", 0.00040059, 2.0319e-5),
    "snowfall_fall": Coefficient("snowfall_fall", 0.0016007, 0.00020739),
    "snowfall_winter": Coefficient("snowfall_winter", 0.00075981, 6.8538e-5),
    "snowfall_spring": Coefficient("snowfall_spring", -0.00029942, 0.00016845),
    "fdd1_sc2_sctf": Coefficient("fdd1_sc2_sctf", 0.00055051, 2.8933e-5),
}


def draw_bar_plot(base_dir, output_filename, models, variables, filename_pattern, five_state=False, sharedy=False, output_csv_file=None):
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
    fig, axes = plt.subplots(len(variables), 2, figsize=(12, 8), sharex=True, sharey=sharedy, dpi=300)
    # sns.despine()
    if "Figure06" in output_filename:
        plt.subplots_adjust(left=0.09, bottom=0.05, right=0.92, top=0.95, wspace=0.14, hspace=0.13)
    else:
        plt.subplots_adjust(left=0.09, bottom=0.05, right=0.92, top=0.95, wspace=0.10, hspace=0.10)


    color_pallete = [
        "#66A4C6",  # blue  hsl (142, 116, 164) in pickpick
        "#FC9630",  # orange  hsl (21, 247, 167) in pickpick
    ]

    sns.set_palette(sns.color_palette(color_pallete))

    raw_data = []

    for i, var_name in enumerate(variables):
        for j, scenario in enumerate(scenarios):
            bar_chart_data = {}

            for model_name in models:
                for period_name in periods:
                    for each_var_name in variables[var_name]:
                        df_hist = pd.read_csv(os.path.join(base_dir, "gridMET_{}_avg.csv".format(each_var_name)))
                        hist_mean = df_hist[coeff[each_var_name].name].mean()
                        if var_name == "fdd1_sc2_sctf":
                            df_hist["fdd1:sc2"] = df_hist["fdd1"] * df_hist["fdd1_sc2_sctf"]
                            hist_mean = df_hist["fdd1:sc2"].mean()

                        output_all_dir = os.path.join(base_dir, model_name)
                        output_all_dir_each = os.path.join(output_all_dir, scenario)
                        output_all_dir_each = os.path.join(output_all_dir_each, period_name)
                        df_model = pd.read_csv(os.path.join(output_all_dir_each, filename_pattern.format(each_var_name)))
                        if model_name == "all_models_avg_9":
                            df_model["GCMs"] = "Avg."
                            model_print_name = "Avg."
                        else:
                            df_model["GCMs"] = model_name
                            model_print_name = model_name
                        df_model["period"] = period_name
                        if var_name == "fdd1_sc2_sctf":
                            future_mean = np.mean(remove_na(df_model["fdd1_sc2_sctf"].to_numpy() * df_model["fdd1"].to_numpy()))
                        else:
                            future_mean = np.mean(remove_na(df_model[coeff[each_var_name].name].to_numpy()))

                        delta_mean = future_mean - hist_mean
                        effect = delta_mean * coeff[each_var_name].value
                        variance = (delta_mean * coeff[each_var_name].SE)**2
                        key = (model_print_name, period_name)
                        if key not in bar_chart_data:
                            bar_chart_data[key] = [effect, variance]
                        else:
                            bar_chart_data[key] = [bar_chart_data[key][0] + effect, bar_chart_data[key][1] + variance]

            bar_chart_list = []
            for key in bar_chart_data:
                model_print_name, period_name = key
                effect, variance = bar_chart_data[key]
                bar_chart_list.append([model_print_name, period_name, effect, variance])

            df = pd.DataFrame(bar_chart_list, columns=["GCMs", "period", field_names[var_name][1], "variance"])
            # print(df.head())

            plotter = _BarPlotter('GCMs', field_names[var_name][1], 'period', df, None, None,
                        np.mean, None, 1000, None, None,
                        None, None, None, 1.0,
                        ".26", None, None, True)
            plotter.plot(axes[i, j], {})
            confint = [[] for _ in plotter.plot_data]
            error_bar_cap = [[] for _ in plotter.plot_data]

            for ii, group_data in enumerate(plotter.plot_data):
                for jj, hue_level in enumerate(plotter.hue_names):
                    hue_mask = plotter.plot_hues[ii] == hue_level

                    stat_data = remove_na(group_data[hue_mask])
                    estimate = np.mean(stat_data)
                    select_gcm = models[ii]
                    if select_gcm == "all_models_avg_9":
                        select_gcm = "Avg."

                    variance = remove_na(df[(df["GCMs"] == select_gcm) & (df["period"] == hue_level)][["variance"]].to_numpy())[0]

                    if estimate >= 0:
                        confint[ii].append((estimate, estimate + np.sqrt(variance) * 1.96))
                        error_bar_cap[ii].append((False, True))
                        raw_data.append([scenario, select_gcm, hue_level, field_names[var_name][1], estimate, np.sqrt(variance) * 1.96, estimate + np.sqrt(variance) * 1.96])
                    else:
                        confint[ii].append((estimate - np.sqrt(variance) * 1.96, estimate))
                        error_bar_cap[ii].append((True, False))
                        raw_data.append([scenario, select_gcm, hue_level, field_names[var_name][1], estimate, np.sqrt(variance) * 1.96, estimate - np.sqrt(variance) * 1.96])

            plotter.confint = np.array(confint)
            # print(plotter.capsize)
            plotter.capsize = 0.15
            barpos = np.arange(len(plotter.plot_data))
            for kk, hue_level in enumerate(plotter.hue_names):
                offpos = barpos + plotter.hue_offsets[kk]
                # Draw the confidence intervals
                if plotter.confint.size:
                    confint = plotter.confint[:, kk]
                    errcolors = [plotter.errcolor] * len(offpos)
                    draw_confints_custom(
                        plotter,
                        axes[i, j],
                        offpos,
                        confint,
                        errcolors,
                        plotter.errwidth,
                        plotter.capsize,
                        [error_bar_cap[i][kk] for i in range(len(error_bar_cap))],)

            if i == 0:
                axes[i, j].set_title(rcp_names[j], fontsize=18, fontweight='bold')
            if i < len(variables) - 1:
                axes[i, j].set_xlabel('')
            else:
                axes[i, j].xaxis.set_tick_params(direction="out")
                axes[i, j].xaxis.set_ticks_position('bottom')
                for label in axes[i, j].get_xticklabels():
                    label.set_rotation(30)
                    label.set_ha('right')
            if j == 1:
                axes[i, j].yaxis.set_tick_params(labelleft=True)
                axes[i, j].yaxis.set_tick_params(labelright=True)
                axes[i, j].yaxis.set_label_position("right")

            axes[i, j].yaxis.set_tick_params(direction="out")
            axes[i, j].yaxis.set_ticks_position('both')
            axes[i, j].legend_.remove()

            axes[i, j].axhline(0, c="black", lw=0.5)

    for i, var_name in enumerate(variables):
        y1_min, y1_max = axes[i, 0].get_ylim()
        y2_min, y2_max = axes[i, 1].get_ylim()
        axes[i, 0].set(ylim=(min(y1_min, y2_min), max(y1_max, y2_max)))
        axes[i, 1].set(ylim=(min(y1_min, y2_min), max(y1_max, y2_max)))

    lines_labels = [ax.get_legend_handles_labels() for ax in fig.axes]
    lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
    fig.legend(lines[:1], labels[:1], loc='lower center', ncol=3, bbox_to_anchor=(0.5, 0.00))
    fig.subplots_adjust(bottom=0.17)

    plt.savefig(output_filename)
    plt.close()

    if output_csv_file is not None:
        df = pd.DataFrame(raw_data, columns=["scenario", "GCMs", "period", "variable_name", "effect_mean", "SE", "error_bar_endpoint"])
        df.to_csv(output_csv_file, index=False)


def write_summary(base_dir, output_filename, models, variables, filename_pattern, five_state=False):
    df_combine = pd.merge(
        pd.read_csv(os.path.join(base_dir, filename_pattern.format("fdd1"))),
        pd.read_csv(os.path.join(base_dir, "CanESM2/rcp45/2080-2100/" + filename_pattern.format("fdd1"))),
        on="county_fips")
    df_combine = df_combine.dropna()
    valid_county = df_combine["county_fips"].to_list()

    df_all = None
    for i, var_name in enumerate(variables):
        df_hist = pd.read_csv(os.path.join(base_dir, filename_pattern.format(var_name)))
        for j, scenario in enumerate(scenarios):
            df = None
            for model_name in models:
                df_hist_copy = df_hist.copy()

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
                    df_model = df_model.merge(df_hist_copy, on=["county_fips"])
                    df_model[field_names[var_name][0]] = df_model[field_names[var_name][0] + "_x"] - df_model[field_names[var_name][0] + "_y"]
                    df_model = df_model[df_model["county_fips"].isin(valid_county)]
                    df_model = df_model.dropna()
                    if df is None:
                        df = df_model
                    else:
                        df = pd.concat([df, df_model], ignore_index=True)

            df = df.rename(columns={field_names[var_name][0]: field_names[var_name][1]})
            df["state_fips"] = df["county_fips"].floordiv(1000)
            if five_state:
                df = df[df["state_fips"].isin([20, 40, 48, 8, 31])]  # Kansas, Oklahoma, Texas, Colorado, and Nebraska
            if df_all is None:
                df_all = df
            else:
                df_all = pd.concat([df_all, df], ignore_index=True)

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

    variables = {
        "fdd1": ("fdd1",),
        "fdd1_sc2_sctf": ("fdd1_sc2_sctf",),
        "gdd_all": (
            "gdd1_fall", "gdd1_winter", "gdd1_spring",
            "gdd2_fall", "gdd2_winter", "gdd2_spring",
            "gdd3_fall", "gdd3_winter", "gdd3_spring",
        ),
        "prcp_all": ("prcp_fall", "prcp_winter", "prcp_spring"),
        "snowfall_all": ("snowfall_fall", "snowfall_winter", "snowfall_spring"),
    }

    base_dir = "N:/WinterWheat/county_map/v20210111-for-Peng-rawdata"  # add more GCMs
    output_dir = "N:/WinterWheat/output/manuscript_20210308_9GCMs"
    make_dirs(output_dir)

    output_file = os.path.join(output_dir, "Figure06_barplot-stress-effect-future_gdp10p-hist_avg-errorbar.png")
    output_csv_file = os.path.join(output_dir, "Figure06_barplot-stress-effect-future_gdp10p-hist_avg-errorbar.csv")

    draw_bar_plot(base_dir, output_file, models, variables, "gridMET_{}_gdd10p.csv", output_csv_file=output_csv_file)


if __name__ == "__main__":
    main()
