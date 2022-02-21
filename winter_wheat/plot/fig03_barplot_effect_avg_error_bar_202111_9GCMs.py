import os
from typing import NamedTuple

import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from seaborn.categorical import _BarPlotter
from seaborn.utils import remove_na

from winter_wheat.util import get_project_root, make_dirs


class Coefficient(NamedTuple):
    name: str
    value: float
    SE: float


scenarios = ["rcp45", "rcp85"]
scenarios_title = {"rcp45": "RCP4.5", "rcp85": "RCP8.5"}
periods = ["2080-2100", ]
field_names = {
    "fdd1": ("effect", "FDD",),
    "fdd1_sc2_sctf": ("effect", "SCF",),
    "gdd3_fall": ("effect", "GDD3_F",),
    "gdd3_winter": ("effect", "GDD3_W",),
    "gdd3_spring": ("effect", "GDD3_S",),
    "gdd1_all": ("effect", "GDD1",),
    "gdd2_all": ("effect", "GDD2",),
    "gdd3_all": ("effect", "GDD3",),
    "gdd_fall_all": ("effect", "GDD$_{fall}$",),
    "gdd_winter_all": ("effect", "GDD$_{winter}$",),
    "gdd_spring_all": ("effect", "GDD$_{spring}$",),
    "gdd_all": ("effect", "GDD",),
    "prcp_all": ("effect", "Rainfall",),
    "snowfall_all": ("effect", "Snowfall",),
    "fdd_gdd_all": ("effect", "FDD+GDD",),
    "fdd_gdd_fall_all": ("effect", "FDD+GDD$_{fall}$",),
    "fdd_gdd_winter_all": ("effect", "FDD+GDD$_{winter}$",),
    "fdd_gdd_spring_all": ("effect", "FDD+GDD$_{spring}$",),
}
rcp_names = ["RCP4.5", "RCP8.5"]


coeff = {
    "fdd1": Coefficient("fdd1", -0.0004858, 3.0851e-05),
    "gdd1_spring": Coefficient("gdd_low_spring", 2.8465e-05, 2.9321e-05),
    "gdd1_winter": Coefficient("gdd_low_winter", 0.00034629, 0.00015747),
    "gdd1_fall": Coefficient("gdd_low_fall", 0.00048517, 8.3004e-05),
    "gdd2_spring": Coefficient("gdd_med_spring", -0.00093619, 8.1603e-05),
    "gdd2_winter": Coefficient("gdd_med_winter", 0.00047636, 0.00023806),
    "gdd2_fall": Coefficient("gdd_med_fall", -0.0020516, 0.0001243),
    "gdd3_spring": Coefficient("gdd_high_spring", -0.0028689, 0.00038701),
    "gdd3_winter": Coefficient("gdd_high_winter", -0.00072446, 0.00011394),
    "gdd3_fall": Coefficient("gdd_high_fall", 0.00089982, 5.7356e-05),
    "prcp_fall": Coefficient("prcp_fall", 0.00027762, 2.1424e-05),
    "prcp_winter": Coefficient("prcp_winter", 0.00034928, 3.5362e-05),
    "prcp_spring": Coefficient("prcp_spring", 0.00040437, 2.0428e-05),
    "snowfall_fall": Coefficient("snowfall_fall", 0.0015504, 0.00020758),
    "snowfall_winter": Coefficient("snowfall_winter", 0.00070676, 6.8557e-05),
    "snowfall_spring": Coefficient("snowfall_spring", -0.00031642, 0.00016836),
    "fdd1_sc2_sctf": Coefficient("fdd1_sc2_sctf", 0.00054959, 2.8993e-05),
    "fdd1_fall": Coefficient("fdd1_fall", -0.0004858, 3.0851e-05),
    "fdd1_winter": Coefficient("fdd1_winter", -0.0004858, 3.0851e-05),
    "fdd1_spring": Coefficient("fdd1_spring", -0.0004858, 3.0851e-05),
}


def draw_bar_plot_only_avg(base_dir, output_filename, models, variables, filename_pattern, five_state=False, sharedy=False, output_csv_file=None, color_pallete=None):
    acre_file = get_project_root() / "output/county_map/v20200814-future-trend/prod/winter_wheat_acre_avg.csv"
    df_acre = pd.read_csv(acre_file)

    sns.set(style="white", palette="muted", color_codes=True)
    params = {
        'legend.fontsize': 20,
        'axes.labelsize': 30,
        'axes.titlesize': 20,
        'xtick.labelsize': 20,
        'ytick.labelsize': 20,
    }
    plt.rcParams.update(params)

    # Set up the matplotlib figure
    fig, axes = plt.subplots(1, 2, figsize=(8.5, 4), gridspec_kw={"width_ratios": [3.6, 2.6], "wspace": 0}, sharey=True, dpi=300, constrained_layout=False)
    # sns.despine()
    # if "Figure03" in output_filename:
    #     plt.subplots_adjust(left=0.11, bottom=0.10, right=0.92, top=0.95, wspace=0.14, hspace=0.18)
    # else:
    #     plt.subplots_adjust(left=0.11, bottom=0.05, right=0.92, top=0.95, wspace=0.10, hspace=0.10)
    plt.subplots_adjust(left=0.14, bottom=0.10, right=0.98, top=0.95, wspace=0.14, hspace=0.18)

    if color_pallete is None:
        color_pallete = [  # "#00A087",
            # "#3C5488", "#E64B35",
            # "#D6E5C7",  # green  hsl (64, 93, 214) in pickpick
            # "#7AB0CD",  # blue  hsl (142, 116, 164) in pickpick
            # "#FCA652",  # orange  hsl (21, 247, 167) in pickpick
            # "#96C468",  # green  hsl (64, 112, 150) in pickpick
            "#66A4C6",  # blue  hsl (142, 116, 164) in pickpick
            "#FC9630",  # orange  hsl (21, 247, 167) in pickpick
        ]
    # color_pallete = [
    #     sns.color_palette()[1],
    #     sns.color_palette()[4],
    # ]
    sns.set_palette(sns.color_palette(color_pallete))

    raw_data = []

    bar_chart_data = {}
    for i, var_name in enumerate(variables):
        for j, scenario in enumerate(scenarios):
            for model_name in models:
                for period_name in periods:
                    for each_var_name in variables[var_name]:
                        df_hist = pd.read_csv(os.path.join(base_dir, filename_pattern.format(each_var_name)))
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
                        df_model["scenario"] = scenario

                        df_combine = pd.merge(df_hist, df_model, on="county_fips")
                        df_combine = pd.merge(df_combine, df_acre, on="county_fips")
                        df_combine = df_combine.dropna()

                        if var_name == "fdd1_sc2_sctf":
                            hist_mean = np.average(df_combine["fdd1:sc2_sctf_x"].to_numpy(), weights=df_combine["acre_survey"].to_numpy())
                            future_mean = np.average(df_combine["fdd1:sc2_sctf_y"].to_numpy(), weights=df_combine["acre_survey"].to_numpy())
                        else:
                            hist_mean = np.average(df_combine[coeff[each_var_name].name + "_x"].to_numpy(), weights=df_combine["acre_survey"].to_numpy())
                            future_mean = np.average(df_combine[coeff[each_var_name].name + "_y"].to_numpy(), weights=df_combine["acre_survey"].to_numpy())

                        delta_mean = future_mean - hist_mean

                        print_key = (model_print_name, scenario, var_name, each_var_name)
                        print("delta mean of {}: {}".format(print_key, delta_mean))
                        if var_name == "fdd1_sc2_sctf":
                            hist_fdd1_scf = np.mean(remove_na(df_combine["fdd1:sc2_sctf_x"].to_numpy()))
                            future_fdd1_scf = np.mean(remove_na(df_combine["fdd1:sc2_sctf_y"].to_numpy()))
                            hist_scf = np.mean(remove_na(df_combine["fdd1_sc2_sctf_x"].to_numpy()))
                            future_scf = np.mean(remove_na(df_combine["fdd1_sc2_sctf_y"].to_numpy()))

                            print("delta mean of {} (FDD1 * SCTF): {} = {} - {}".format(print_key,
                                                                                        future_fdd1_scf - hist_fdd1_scf,
                                                                                        future_fdd1_scf,
                                                                                        hist_fdd1_scf))
                            print("delta mean of {} (SCTF): {} = {} - {}".format(print_key, future_scf - hist_scf,
                                                                                 future_scf, hist_scf))

                        delta_mean *= 100  # convert to % unit
                        effect = delta_mean * coeff[each_var_name].value

                        variance = (delta_mean * coeff[each_var_name].SE)**2
                        key = (model_print_name, scenario, var_name)
                        if key not in bar_chart_data:
                            bar_chart_data[key] = [effect, variance]
                        else:
                            bar_chart_data[key] = [bar_chart_data[key][0] + effect, bar_chart_data[key][1] + variance]
                key = (model_print_name, scenario, var_name)

    draw_subplot_for_avg(bar_chart_data, raw_data, ax=axes[0], selected_variables=["fdd1", "fdd1_sc2_sctf", "gdd_all"])
    draw_subplot_for_avg(bar_chart_data, raw_data, ax=axes[1], selected_variables=["prcp_all", "snowfall_all"])

    # axes[i].set_xlabel('')
    for i in range(2):
        axes[i].xaxis.set_tick_params(direction="out")
        axes[i].xaxis.set_ticks_position('bottom')

        if i == 0:
            axes[i].yaxis.set_tick_params(labelleft=True)
            axes[i].yaxis.set_tick_params(labelright=False)
            axes[i].yaxis.set_label_position("left")
            axes[i].yaxis.set_ticks_position('left')
            axes[i].set(ylabel="Yield Effect (%)")
            axes[i].set(xlim=(-0.8, 2.8))
        elif i == 1:
            axes[i].yaxis.set_tick_params(labelleft=False)
            axes[i].yaxis.set_tick_params(labelright=False)
            axes[i].yaxis.set_ticks_position('none')
            axes[i].set(ylabel="")
            axes[i].set(xlim=(-0.8, 1.8))

        axes[i].yaxis.set_tick_params(direction="out")
        axes[i].legend_.remove()
        axes[i].set(xlabel="")

        axes[i].axhline(0, c="black", lw=0.5)
        axes[i].set_yticks([-15, -10, -5, 0, 5, 10])

    lines_labels = [ax.get_legend_handles_labels() for ax in fig.axes]
    lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
    fig.align_ylabels(axes[:])
    fig.legend(lines[:2], labels[:2], loc='lower right', ncol=1, bbox_to_anchor=(0.98, 0.15))

    plt.savefig(output_filename)
    plt.close()

    if output_csv_file is not None:
        df = pd.DataFrame(raw_data, columns=["scenario", "GCMs", "period", "variable_name", "effect_mean", "SE", "error_bar_endpoint"])
        df.to_csv(output_csv_file, index=False)


def draw_subplot_for_avg(bar_chart_data, raw_data, ax=None, selected_variables=None):
    bar_chart_list = []
    for key in bar_chart_data:
        model_print_name, scenario_name, var_name = key
        if var_name in selected_variables:
            effect, variance = bar_chart_data[key]
            bar_chart_list.append([model_print_name, scenarios_title[scenario_name], field_names[var_name][1], effect, variance])

    df = pd.DataFrame(bar_chart_list, columns=["GCMs", "scenario", "variable", "effect", "variance"])

    plotter = _BarPlotter('variable', "effect", 'scenario', df, None, None,
                np.mean, None, 1000, None, None,
                None, None, None, 1.0,
                ".26", None, None, True)
    plotter.plot(ax, {})
    confint = [[] for _ in plotter.plot_data]
    error_bar_cap = [[] for _ in plotter.plot_data]

    for ii, group_data in enumerate(plotter.plot_data):
        for jj, hue_level in enumerate(plotter.hue_names):
            hue_mask = plotter.plot_hues[ii] == hue_level
            stat_data = remove_na(group_data[hue_mask])
            estimate = np.mean(stat_data)

            variance = remove_na(df[(df["scenario"] == hue_level) & (df["variable"] == field_names[selected_variables[ii]][1])][["variance"]].to_numpy())[0]

            if estimate >= 0:
                confint[ii].append((estimate, estimate + np.sqrt(variance) * 1.96))
                error_bar_cap[ii].append((False, True))
                raw_data.append([scenarios[jj], "Avg.", hue_level, field_names[selected_variables[ii]][1], estimate, np.sqrt(variance) * 1.96, estimate + np.sqrt(variance) * 1.96])
            else:
                confint[ii].append((estimate - np.sqrt(variance) * 1.96, estimate))
                error_bar_cap[ii].append((True, False))
                raw_data.append([scenarios[jj], "Avg.", hue_level, field_names[selected_variables[ii]][1], estimate, np.sqrt(variance) * 1.96, estimate - np.sqrt(variance) * 1.96])

    plotter.confint = np.array(confint)
    plotter.capsize = 0.15
    barpos = np.arange(len(plotter.plot_data))
    for kk, hue_level in enumerate(plotter.hue_names):
        offpos = barpos + plotter.hue_offsets[kk]
        # Draw the confidence intervals
        if plotter.confint.size:
            # print("confint", plotter.confint)
            confint = plotter.confint[:, kk]
            errcolors = [plotter.errcolor] * len(offpos)

            draw_confints_custom(
                plotter,
                ax,
                offpos,
                confint,
                errcolors,
                plotter.errwidth,
                plotter.capsize,
                [error_bar_cap[i][kk] for i in range(len(error_bar_cap))],)



def draw_confints_custom(plotter, ax, at_group, confint, colors,
                  errwidth=None, capsize=None, errorbar_cap=None, **kws):

    if errwidth is not None:
        kws.setdefault("lw", errwidth)
    else:
        kws.setdefault("lw", mpl.rcParams["lines.linewidth"] * 1)

    for at, (ci_low, ci_high), color, (is_cap_low, is_cap_high) in zip(at_group,
                                                                       confint,
                                                                       colors,
                                                                       errorbar_cap):

        if plotter.orient == "v":
            ax.plot([at, at], [ci_low, ci_high], color=color, **kws)
            if capsize is not None:
                if is_cap_low:
                    ax.plot([at - capsize / 2, at + capsize / 2],
                            [ci_low, ci_low], color=color, **kws)
                if is_cap_high:
                    ax.plot([at - capsize / 2, at + capsize / 2],
                            [ci_high, ci_high], color=color, **kws)
        else:
            ax.plot([ci_low, ci_high], [at, at], color=color, **kws)
            if capsize is not None:
                if is_cap_low:
                    ax.plot([ci_low, ci_low],
                            [at - capsize / 2, at + capsize / 2],
                            color=color, **kws)
                if is_cap_high:
                    ax.plot([ci_high, ci_high],
                            [at - capsize / 2, at + capsize / 2],
                            color=color, **kws)


def write_summary(base_dir, output_filename, models, variables, filename_pattern, five_state=False):
    acre_file = get_project_root() / "output/county_map/v20200814-future-trend/prod/winter_wheat_acre_avg.csv"
    df_acre = pd.read_csv(acre_file)

    df_combine = pd.merge(
        pd.read_csv(os.path.join(base_dir, filename_pattern.format("fdd1"))),
        pd.read_csv(os.path.join(base_dir, "CanESM2/rcp45/2080-2100/" + filename_pattern.format("fdd1"))),
        on="county_fips")
    df_combine = pd.merge(df_combine, df_acre, on="county_fips")
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
            # g = sns.barplot(x='GCMs', y=field_names[var_name][1], data=df, hue='period', ax=axes[i, j], ci=None)
            if df_all is None:
                df_all = df
            else:
                df_all = pd.concat([df_all, df], ignore_index=True)

    column_list = ["scenario", "GCMs", "period", ]
    for var_name in variables:
        column_list.append(field_names[var_name][1])
    df_all = df_all[column_list]

    df_all.to_csv(output_filename+"_test.csv")
    df_summary = df_all.groupby(["scenario", "GCMs", "period"]).describe().stack(level=0)[['25%', '50%', '75%', 'mean']]
    df_summary.to_csv(output_filename)


def draw_new_figures():
    models = [
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

    output_dir = "N:/WinterWheat/output/manuscript_20211130_9GCMs"

    make_dirs(output_dir)

    output_file = os.path.join(output_dir, "Figure03_barplot-stress-effect-avg-errorbar.png")
    output_csv_file = os.path.join(output_dir, "Figure03_barplot-stress-effect-avg-errorbar.csv")

    draw_bar_plot_only_avg(base_dir, output_file, models, variables, "gridMET_{}_avg.csv", output_csv_file=output_csv_file)

    # output_file = os.path.join(output_dir, "Figure03_barplot-stress-effect-avg.csv")
    # write_summary(base_dir, output_file, models, variables, "gridMET_{}_avg.csv")


def main():
    draw_new_figures()  # only average model


if __name__ == "__main__":
    main()
