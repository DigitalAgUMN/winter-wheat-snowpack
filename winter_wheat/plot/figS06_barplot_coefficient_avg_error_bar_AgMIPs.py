import csv
import os
from typing import NamedTuple

import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from PIL import Image
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
    "prcp_fall": ("effect", "Rainfall$_{fall}$",),
    "prcp_winter": ("effect", "Rainfall$_{winter}$",),
    "prcp_spring": ("effect", "Rainfall$_{spring}$",),
    "snowfall_all": ("effect", "Snowfall",),
    "snowfall_fall": ("effect", "Snowfall$_{fall}$",),
    "snowfall_winter": ("effect", "Snowfall$_{winter}$",),
    "snowfall_spring": ("effect", "Snowfall$_{spring}$",),
    "fdd_gdd_all": ("effect", "FDD+GDD",),
    "fdd_gdd_fall_all": ("effect", "FDD+GDD$_{fall}$",),
    "fdd_gdd_winter_all": ("effect", "FDD+GDD$_{winter}$",),
    "fdd_gdd_spring_all": ("effect", "FDD+GDD$_{spring}$",),
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
    "fdd1_fall": Coefficient("fdd1_fall", -0.00050632, 3.0832e-5),
    "fdd1_winter": Coefficient("fdd1_winter", -0.00050632, 3.0832e-5),
    "fdd1_spring": Coefficient("fdd1_spring", -0.00050632, 3.0832e-5),
}

MODEL_AgMIP_NAMES = [
    "APSIM",
    "EPIC_IIASA",
    "EPIC-TAMU",
    "GEPIC",
    "LPJ_GUESS",
    "pDSSAT",
    "PEPIC"
]

model_agmip_name = {
    "APSIM": "APSIM UGOE",
    "EPIC_IIASA": "EPIC IIASA",
    "EPIC-TAMU": "EPIC TAMU",
    "GEPIC": "GEPIC",
    "LPJ_GUESS": "LPJ GUESS",
    "pDSSAT": "pDSSAT",
    "PEPIC": "PEPIC",
}

coeff_dict = None


def update_parameter(agmip_model_name):
    global coeff
    global coeff_dict

    if agmip_model_name == None:
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
            "fdd1_fall": Coefficient("fdd1_fall", -0.00050632, 3.0832e-5),
            "fdd1_winter": Coefficient("fdd1_winter", -0.00050632, 3.0832e-5),
            "fdd1_spring": Coefficient("fdd1_spring", -0.00050632, 3.0832e-5),
        }
        return

    if coeff_dict is None:
        build_coeff_dict()

    coeff = coeff_dict[agmip_model_name]


def get_model_coeff_linedata(agmip_model_name):
    if coeff_dict is None:
        build_coeff_dict()
    if agmip_model_name not in coeff_dict:
        print("{} is not found in coeff_dict".format(agmip_model_name))

    return (0, 1), (coeff_dict[agmip_model_name]["fdd1"].value * 100,
                    (coeff_dict[agmip_model_name]["fdd1"].value + coeff_dict[agmip_model_name]["fdd1_sc2_sctf"].value) * 100)


def build_coeff_dict():
    coeff_index_info = [
        ("fdd1", "fdd1", 2),
        ("gdd1_spring", "gdd_low_spring", 3),
        ("gdd1_winter", "gdd_low_winter", 4),
        ("gdd1_fall", "gdd_low_fall", 5),
        ("gdd2_spring", "gdd_med_spring", 6),
        ("gdd2_winter", "gdd_med_winter", 7),
        ("gdd2_fall", "gdd_med_fall", 8),
        ("gdd3_spring", "gdd_high_spring", 9),
        ("gdd3_winter", "gdd_high_winter", 10),
        ("gdd3_fall", "gdd_high_fall", 11),
        ("prcp_fall", "prcp_fall", 12),
        ("prcp_winter", "prcp_winter", 13),
        ("prcp_spring", "prcp_spring", 14),
        ("snowfall_fall", "snowfall_fall", 15),
        ("snowfall_winter", "snowfall_winter", 16),
        ("snowfall_spring", "snowfall_spring", 17),
        ("fdd1_sc2_sctf", "fdd1_sc2_sctf", 1),
        ("fdd1_fall", "fdd1_fall", 2),
        ("fdd1_winter", "fdd1_winter", 2),
        ("fdd1_spring", "fdd1_spring", 2),
    ]
    global coeff_dict
    coeff_dict = {}
    with open(os.path.join(get_project_root() / "input/AgMIP/coef_agmip.csv")) as f1:
        with open(os.path.join(get_project_root() / "input/AgMIP/coef_SE_agmip.csv")) as f2:
            csv_reader_coeff = csv.reader(f1)
            csv_reader_se = csv.reader(f2)
            next(csv_reader_coeff)  # skip header

            for row_coeff, row_se in zip(csv_reader_coeff, csv_reader_se):
                coeff_data_for_model = {}
                for coeff_name, coeff_field, coeff_idx in coeff_index_info:
                    coeff_data_for_model[coeff_name] = Coefficient(coeff_field, float(row_coeff[coeff_idx]),
                                                                   float(row_se[coeff_idx - 1]))
                coeff_dict[row_coeff[0]] = coeff_data_for_model


def draw_bar_plot_three_pane(base_dir, output_filename, models, variables, agmip_models, filename_pattern, five_state=False, sharedy=False, output_csv_file=None, color_palette=None):
    acre_file = get_project_root() / "output/county_map/v20200814-future-trend/prod/winter_wheat_acre_avg.csv"
    df_acre = pd.read_csv(acre_file)

    sns.set(style="white", palette="muted", color_codes=True)
    params = {
        'legend.fontsize': 12,
        'legend.handlelength': 2,
        'axes.labelsize': 14,
        'axes.titlesize': 14,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
    }

    plt.rcParams.update(params)

    # Set up the matplotlib figure
    fig, axes_all = plt.subplots(3, 2, figsize=(11, 9), gridspec_kw={"height_ratios": [1, 1, 1]}, dpi=300, constrained_layout=False)

    plt.subplots_adjust(left=0.09, bottom=0.02, right=0.94, top=0.96, wspace=0.05, hspace=0.18)

    if color_palette is None:
        color_palette = [
            "#FDC883",
            "#FF870F",
            "#C1E5A1",
            "#37A230",
            "#FB9A99",
            "#E62F31",
            "#CDB6D8",
        ]

    sns.set_palette(sns.color_palette(color_palette))

    raw_data = []

    bar_chart_data = {}
    scenario = "rcp85"
    model_name = "all_models_avg_9"
    period_name = "2080-2100"

    update_parameter(None)
    for i, each_var_name in enumerate(variables):
        df_hist = pd.read_csv(os.path.join(base_dir, filename_pattern.format(each_var_name)))
        output_all_dir = os.path.join(base_dir, model_name)
        output_all_dir_each = os.path.join(output_all_dir, scenario)
        output_all_dir_each = os.path.join(output_all_dir_each, period_name)
        df_model = pd.read_csv(os.path.join(output_all_dir_each, filename_pattern.format(each_var_name)))
        if model_name == "all_models_avg_9":
            df_model["GCMs"] = "Avg."
            df_model["AgMIP"] = "ThisStudy"
            model_print_name = "ThisStudy"
        else:
            df_model["GCMs"] = model_name
            df_model["AgMIP"] = "ThisStudy"
            model_print_name = "ThisStudy"
        df_model["period"] = period_name
        df_model["scenario"] = scenario

        df_combine = pd.merge(df_hist, df_model, on="county_fips")
        df_combine = pd.merge(df_combine, df_acre, on="county_fips")
        df_combine = df_combine.dropna()

        effect = coeff[each_var_name].value
        variance = coeff[each_var_name].SE ** 2
        key = (model_print_name, scenario, each_var_name)
        if key not in bar_chart_data:
            bar_chart_data[key] = [effect, variance]
        else:
            bar_chart_data[key] = [bar_chart_data[key][0] + effect, bar_chart_data[key][1] + variance]

    for agmip_name in agmip_models:
        update_parameter(agmip_name)
        for i, each_var_name in enumerate(variables):
            df_hist = pd.read_csv(os.path.join(base_dir, filename_pattern.format(each_var_name)))

            output_all_dir = os.path.join(base_dir, model_name)
            output_all_dir_each = os.path.join(output_all_dir, scenario)
            output_all_dir_each = os.path.join(output_all_dir_each, period_name)
            df_model = pd.read_csv(os.path.join(output_all_dir_each, filename_pattern.format(each_var_name)))
            if model_name == "all_models_avg_9":
                df_model["GCMs"] = "Avg."
                df_model["AgMIP"] = model_agmip_name[agmip_name]
                model_print_name = model_agmip_name[agmip_name]
            else:
                df_model["GCMs"] = model_name
                df_model["AgMIP"] = model_agmip_name[agmip_name]
                model_print_name = model_agmip_name[agmip_name]
            df_model["period"] = period_name
            df_model["scenario"] = scenario

            df_combine = pd.merge(df_hist, df_model, on="county_fips")
            df_combine = pd.merge(df_combine, df_acre, on="county_fips")
            df_combine = df_combine.dropna()

            effect = coeff[each_var_name].value
            variance = coeff[each_var_name].SE**2

            key = (model_print_name, scenario, each_var_name)
            if key not in bar_chart_data:
                bar_chart_data[key] = [effect, variance]
            else:
                bar_chart_data[key] = [bar_chart_data[key][0] + effect, bar_chart_data[key][1] + variance]

    draw_subplot_for_agmip(bar_chart_data, raw_data, ax=axes_all[0, 0],
                           selected_variables=["prcp_fall", ], obs_color=color_palette[-1])
    draw_subplot_for_agmip(bar_chart_data, raw_data, ax=axes_all[1, 0],
                           selected_variables=["prcp_winter", ], obs_color=color_palette[-1])
    draw_subplot_for_agmip(bar_chart_data, raw_data, ax=axes_all[2, 0],
                           selected_variables=["prcp_spring", ], obs_color=color_palette[-1])
    draw_subplot_for_agmip(bar_chart_data, raw_data, ax=axes_all[0, 1],
                           selected_variables=["snowfall_fall", ], obs_color=color_palette[-1])
    draw_subplot_for_agmip(bar_chart_data, raw_data, ax=axes_all[1, 1],
                           selected_variables=["snowfall_winter", ], obs_color=color_palette[-1])
    draw_subplot_for_agmip(bar_chart_data, raw_data, ax=axes_all[2, 1],
                           selected_variables=["snowfall_spring", ], obs_color=color_palette[-1])

    ylabels = [
        ["Rainfall$_{fall}$", "Snowfall$_{fall}$"],
        ["Rainfall$_{winter}$", "Snowfall$_{winter}$"],
        ["Rainfall$_{spring}$", "Snowfall$_{spring}$"]

        ]

    for r in range(3):
        for c in range(2):
            ax = axes_all[r, c]
            ax.xaxis.set_tick_params(labelbottom=False)
            ax.xaxis.set_ticks_position('none')
            ax.set(xlabel="", )
            ax.set(title=ylabels[r][c])
            if c == 0:
                ax.set(ylabel="Estimated coefficients")
            else:
                ax.set(ylabel="")

            if c == 0:
                ax.yaxis.set_tick_params(labelleft=True)
                ax.yaxis.set_tick_params(labelright=False)
                ax.yaxis.set_label_position("left")
                ax.yaxis.set_ticks_position('both')
                ax.set(xlim=(-0.5, 0.5))
            elif c == 1:
                ax.yaxis.set_tick_params(labelleft=False)
                ax.yaxis.set_tick_params(labelright=True)
                ax.yaxis.set_label_position("right")
                ax.yaxis.set_ticks_position('both')
                ax.set(xlim=(-0.5, 0.5))

            ax.yaxis.set_tick_params(direction="in")
            ax.legend_.remove()
            ax.set(xlabel="")

            y_lim = ax.get_ylim()
            ax.set(ylim=(y_lim[0]*1.1, y_lim[1]*1.1))

            ax.axhline(0, c="black", lw=1.5)

    y_lim = axes_all[0, 1].get_ylim()
    axes_all[0, 1].set(ylim=(y_lim[0], y_lim[1]*1.3))

    leg = axes_all[0, 1].legend(loc='upper right', bbox_to_anchor=(0.98, 0.98),
                ncol=2, fancybox=False, shadow=False, frameon=False)

    legend_dict = {}
    for ci, agmip_name in enumerate(agmip_models):
        legend_dict[agmip_name] = color_palette[ci]
    legend_dict["Observation"] = color_palette[-1]

    patchList = []
    for key in legend_dict:
        data_key = mpatches.Patch(color=legend_dict[key], label=key)
        patchList.append(data_key)

    axes_all[0, 1].legend(handles=patchList, loc='upper right', bbox_to_anchor=(0.98, 0.98),
                          ncol=2, fancybox=False, shadow=False, frameon=True)

    for line in leg.get_lines():
        line.set_linewidth(6)


    lines_labels = [ax.get_legend_handles_labels() for ax in fig.axes]
    lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]

    fig.align_ylabels(axes_all[:, 0])
    fig.align_ylabels(axes_all[:, 1])

    plt.savefig(output_filename)
    plt.close()

    if output_csv_file is not None:
        df = pd.DataFrame(raw_data,
                          columns=["scenario", "GCMs", "AgMIP", "variable_name", "effect_mean", "SE",
                                   "error_bar_endpoint"])
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

            variance = remove_na(df[(df["scenario"] == hue_level)][["variance"]].to_numpy())[0]

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


def draw_subplot_for_agmip(bar_chart_data, raw_data, ax=None, selected_variables=None, obs_color=None):
    bar_chart_list = []
    bar_chart_this_study = []

    for key in bar_chart_data:
        model_print_name, scenario_name, var_name = key
        if model_print_name == "ThisStudy":
            if var_name in selected_variables:
                effect, variance = bar_chart_data[key]
                bar_chart_this_study.append(
                    [model_print_name, scenarios_title[scenario_name], field_names[var_name][1], effect, variance])
        else:
            if var_name in selected_variables:
                effect, variance = bar_chart_data[key]
                bar_chart_list.append([model_print_name, scenarios_title[scenario_name], field_names[var_name][1], effect, variance])

    df = pd.DataFrame(bar_chart_list, columns=["AgMIP", "scenario", "variable", "effect", "variance"])

    plotter = _BarPlotter('variable', "effect", 'AgMIP', df, None, None,
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

            variance = remove_na(df[(df["AgMIP"] == hue_level)][["variance"]].to_numpy())[0]

            if estimate >= 0:
                confint[ii].append((estimate, estimate + np.sqrt(variance) * 1.96))
                error_bar_cap[ii].append((False, True))
                raw_data.append([scenarios[1], "Avg.", hue_level, field_names[selected_variables[ii]][1], estimate, np.sqrt(variance) * 1.96, estimate + np.sqrt(variance) * 1.96])
            else:
                confint[ii].append((estimate - np.sqrt(variance) * 1.96, estimate))
                error_bar_cap[ii].append((True, False))
                raw_data.append([scenarios[1], "Avg.", hue_level, field_names[selected_variables[ii]][1], estimate, np.sqrt(variance) * 1.96, estimate - np.sqrt(variance) * 1.96])

    plotter.confint = np.array(confint)
    plotter.capsize = 0.08
    barpos = np.arange(len(plotter.plot_data))
    for kk, hue_level in enumerate(plotter.hue_names):
        offpos = barpos + plotter.hue_offsets[kk]
        # Draw the confidence intervals
        if plotter.confint.size:
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

    x = [-0.5, 0.5]
    y_est = bar_chart_this_study[0][3]
    y_err = np.sqrt(bar_chart_this_study[0][4]) * 1.96
    if obs_color is None:
        obs_color = "#7C8DB0"

    ax.plot(x, [y_est, y_est], '--', color=obs_color)
    ax.fill_between(x, y_est - y_err, y_est + y_err, alpha=0.3, color=obs_color, zorder=2)


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


def write_summary(base_dir, output_filename, models, variables, agmip_model, filename_pattern, five_state=False):
    update_parameter(agmip_model)
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

    df_all.to_csv(output_filename+"_test.csv")
    df_summary = df_all.groupby(["scenario", "GCMs", "period"]).describe().stack(level=0)[['25%', '50%', '75%', 'mean']]
    df_summary.to_csv(output_filename)


def draw_barchart_together(color_palette=None):
    models = [
        "all_models_avg_9",
    ]

    variables = [
        "prcp_fall", "prcp_winter", "prcp_spring",
        "snowfall_fall", "snowfall_winter", "snowfall_spring"
    ]

    base_dir = "N:/WinterWheat/county_map/v20210111-for-Peng-rawdata"  # add more GCMs
    output_dir = "N:/WinterWheat/output/manuscript_20220215_9GCMs"

    make_dirs(output_dir)

    output_file = os.path.join(output_dir, "FigureS06_barplot-coeff-avg-errorbar.pdf")
    output_csv_file = os.path.join(output_dir, "Figure06_barplot-coeff-avg-errorbar.csv")

    draw_bar_plot_three_pane(base_dir, output_file, models, variables, MODEL_AgMIP_NAMES, "gridMET_{}_avg.csv",
                             output_csv_file=output_csv_file, color_palette=color_palette)


def draw_lineplot(output_filename, color_palette=None):
    sns.set(style="white", palette="muted", color_codes=True)

    params = {
        'legend.fontsize': 20,
        'legend.handlelength': 3,
        'axes.labelsize': 23,
        'axes.titlesize': 18,
        'xtick.labelsize': 20,
        'ytick.labelsize': 20,
        'axes.linewidth': 1.5,
    }

    plt.rcParams.update(params)

    # Set up the matplotlib figure
    fig, axes_all = plt.subplots(1, 1, figsize=(18, 6), sharey=True, dpi=300, constrained_layout=False)
    plt.subplots_adjust(left=0.12, bottom=0.17, right=0.5, top=0.98, wspace=0.14, hspace=0.18)

    if color_palette is None:
        color_palette = [
            "#FDC883",
            "#FF870F",
            "#C1E5A1",
            "#37A230",
            "#FB9A99",
            "#E62F31",
            "#CDB6D8",
        ]

    sns.set_palette(sns.color_palette(color_palette))

    axes = axes_all
    axes.axhline(0, c="black", lw=2.0)

    for i, model_agmip in enumerate(MODEL_AgMIP_NAMES):
        line_data = get_model_coeff_linedata(model_agmip)
        axes.plot(line_data[0], line_data[1], lw=6, label=model_agmip_name[model_agmip], c=color_palette[i],
                  alpha=1.0)

        axes.xaxis.set_tick_params(direction="out")
        axes.xaxis.set_ticks_position('bottom')

        axes.yaxis.set_tick_params(labelleft=True)
        axes.yaxis.set_tick_params(labelright=False)
        axes.yaxis.set_label_position("left")
        axes.yaxis.set_ticks_position('left')
        axes.set(ylabel="Sensitivity to FDD (%/degree days)", )
        axes.set(xlim=(0, 1.0))

        axes.yaxis.set_tick_params(direction="out")
        axes.set(xlabel="Snow cover fraction")

    # Draw line for This study
    update_parameter(None)
    p = axes.plot((0, 1),
                  (coeff["fdd1"].value * 100, (coeff["fdd1"].value + coeff["fdd1_sc2_sctf"].value) * 100,),
                  lw=6, label="Observation", c=color_palette[-1], alpha=1.0)

    leg = axes.legend(loc='upper left', bbox_to_anchor=(1.86, 1.05),
                ncol=1, fancybox=False, shadow=False, frameon=False)

    for line in leg.get_lines():
        line.set_linewidth(6)

    plt.text(0.7139, 0.0485, 'Climate variables',
             verticalalignment='bottom', horizontalalignment='center',
             transform=plt.gcf().transFigure,
             color='black', fontsize=23)

    plt.text(0.01, 0.99, '(a)',
             verticalalignment='top', horizontalalignment='left',
             transform=plt.gcf().transFigure,
             color='black', fontsize=30)

    plt.text(0.51, 0.99, '(b)',
             verticalalignment='top', horizontalalignment='left',
             transform=plt.gcf().transFigure,
             color='black', fontsize=30)

    plt.savefig(output_filename)
    plt.close()


def merge_two_figure(output_filename, lineplot_filename, barplot_filename):
    im1 = Image.open(lineplot_filename)
    im2 = Image.open(barplot_filename)

    dst = Image.new('RGB', (im1.width, im1.height))
    dst.paste(im1, (0, 0))
    im2 = im2.crop((0, 0, im2.width, int(im2.height * .9)))
    dst.paste(im2, (int(im1.width * .55), 0))
    dst.save(output_filename)


def main():
    color_palette = [
        "#77aadd",
        "#ee8866",
        "#eedd88",
        "#ffaabb",
        "#99ddff",
        "#44bb99",
        "#bbcc33",
        "#999999",
    ]

    draw_barchart_together(color_palette=color_palette)


if __name__ == "__main__":
    main()
