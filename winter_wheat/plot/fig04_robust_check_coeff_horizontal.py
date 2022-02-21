import csv
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


def draw_bar_plot_only_avg(base_dir, output_filename, models, variables, agmip_model, filename_pattern, five_state=False, sharedy=False, output_csv_file=None, color_palette=None):
    update_parameter(agmip_model)
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
    fig, axes = plt.subplots(1, 2, figsize=(8, 4), gridspec_kw={"width_ratios": [3.6, 2.6], "wspace": 0}, sharey=True, dpi=300, constrained_layout=False)
    # sns.despine()
    plt.subplots_adjust(left=0.11, bottom=0.10, right=0.92, top=0.95, wspace=0.14, hspace=0.18)

    if color_palette is None:
        color_palette = [
            "#66A4C6",  # blue  hsl (142, 116, 164) in pickpick
            "#FC9630",  # orange  hsl (21, 247, 167) in pickpick
        ]

    sns.set_palette(sns.color_palette(color_palette))

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
                        df_combine = df_combine.dropna()

                        if var_name == "fdd1_sc2_sctf":
                            hist_mean = np.mean(remove_na(df_combine["fdd1:sc2_sctf_x"].to_numpy()))
                            future_mean = np.mean(remove_na(df_combine["fdd1:sc2_sctf_y"].to_numpy()))
                        else:
                            hist_mean = np.mean(remove_na(df_combine[coeff[each_var_name].name + "_x"].to_numpy()))
                            future_mean = np.mean(remove_na(df_combine[coeff[each_var_name].name + "_y"].to_numpy()))

                        delta_mean = future_mean - hist_mean
                        effect = delta_mean * coeff[each_var_name].value

                        variance = (delta_mean * coeff[each_var_name].SE)**2
                        key = (model_print_name, scenario, var_name)
                        if key not in bar_chart_data:
                            bar_chart_data[key] = [effect, variance]
                        else:
                            bar_chart_data[key] = [bar_chart_data[key][0] + effect, bar_chart_data[key][1] + variance]

    draw_subplot_for_avg(bar_chart_data, raw_data, ax=axes[0], selected_variables=["fdd1", "fdd1_sc2_sctf", "gdd_all"])
    draw_subplot_for_avg(bar_chart_data, raw_data, ax=axes[1], selected_variables=["prcp_all", "snowfall_all"])

    for i in range(2):
        axes[i].xaxis.set_tick_params(direction="out")
        axes[i].xaxis.set_ticks_position('bottom')

        if i == 0:
            axes[i].yaxis.set_tick_params(labelleft=True)
            axes[i].yaxis.set_tick_params(labelright=False)
            axes[i].yaxis.set_label_position("left")
            axes[i].yaxis.set_ticks_position('left')
            axes[i].set(ylabel="Coefficient")
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



    lines_labels = [ax.get_legend_handles_labels() for ax in fig.axes]
    lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
    fig.align_ylabels(axes[:])
    fig.legend(lines[:2], labels[:2], loc='upper right', ncol=1, bbox_to_anchor=(0.9, 0.90))

    plt.savefig(output_filename)
    plt.close()

    if output_csv_file is not None:
        df = pd.DataFrame(raw_data, columns=["scenario", "GCMs", "period", "variable_name", "effect_mean", "SE", "error_bar_endpoint"])
        df.to_csv(output_csv_file, index=False)


def print_avg_models(models):
    coeff = []
    variance = 0

    for key in models:
        coeff.append(models[key].value)
        variance += models[key].SE ** 2

    print(coeff)
    print(np.mean(coeff))
    print(np.sqrt(variance))


def draw_bar_plot_two_pane(base_dir, output_filename, variables, filename_pattern, five_state=False, sharedy=False, output_csv_file=None, color_palette=None):
    agmip_models = [
        "M01",
        "M02",
        "M03",
        "M04",
        "M05",
        "M06",
        "M07",
        "M08",
        "M09",
        "M10",
    ]

    model_agmip_name = {
        "M01": "Test 1",
        "M02": "Test 2",
        "M03": "Test 3",
        "M04": "Test 4",
        "M05": "Test 5",
        "M06": "Test 6",
        "M07": "Test 7",
        "M08": "Test 8",
        "M09": "Test 9",
        "M10": "Test 10",
    }

    model_coeff = {
        "M01": Coefficient("fdd1", -0.0486, (-0.0425 - (-0.0486)) / 1.96),
        "M02": Coefficient("fdd1", -0.0424, (-0.0368 - (-0.0424)) / 1.96),
        "M03": Coefficient("fdd1", -0.0409, (-0.0357 - (-0.0409)) / 1.96),
        "M04": Coefficient("fdd1", -0.0424, (-0.0374 - (-0.0424)) / 1.96),
        "M05": Coefficient("fdd1", -0.0434, (-0.0434 - (-0.0434)) / 1.96),
        "M06": Coefficient("fdd1", -0.0680, (-0.0631 - (-0.0680)) / 1.96),
        "M07": Coefficient("fdd1", -0.0680, (-0.0615 - (-0.0680)) / 1.96),
        "M08": Coefficient("fdd1", -0.0561, (-0.0498 - (-0.0561)) / 1.96),
        "M09": Coefficient("fdd1", -0.0605, (-0.0515 - (-0.0605)) / 1.96),
        "M10": Coefficient("fdd1", -0.0623, (-0.0545 - (-0.0623)) / 1.96),
    }
    model_coeff_scf = {
        "M01": Coefficient("fdd1_sc2_sctf", 0.0550, (0.0606 - 0.0550) / 1.96),
        "M02": Coefficient("fdd1_sc2_sctf", 0.0490, (0.0542 - 0.0490) / 1.96),
        "M03": Coefficient("fdd1_sc2_sctf", 0.0404, (0.0459 - 0.0404) / 1.96),
        "M04": Coefficient("fdd1_sc2_sctf", 0.0422, (0.0486 - 0.0422) / 1.96),
        "M05": Coefficient("fdd1_sc2_sctf", 0.0514, (0.0514 - 0.0514) / 1.96),
        "M06": Coefficient("fdd1_sc2_sctf", 0.0635, (0.0691 - 0.0635) / 1.96),
        "M07": Coefficient("fdd1_sc2_sctf", 0.0630, (0.0690 - 0.0630) / 1.96),
        "M08": Coefficient("fdd1_sc2_sctf", 0.0442, (0.0498 - 0.0442) / 1.96),
        "M09": Coefficient("fdd1_sc2_sctf", 0.0606, (0.0690 - 0.0606) / 1.96),
        "M10": Coefficient("fdd1_sc2_sctf", 0.0678, (0.0751 - 0.0678) / 1.96),
    }

    print_avg_models(model_coeff)
    print_avg_models(model_coeff_scf)


    acre_file = get_project_root() / "output/county_map/v20200814-future-trend/prod/winter_wheat_acre_avg.csv"
    df_acre = pd.read_csv(acre_file)

    sns.set(style="white", palette="muted", color_codes=True)
    params = {
        'legend.fontsize': 16,
        'axes.labelsize': 23,
        'axes.titlesize': 18,
        'xtick.labelsize': 20,
        'ytick.labelsize': 20,
        'axes.linewidth': 1.5,
    }

    plt.rcParams.update(params)

    # Set up the matplotlib figure
    if len(variables) == 4:
        # fig, axes_all = plt.subplots(2, 2, figsize=(3, 5), gridspec_kw={"width_ratios": [2.6, 2.6], "wspace": 0},
        #                              sharey=True, dpi=300, constrained_layout=False)
        fig, axes_all = plt.subplots(2, 2, figsize=(5, 6), gridspec_kw={"width_ratios": [2.6, 2.6], "wspace": 0},
                                     sharey="row", dpi=300, constrained_layout=False)
    else:
        fig, axes_all = plt.subplots(1, 2, figsize=(12, 6), gridspec_kw={"width_ratios": [2.6, 2.6], "wspace": 0},
                                     sharey=True, dpi=300, constrained_layout=False)
    # sns.despine()
    plt.subplots_adjust(left=0.10, bottom=0.07, right=0.8, top=0.98)

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
    for i, var_name in enumerate(variables):
        for each_var_name in variables[var_name]:
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
            variance = (coeff[each_var_name].SE)**2

            key = (model_print_name, scenario, var_name)
            if key not in bar_chart_data:
                bar_chart_data[key] = [effect, variance]
            else:
                bar_chart_data[key] = [bar_chart_data[key][0] + effect, bar_chart_data[key][1] + variance]

    for agmip_name in agmip_models:
        # update_parameter(agmip_name)
        for i, var_name in enumerate(variables):
            for each_var_name in variables[var_name]:

                output_all_dir = os.path.join(base_dir, model_name)
                output_all_dir_each = os.path.join(output_all_dir, scenario)
                output_all_dir_each = os.path.join(output_all_dir_each, period_name)
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

                effect = model_coeff[agmip_name].value
                variance = (model_coeff[agmip_name].SE) ** 2

                if var_name == "fdd1_sc2_sctf":
                    effect = model_coeff_scf[agmip_name].value
                    variance = (model_coeff_scf[agmip_name].SE) ** 2

                key = (model_print_name, scenario, var_name)
                if key not in bar_chart_data:
                    bar_chart_data[key] = [effect, variance]
                else:
                    bar_chart_data[key] = [bar_chart_data[key][0] + effect, bar_chart_data[key][1] + variance]

    if len(variables) == 2:
        draw_subplot_for_agmip(bar_chart_data, raw_data, ax=axes_all[0],
                               selected_variables=["fdd1", ], obs_color=color_palette[-1])
        draw_subplot_for_agmip(bar_chart_data, raw_data, ax=axes_all[1], selected_variables=["fdd1_sc2_sctf", ],
                               obs_color=color_palette[-1])

    else:
        draw_subplot_for_agmip(bar_chart_data, raw_data, ax=axes_all[0, 0],
                               selected_variables=["fdd1", ], obs_color=color_palette[-1])
        draw_subplot_for_agmip(bar_chart_data, raw_data, ax=axes_all[0, 1],
                               selected_variables=["fdd1_sc2_sctf", ], obs_color=color_palette[-1])
        draw_subplot_for_agmip(bar_chart_data, raw_data, ax=axes_all[1, 0],
                               selected_variables=["prcp_all", ], obs_color=color_palette[-1])
        draw_subplot_for_agmip(bar_chart_data, raw_data, ax=axes_all[1, 1],
                               selected_variables=["snowfall_all", ], obs_color=color_palette[-1])

    for i in range(2):
        axes_all[i].set_yticks([-0.1, -0.05, 0, 0.05, 0.10])

    axes = axes_all
    for i in range(2):
        axes[i].xaxis.set_tick_params(direction="out")
        axes[i].xaxis.set_ticks_position('bottom')

        if i == 0:
            axes[i].yaxis.set_tick_params(labelleft=True)
            axes[i].yaxis.set_tick_params(labelright=False)
            axes[i].yaxis.set_label_position("left")
            axes[i].yaxis.set_ticks_position('left')
            axes[i].set(ylabel="Coefficient",)
            axes[i].set(xlim=(-0.5, 0.5))
        elif i == 1:
            axes[i].yaxis.set_tick_params(labelleft=False)
            axes[i].yaxis.set_tick_params(labelright=False)
            # axes[i].yaxis.set_label_position("none")
            axes[i].yaxis.set_ticks_position('none')
            axes[i].set(ylabel="")
            axes[i].set(xlim=(-0.5, 0.5))

        axes[i].yaxis.set_tick_params(direction="out")
        axes[i].legend_.remove()
        axes[i].set(xlabel="")

        # axes[i].set_xticklabels([])

        axes[i].axhline(0, c="black", lw=1.5)
    if i == 0:
        right_side = axes[i].spines["right"]
        right_side.set_visible(False)
    else:
        left_side = axes[i].spines["left"]
        left_side.set_visible(False)

    lines_labels = [ax.get_legend_handles_labels() for ax in fig.axes]
    lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
    fig.align_ylabels(axes[:])
    fig.legend(lines[:len(agmip_models)], labels[:len(agmip_models)], loc='upper right', ncol=1, bbox_to_anchor=(0.98, 0.97), borderpad=0.3,
               fancybox=False, shadow=False, frameon=False, columnspacing=1, handlelength=1.5)

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
    print(df.head())

    plotter = _BarPlotter('variable', "effect", 'scenario', df, None, None,
                np.mean, None, 1000, None, None,
                None, None, None, 1.0,
                ".26", None, None, True)
    plotter.plot(ax, {})
    confint = [[] for _ in plotter.plot_data]
    error_bar_cap = [[] for _ in plotter.plot_data]

    for ii, group_data in enumerate(plotter.plot_data):
        print("group_data: ", group_data)
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
    # print(plotter.capsize)
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
    print(df.head())

    plotter = _BarPlotter('variable', "effect", 'AgMIP', df, None, None,
                np.mean, None, 1000, None, None,
                None, None, None, 1.0,
                ".26", None, None, True)
    plotter.plot(ax, {})
    confint = [[] for _ in plotter.plot_data]
    error_bar_cap = [[] for _ in plotter.plot_data]

    for ii, group_data in enumerate(plotter.plot_data):
        print("group_data: ", group_data)
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
    plotter.capsize = 0.05
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


    # Draw vertical base line (this study)
    # print(bar_chart_this_study)
    x = [-0.5, 0.5]

    mean_models = 0
    var_models = 0

    if "fdd1" in selected_variables:
        mean_models = -0.053259999999999995
        var_models = 0.009792729072661288

    if "fdd1_sc2_sctf" in selected_variables:
        mean_models = 0.05371
        var_models = 0.009574724205262024

    y_est = mean_models
    y_err = var_models * 1.96
    if obs_color is None:
        obs_color = "#7C8DB0"
    #
    ax.plot(x, [y_est, y_est], '--', color=obs_color)
    ax.fill_between(x, y_est - y_err, y_est + y_err, alpha=0.2, color=obs_color, zorder=2)


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
        # print(ci_low, ci_high)
        # print(is_cap_low, is_cap_high)
        if ci_low == ci_high:
            continue
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


def draw_barchart_each_agmip_model(color_palette=None):
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
    output_dir = "N:/WinterWheat/output/manuscript_20210219_9GCMs_AgMIP"

    make_dirs(output_dir)

    for model_name in MODEL_AgMIP_NAMES:
        output_file = os.path.join(output_dir, "Figure04_barplot-stress-effect-avg-errorbar_{}.png".format(model_name))
        output_csv_file = os.path.join(output_dir, "Figure04_barplot-stress-effect-avg-errorbar_{}.csv".format(model_name))

        draw_bar_plot_only_avg(base_dir, output_file, models, variables, model_name, "gridMET_{}_avg.csv", output_csv_file=output_csv_file, color_palette=color_palette)


def draw_barchart_together(color_palette=None):
    variables = {
        "fdd1": ("fdd1",),
        "fdd1_sc2_sctf": ("fdd1_sc2_sctf",),
    }

    base_dir = "N:/WinterWheat/county_map/v20210111-for-Peng-rawdata"  # add more GCMs
    output_dir = "N:/WinterWheat/output/manuscript_20211214_robust_check"

    make_dirs(output_dir)

    output_file = os.path.join(output_dir, "Figure04d_barplot-coeff-robust-check-horizontal.png")
    output_csv_file = os.path.join(output_dir, "Figure04d_barplot-coeff-robust-check-horizontal.csv")

    draw_bar_plot_two_pane(base_dir, output_file, variables, "gridMET_{}_avg.csv",
                           output_csv_file=output_csv_file, color_palette=color_palette)


def main():
    color_palette = [
        "#E64B35",
        "#4DBBD5",
        "#00A087",
        "#F39B7F",
        "#B09C85",
        "#91D1C2",
        "#7E6148",
        "#3C5488",

        "#E62F31",
        "#CDB6D8",
        "#7C8DB0",
    ]

    draw_barchart_together(color_palette=color_palette)


if __name__ == "__main__":
    main()
