import sys
import pandas as pd
from pathlib import Path
import argparse
import sys
import os
import re
import collections.abc
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import seaborn as sns
import string
import random
import json
import numpy as np

"""
Utils for plot_utils
"""


def get_paths(prefix):
    data_paths = []
    for subdirname in os.listdir(prefix):
        #         import ipdb
        #         ipdb.set_trace()
        path = os.path.join(prefix, subdirname)
        if os.path.isdir(path) and len(os.listdir(path)) > 0:
            data_paths.append(path)
    return data_paths


def get_size(start_path='.'):
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(start_path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            # skip if it is symbolic link
            if not os.path.islink(fp):
                total_size += os.path.getsize(fp)
    return total_size


def randomString(stringLength=10):
    """Generate a random string of fixed length """
    letters = string.ascii_lowercase
    return ''.join(random.choice(letters) for i in range(stringLength))


def save(filepath, fig=None):
    '''Save the current image with no whitespace
    Example filepath: "myfig.png" or r"C:\myfig.pdf"
    '''
    import matplotlib.pyplot as plt
    if not fig:
        fig = plt.gcf()

    plt.subplots_adjust(0, 0, 1, 1, 0, 0)
    for ax in fig.axes:
        ax.axis('off')
        ax.margins(0, 0)
        ax.xaxis.set_major_locator(plt.NullLocator())
        ax.yaxis.set_major_locator(plt.NullLocator())
    fig.savefig(filepath, pad_inches=0, bbox_inches='tight')


def plot_combineddf(df, suffix="K", xaxis_trunc=0):
    sns.set()
    # plt.style.use('seaborn-dark')

    if suffix == "K":
        df['Number of env steps total'] = df['Number of env steps total'] / 1000
    elif suffix == "M":
        df['Number of env steps total'] = df['Number of env steps total']/int(1e6)
    else:
        raise NotImplementedError
    tasks = [("Mean Blocks Stacked", 'Number of env steps total', "Mean Blocks Stacked")]
    # ("Test Percent Solved", 'Number of env steps total', "Fraction Solved")]
    fig, axs = plt.subplots(len(tasks), 1, figsize=(16, 10), sharex=True, squeeze=False)

    lines = []

    plt.rcParams["legend.fontsize"] = 17

    for idx, item in enumerate(tasks):
        y, xlabel, ylabel = item
        # print("num unique")
        # print(f"{df['Experiment'].nunique()}")
        # hue_ordering = ["Mlp - Direct", "Mlp - Sequential", "ReNN - Direct",
        #                 "ReNN - Uniform", "ReNN - Sequential"]
        hue_ordering = ["MLP - Curriculum"]
        # hue_ordering=["seed0", "seed1", "seed2", "mlp_direct", "mlp_sequential", "relational_direct", "relational_uniform"]

        # palette = sns.color_palette('Set1', n_colors=len(hue_ordering))

        # import pdb
        # pdb.set_trace()
        palette = sns.color_palette('Set1', n_colors=5)

        palette = palette[3:] + palette[:3]

        palette = palette[:len(hue_ordering)]

        dash_styles = ["",
                       (4, 1.5),
                       (1, 1),
                       (3, 1, 1.5, 1),
                       (5, 1, 1, 1),
                       (5, 1, 2, 1, 2, 1),
                       (2, 2, 3, 1.5),
                       (1, 2.5, 3, 1.2)]

        ax = sns.lineplot(x='Number of env steps total', y=y, hue="Experiment", style="Experiment", palette=palette,
                          data=df, ax=axs[idx, 0], hue_order=hue_ordering, dashes=False, ci="sd",
                          estimator='mean')
        lines.append(ax.lines)

        plt.xlabel('Environment Steps', fontsize=20, labelpad=20)
        # ax.set_xlabel('')
        ax.set_ylabel(ylabel, fontsize=20, labelpad=20)
        # if "Average return" in item:
        #     ax.set_ylim([-800, 0])
        ax.set_ylim([0, 6])
        ax.set_xlim([-1.9078500000000003, 40.10335])
        if "Fraction solved" in item:
            ax.set_ylim([0, 1])
        ax.xaxis.set_major_formatter(FuncFormatter(lambda x, pos: str(f'%.{xaxis_trunc}f' % (x)) + F"{suffix} "))

        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.85, box.height])  # resize position

        ax.tick_params(axis='both', labelsize=20)
        ax.legend(prop=dict(size=17), loc="upper left")
        # Put a legend to the right side
        # ax.legend(loc="upper left", bbox_to_anchor=(-1, -1))
        #
        #
        # ax.legend(loc="upper left", bbox_to_anchor=(1, 1))
        # ax.legend(loc='center right', bbox_to_anchor=(1.75, 0.5), ncol=1)

    # gca legend messes up the above legend command
    # plt.gca().legend(loc="upper left")
    # plt.gca().legend().set_title('')
    # plt.rcParams["font.size"] = 20
    # plt.rcParams["axes.titlesize"] = 20


    plt.subplots_adjust(left=None, bottom=None, right=.7, top=None, wspace=None, hspace=None)
    ax = fig.add_subplot(111, frameon=False)
    ax.grid(False)
    # ax.legend(loc='center left', bbox_to_anchor=(1.75, 0.5), ncol=1)
    # hide tick and tick label of the big axis
    plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)

    fn = f"./plots/combined.png"
    print(f"FN {fn}")
    ax.get_figure().savefig(fn)
    return lines


def get_df(folder, num_blocks, filter_epoch, num_workers=35):
    data_list = []
    if filter_epoch is None:
        filter_epoch = float("inf")
    assert filter_epoch > 0  # filter_epoch == 0 makes the df empty
    files = list(Path(folder).glob('**/progress.csv'))
    assert files
    df = pd.concat((pd.read_csv(f) for f in files), sort=False)
    assert not df.empty

    if num_blocks is None:
        df['Mean Blocks Stacked'] = df['Test Num Blocks Stacked']
    else:
        df['Mean Blocks Stacked'] = num_blocks + df['Test Rewards Mean'].clip(lower=-num_blocks, upper=0)
    df = df[df['Epoch'] <= filter_epoch]
    df['Number of env steps total'] = df['Number of env steps total'] * num_workers
    data_list.append(df)
    print(F"{folder} {filter_epoch}")
    return pd.concat(data_list, ignore_index=False, sort=False)


def get_dfs(baseline_paths):
    dfs = []
    df_names = []

    for baseline in baseline_paths:
        print("\n")
        with open(baseline + "/settings.json") as f:
            js = json.load(f)
        if "seeds" in js and js["seeds"]:
            tmp_dfs, tmp_d = get_dfs(get_paths(baseline))
            max_size_df = sorted(tmp_dfs, key=lambda x: max(x['Number of env steps total']), reverse=True)[0]

            max_step_seed = max_size_df['Experiment'].iloc[0]

            max_step_seed_dfs = filter(lambda df: df['Experiment'].iloc[0] == max_step_seed, tmp_dfs)
            print("Max step seed dfs")
            max_size_xs = pd.concat(sorted(max_step_seed_dfs, key=lambda df: df['Number of env steps total'].iloc[0], reverse=False), ignore_index=False, sort=False)['Number of env steps total']

            combined_df = pd.concat(tmp_dfs, ignore_index=False, sort=False)
            for seed in combined_df['Experiment'].unique():
                seed_dfs = combined_df[combined_df['Experiment']==seed]
                # seed_df_across_curriculum = pd.concat(sorted(seed_dfs, key=lambda df: df['Number of env steps total'].iloc[0], reverse=False), ignore_index=False, sort=False)
                seed_df_across_curriculum = seed_dfs.sort_values(by=['Number of env steps total'])

                seed_xs = max_size_xs.to_numpy()[max_size_xs.to_numpy() <= max(seed_df_across_curriculum['Number of env steps total'])]
                dict = {"Mean Blocks Stacked": np.interp(seed_xs, seed_df_across_curriculum['Number of env steps total'].to_numpy(), seed_df_across_curriculum["Mean Blocks Stacked"].to_numpy()),
                        'Number of env steps total': seed_xs,
                        'Experiment': "GNN - Curriculum"}
                dfs.append(pd.DataFrame(dict))


            # new_dfs = []
            # for df in tmp_dfs:
            #     x, xp, fp = max_size_xs, df['Number of env steps total'], df['Mean Blocks Stacked']
            #     dict = {"Mean Blocks Stacked": np.interp(x.to_numpy(), xp.to_numpy(), fp.to_numpy()),
            #             'Number of env steps total': x.to_numpy(),
            #             'Experiment': "relational_sequential (Ours)"}
            #     new_dfs.append(pd.DataFrame(dict))
            # dfs += new_dfs
            continue

        num_blocks = js['num_block_list']
        filter_epoch_list = js['filter_epoch_list']

        curriculum_stages = list(
            filter(lambda x: x[len(baseline) + 1:].count(os.sep) == 0, [x[0] for x in os.walk(baseline)]))
        curriculum_stages = curriculum_stages[1:]
        curriculum_stages = sorted(curriculum_stages, key=lambda x: os.path.basename(x)[0])
        # print(exps)
        for stage_idx, stage in enumerate(curriculum_stages):
            # if filter_epoch_list[stage_idx] == 0:
            #     continue
            df = get_df(stage, num_blocks[stage_idx], filter_epoch=filter_epoch_list[stage_idx])
            if stage_idx == 0:
                # prev_filter_timestep = max(df['Timesteps'])
                prev_filter_timestep = max(df['Number of env steps total'])
            else:
                # df['Timesteps'] = df['Timesteps'] + prev_filter_timestep
                df['Number of env steps total'] = df['Number of env steps total'] + prev_filter_timestep
                # prev_filter_timestep = max(df['Timesteps'])
                prev_filter_timestep = max(df['Number of env steps total'])
                print(f"{baseline} stage_idx: {stage_idx} prev timestep: {prev_filter_timestep // (1e6)}")
            dfs.append(df)
            df_names.append(os.path.basename(stage))
            print(F"Stage: {stage}")

            df['Experiment'] = os.path.basename(baseline)
    return dfs, df_names


def main():
    global args
    parser = argparse.ArgumentParser()
    parser.add_argument("--prefix", type=str)
    parser.add_argument("--example_dir", type=str)
    parser.add_argument("--suffix", type=str, default="M")
    parser.add_argument('prefix_pos')

    args = parser.parse_args(sys.argv[1:])

    args.data_paths = []
    prefix = args.prefix if args.prefix else args.prefix_pos

    for subdirname in os.listdir(prefix):
        path = os.path.join(prefix, subdirname)
        if os.path.isdir(path) and len(os.listdir(path)) > 0:
            args.data_paths.append(path)

    dfs, dfs_names = get_dfs(args.data_paths)
    # for plot_idx, df in enumerate(dfs):
    combined_df = pd.concat(dfs, ignore_index=False, sort=False)
    plot_combineddf(combined_df, suffix=args.suffix)


if __name__ == "__main__":
    main()