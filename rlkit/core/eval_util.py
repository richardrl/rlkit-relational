"""
Common evaluation utilities.
"""

from collections import OrderedDict
from numbers import Number

import numpy as np


def get_generic_path_information(paths, stat_prefix='', num_blocks=None):
    """
    Get an OrderedDict with a bunch of statistic names and values.
    """
    # assert num_blocks is not None
    # assert len(paths) == 21, len(paths)
    statistics = OrderedDict()
    returns = [sum(path["rewards"]) for path in paths]
    statistics.update(create_stats_ordered_dict('Returns', returns,
                                                stat_prefix=stat_prefix))

    rewards = np.vstack([path["rewards"] for path in paths])
    statistics.update(create_stats_ordered_dict('Rewards', rewards,
                                                stat_prefix=stat_prefix))

    assert np.all([path['mask'][0] == path['mask'][x] for path in paths for x in range(len(path))])

    final_num_blocks_stacked = [path['mask'][0].sum() - np.clip(np.abs(path["rewards"][-1]), None, path['mask'][0].sum()) for path in paths]
    statistics[F'{stat_prefix} Final Num Blocks Stacked'] = np.mean(final_num_blocks_stacked)

    mean_num_blocks_stacked = [(path['mask'][0].sum() - np.clip(np.abs(path['rewards']), None, path['mask'][0].sum())).mean() for path in paths]
    assert all(x >= 0 for x in mean_num_blocks_stacked), mean_num_blocks_stacked
    statistics[F'{stat_prefix} Mean Num Blocks Stacked'] = np.mean(mean_num_blocks_stacked)

    if isinstance(paths[0], dict) and num_blocks:
        # Keys are block IDs, values are final goal distances.
        # Each block ID is a list of final goal distances for all paths
        final_goal_dist = dict()
        seq = []
        for block_id in range(num_blocks):
            final_goal_dist[block_id] = [np.linalg.norm(path['observations'][-1]['achieved_goal'][block_id*3:(block_id+1)*3] - path['observations'][-1]['desired_goal'][block_id*3:(block_id+1)*3]) for path in paths]
            # statistics.update(create_stats_ordered_dict(F"Fin Goal Dist Blk {block_id}", final_goal_dist[block_id],
            #                                             stat_prefix=stat_prefix))
            seq.append(np.array([np.linalg.norm(path['observations'][-1]['achieved_goal'][block_id*3:(block_id+1)*3] - path['observations'][-1]['desired_goal'][block_id*3:(block_id+1)*3]) for path in paths]))

        block_dists = np.vstack(seq)
        assert len(block_dists.shape) == 2
        sorted = np.sort(block_dists, axis=0)
        # sorted = block_dists

        # for block_id in range(num_blocks):
        #     statistics.update(create_stats_ordered_dict(F"Fin Goal Dist Blk {block_id}", sorted[block_id], stat_prefix=stat_prefix))

        total_solved = 0
        goal_threshold = .05
        for path_fd_tuple_across_blocks in zip(*list(final_goal_dist.values())):
            total_solved +=all(fd_blocki < goal_threshold for fd_blocki in path_fd_tuple_across_blocks)

        assert len(paths) == len(final_goal_dist[0])
        percent_solved = total_solved/len(paths)
        assert 0 <= percent_solved <= 1, (total_solved, len(paths), final_goal_dist)
        statistics[F"{stat_prefix} Percent Solved"] = percent_solved

    actions = [path["actions"] for path in paths]
    if len(actions[0].shape) == 1:
        actions = np.hstack([path["actions"] for path in paths])
    else:
        actions = np.vstack([path["actions"] for path in paths])
    statistics.update(create_stats_ordered_dict(
        'Actions', actions, stat_prefix=stat_prefix
    ))
    statistics['Num Paths'] = len(paths)

    return statistics


def get_average_returns(paths):
    returns = [sum(path["rewards"]) for path in paths]
    return np.mean(returns)


def create_stats_ordered_dict(
        name,
        data,
        stat_prefix=None,
        always_show_all_stats=True,
        exclude_max_min=False,
):
    if stat_prefix is not None:
        name = "{} {}".format(stat_prefix, name)
    if isinstance(data, Number):
        return OrderedDict({name: data})

    if len(data) == 0:
        return OrderedDict()

    if isinstance(data, tuple):
        ordered_dict = OrderedDict()
        for number, d in enumerate(data):
            sub_dict = create_stats_ordered_dict(
                "{0}_{1}".format(name, number),
                d,
            )
            ordered_dict.update(sub_dict)
        return ordered_dict

    if isinstance(data, list):
        try:
            iter(data[0])
        except TypeError:
            pass
        else:
            data = np.concatenate(data)

    if (isinstance(data, np.ndarray) and data.size == 1
            and not always_show_all_stats):
        return OrderedDict({name: float(data)})

    stats = OrderedDict([
        (name + ' Mean', np.mean(data)),
        (name + ' Std', np.std(data)),
    ])
    if not exclude_max_min:
        stats[name + ' Max'] = np.max(data)
        stats[name + ' Min'] = np.min(data)
    return stats
