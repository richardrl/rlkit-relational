# rlkit-relational
Framework for relational reinforcement learning implemented in PyTorch. 

We include additional features, beyond RLKit, aimed at supporting high step complexity tasks and relational RL:
- Training with multiple parallel workers via MPI
- Concise implementations of inductive graph neural net architectures
- Padded and masked minibatching for simultaneous training over variable-sized observation graphs
- Observation graph input module for block construction task
- Tuned example scripts for block construction task

Implemented algorithms:
 - ReNN (*Towards Practical Robotic Manipulation using Relational Reinforcement Learning*)
    - [example script](examples/relationalrl/train_pickandplace1.py)
    - [ReNN paper](#)
    - [Documentation](docs/ReNN.md)

To get started, checkout the example scripts, linked above.

## Installation
1. Copy `config_template.py` to `config.py` and fill out with desired config settings:
```
cp rlkit/launchers/config_template.py rlkit/launchers/config.py
```
2. Install a virtualenv with the required packages.
```
virtualenv -p python3 requirements.txt relationalrl_venv

```

Activate the virtualenv.
```
source relationalrl_venv/bin/activate

```

3. Install Fetch Block Construction environment by cloning the repo: 
```
https://github.com/richardrl/fetch-block-construction
```

and running inside the repo root
```
pip install -e .
```

Make sure Mujoco 1.5 is installed at ~/.mujoco

## Running scripts
Make sure to set `mode` in the scripts:
- `here_no_doodad`: run locally, without Docker
- `local_docker`: locally with Docker
- `ec2`: Amazon EC2

To run multiple workers under the `here_no_doodad` setting, run the following command in the command line:
```
mpirun -np <numworkers> python examples/relationalrl/train_pickandplace1.py
```

## Using a GPU
You can use a GPU by calling
```
import rlkit.torch.pytorch_util as ptu
ptu.set_gpu_mode(True)
```
before launching the scripts.

If you are using `doodad` (see below), simply use the `use_gpu` flag:
```
run_experiment(..., use_gpu=True)
```

## Visualizing a policy and seeing results
During training, the results will be saved to a file called under
```
LOCAL_LOG_DIR/<exp_prefix>/<foldername>
```
 - `LOCAL_LOG_DIR` is the directory set by `rlkit.launchers.config.LOCAL_LOG_DIR`. Default name is 'output'.
 - `<exp_prefix>` is given either to `setup_logger`.
 - `<foldername>` is auto-generated and based off of `exp_prefix`.
 - inside this folder, you should see a file called `params.pkl`. To visualize a policy, run

```
(rlkit) $ python scripts/sim_policy.py LOCAL_LOG_DIR/<exp_prefix>/<foldername>/params.pkl
```

If you have rllab installed, you can also visualize the results
using `rllab`'s viskit, described at
the bottom of [this page](http://rllab.readthedocs.io/en/latest/user/cluster.html)

tl;dr run

```bash
python rllab/viskit/frontend.py LOCAL_LOG_DIR/<exp_prefix>/
```
to visualize all experiments with a prefix of `exp_prefix`. To only visualize a single run, you can do
```bash
python rllab/viskit/frontend.py LOCAL_LOG_DIR/<exp_prefix>/<folder name>
```

Alternatively, if you don't want to clone all of `rllab`, a repository containing only viskit can be found [here](https://github.com/vitchyr/viskit). You can similarly visualize results with.
```bash
python viskit/viskit/frontend.py LOCAL_LOG_DIR/<exp_prefix>/
```
This `viskit` repo also has a few extra nice features, like plotting multiple Y-axis values at once, figure-splitting on multiple keys, and being able to filter hyperparametrs out.

## Launching jobs with `doodad`
The `run_experiment` function makes it easy to run Python code on Amazon Web
Services (AWS) or Google Cloud Platform (GCP) by using
[doodad](https://github.com/justinjfu/doodad/).

It's as easy as:
```
from rlkit.launchers.launcher_util import run_experiment

def function_to_run(variant):
    learning_rate = variant['learning_rate']
    ...

run_experiment(
    function_to_run,
    exp_prefix="my-experiment-name",
    mode='ec2',  # or 'gcp'
    variant={'learning_rate': 1e-3},
)
```
You will need to set up parameters in config.py (see step one of Installation).
This requires some knowledge of AWS and/or GCP, which is beyond the scope of
this README.
To learn more, more about `doodad`, [go to the repository](https://github.com/justinjfu/doodad/).

## Credits
Much of the infrastructure and base algorithm implementations are courtesy of [RLKit](https://github.com/vitchyr/rlkit).

The Dockerfile is based on the [OpenAI mujoco-py Dockerfile](https://github.com/openai/mujoco-py/blob/master/Dockerfile).

## TODOs