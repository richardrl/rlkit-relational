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
1. Install a virtualenv with the required packages.
```
virtualenv -p python3 relationalrl_venv
```

Activate the virtualenv.
```
source relationalrl_venv/bin/activate
```
2. Install numpy.
```
pip install numpy
```
3. Prepare for [mujoco-py](https://github.com/openai/mujoco-py) installation.
    1. Download [mjpro150](https://www.roboti.us/index.html)
    2. `cd ~`
    3. `mkdir .mujoco`
    4. Move mjpro150 folder to `.mujoco`
    5. Move mujoco license key `mjkey.txt` to `~/.mujoco/mjkey.txt`
    6. Set LD_LIBRARY_PATH:
    
    `export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/.mujoco/mjpro150/bin`

    7. For Ubuntu, run:
    
    `sudo apt install libosmesa6-dev libgl1-mesa-glx libglfw3`
    
    `sudo apt install -y patchelf`

4. Install supporting packages
```
pip install -r requirements.txt
```

Make sure pip is with python3!!

5. Install Fetch Block Construction environment: 
```
git clone https://github.com/richardrl/fetch-block-construction
```

```
cd fetch-block-construction
```

```
pip install -e .
```

6. Copy `config_template.py` to `config.py` and fill out `config.py` with desired config settings:
```
cp rlkit/launchers/config_template.py rlkit/launchers/config.py
```

7. Set PYTHONPATH:

    `export PYTHON_PATH=$PYTHON_PATH:<path/to/rlkit-relational>`
    
8. Add the export statements above to `.bashrc` to avoid needing to run them everytime you login. 


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
You can use a GPU by setting
`mode="gpu_opt"` in the example scripts.

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

To visualize results, download [viskit](https://github.com/vitchyr/viskit). You can visualize results with:
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
Much of the coding infrastructure and base algorithm implementations are courtesy of [RLKit](https://github.com/vitchyr/rlkit).

The Dockerfile is based on the [OpenAI mujoco-py Dockerfile](https://github.com/openai/mujoco-py/blob/master/Dockerfile).