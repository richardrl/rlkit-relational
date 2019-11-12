## ReNN

#### Towards Practical Robotic Manipulation using Relational Reinforcement Learning

To replicate the main result of the paper, you should run the following with **35** workers. You should have at least the same number of available threads as workers. Having more threads can speed up wall-clock training time by 30%+.

1. Train on Pick and Place task (equivalent to Single Tower task with `stackonly=False`) with 1 block:

    1. Run with 35 workers
    
    `mpirun -np 35 python examples/relationalrl/train_pickandplace1.py`

Your results should appear in `data` folder under `rlkit-relational` project root.

2. Transfer to Pick and Place task with 2 blocks:

    1. Select a .PKL file from the folder in data at the desired checkpoint. Out of 35 worker folders, only one is checkpointing weights. You can find it by selecting the worker folder with the biggest size on disk. You should select a checkpoint where the policy is beginning to converge.
    
    2. Assign filename on line 92 of `examples/relationalrl/train_sequential_transfer.py` to be the .PKL selected in step 1.
    
    3. Run with 35 workers/threads.
`mpirun -np 35 python examples/relationarl/train_sequential_transfer.py`

3. Transfer to Single Tower task (`stackonly=True`) for 2+ blocks by repeating Step 2 with the most recently trained .PKL and changing `num_blocks` to be the desired number of blocks when prompted.