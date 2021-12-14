# Beyond Prioritized Experience Replay

Ablation study of the [Prioritized Experience Replay](https://arxiv.org/pdf/1511.05952.pdf) presentend by Schau et al. 



## Prerequisite
- Pytorch
- Numpy
- random
- Gym
- Matplotlib
- pyvirtualdisplay
- JSAnimation.IPython_display
- IPython
- python-opengl
- ffmpeg
- xvfb

## Run
To replicate the results, we suggest running the code on COLAB in the same way as it is run in Tests.ipynb


The code in this repo is based on the following repo:

- [Rainbow is all you need](https://github.com/Curt-Park/rainbow-is-all-you-need.git)

Here are some of the results we got:

- Performance for priority-based PER in different environments: 
![alt text](https://github.com/emiled16/Beyond_prioritized_experience_replay/blob/master/images/env.JPG)
- Performance for priority-based PER with different memory size(Cartpole v0)
![alt text](https://github.com/emiled16/Beyond_prioritized_experience_replay/blob/master/images/memory.JPG)
- Performance for priority-based PER with different stalness Coeff (Cartpole v0)
![alt text](https://github.com/emiled16/Beyond_prioritized_experience_replay/blob/master/images/priority_staleness.JPG)
- Performance for rank-based PER with different stalness Coeff (Cartpole v0)
![alt text](https://github.com/emiled16/Beyond_prioritized_experience_replay/blob/master/images/rank_staleness.JPG)
- Performance for priority-based PER with different Positive penalty coefficients (Cartpole
v0)
![alt text](https://github.com/emiled16/Beyond_prioritized_experience_replay/blob/master/images/priority_positive.JPG)
- Performance for rank-based PER with different positive penalty coeff (Cartpole v0)
![alt text](https://github.com/emiled16/Beyond_prioritized_experience_replay/blob/master/images/rank_positive.JPG)
- Performance for priority-based PER without and with differential method (Cartpole v0)
![alt text](https://github.com/emiled16/Beyond_prioritized_experience_replay/blob/master/images/priority_diff.JPG)
- Performance for rank-based PER without and with differential method (Cartpole v0)
![alt text](https://github.com/emiled16/Beyond_prioritized_experience_replay/blob/master/images/rank_diff.JPG)
- Performance for the 3 different types of PER (priority-based, rank-based and hybrid approach comibining positive penalty and staleness)(Cartpole v0)
![alt text](https://github.com/emiled16/Beyond_prioritized_experience_replay/blob/master/images/all.JPG)
