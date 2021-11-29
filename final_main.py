
import sys
import os
import random
from typing import Dict, List, Tuple

import gym
import matplotlib.pyplot as plt
import numpy as np
import torch
from final_agent import DQNAgent

def run_experiment(env_id="CartPole-v0", seed=777, num_episodes=2000, memory_size=10000,
                  batch_size=32, target_update=100, epsilon_decay=1/2000,
                  alpha: float = 0.2, beta: float = 0.6, prior_eps: float = 1e-6,
                  staleness=0.0001, positive_reward=0.0001, differential=False,
                  priority_based='rank', episodic=False):

    IN_COLAB = "google.colab" in sys.modules

    if IN_COLAB:
        from pyvirtualdisplay import Display
        # Start virtual display
        dis = Display(visible=0, size=(400, 400))
        dis.start()

    # environment
    env = gym.make(env_id)
    if IN_COLAB:
        env = gym.wrappers.Monitor(env, "videos", force=True)

    def seed_torch(seed):
        torch.manual_seed(seed)
        if torch.backends.cudnn.enabled:
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True

    np.random.seed(seed)
    random.seed(seed)
    seed_torch(seed)
    env.seed(seed)

    # train
    agent = DQNAgent(env=env, memory_size=memory_size, batch_size=batch_size,
                    target_update=target_update, epsilon_decay=epsilon_decay,
                    alpha=alpha, beta=beta, prior_eps=prior_eps, staleness=staleness,
                    positive_reward=positive_reward, differential=differential,
                    priority_based=priority_based, episodic=episodic)

    agent.train(num_episodes)

    frames = agent.test()

    if IN_COLAB:  # for colab
        import base64
        import glob
        import io

        from IPython.display import HTML, display

        def ipython_show_video(path: str) -> None:
            """Show a video at `path` within IPython Notebook."""
            if not os.path.isfile(path):
                raise NameError("Cannot access: {}".format(path))

            video = io.open(path, "r+b").read()
            encoded = base64.b64encode(video)

            display(HTML(
                data="""
                <video alt="test" controls>
                <source src="data:video/mp4;base64,{0}" type="video/mp4"/>
                </video>
                """.format(encoded.decode("ascii"))
            ))

        list_of_files = glob.glob("videos/*.mp4")
        latest_file = max(list_of_files, key=os.path.getctime)
        print(latest_file)
        ipython_show_video(latest_file)

    else:  # for jupyter
        from matplotlib import animation
        from JSAnimation.IPython_display import display_animation
        from IPython.display import display

        def display_frames_as_gif(frames: List[np.ndarray]) -> None:
            """Displays a list of frames as a gif, with controls."""
            patch = plt.imshow(frames[0])
            plt.axis('off')

            def animate(i):
                patch.set_data(frames[i])

            anim = animation.FuncAnimation(
                plt.gcf(), animate, frames=len(frames), interval=50)
            display(display_animation(anim, default_mode='loop'))

        # display
        display_frames_as_gif(frames)

