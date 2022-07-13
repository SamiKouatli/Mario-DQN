#!/usr/bin/env python
# -*- encoding: utf-8 -*-

# SPDX-License-Identifier: MIT

import datetime
import os
import shutil
from collections import deque

from tqdm import tqdm

from config.paths import ENVIRONMENT_CONFIG_PATH
from config.paths import NN_MODEL_SAVE_PATH
from config.paths import RESULTS_SAVE_PATH
from config.paths import RUNNER_CONFIG_PATH
from config.paths import SAVE_PATH
from mario_dqn.agent.agent import Agent
from mario_dqn.environment.environment import Environment
from mario_dqn.runner.runner_parser import RunnerParser
from mario_dqn.scorer import Scorer
from mario_dqn.utils import get_file_path_from_root


class Runner:
    """
    Main class of project, runs the whole AI work.
    """

    def __init__(self):
        """
        Initialize runner parameters.
        """
        # Get save path
        current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        save_path = f"{SAVE_PATH}/{current_time}/"
        self.save_path = get_file_path_from_root(save_path)
        self.last_results = deque(maxlen = 10)

    def train(self, pretrained_model_path, is_recording, is_verbose):
        """
        Train the AI model.

        Parameters
        ----------
        pretrained_model_path: str|None
            Path to pretrained model
        is_recording: bool
            Is record enabled.
        is_verbose: bool
            Is verbose enabled.
        """
        pretrained = pretrained_model_path is not None

        self._run(is_training = True,
                  pretrained = pretrained,
                  load_path = pretrained_model_path,
                  is_recording = is_recording,
                  is_verbose = is_verbose)

    def infer(self, pretrained_model_path, is_recording, is_verbose):
        """
        Infer the AI model.

        Parameters
        ----------
        pretrained_model_path: str
            Path to pretrained model
        is_recording: bool
            Is record enabled.
        is_verbose: bool
            Is verbose enabled.
        """
        pretrained = pretrained_model_path is not None

        if not pretrained:
            print(f"[Runner] Error when in inference, you need to fill '--pretrained' parameter")
            exit()

        self._run(is_training = False,
                  pretrained = pretrained,
                  load_path = pretrained_model_path,
                  is_recording = is_recording,
                  is_verbose = is_verbose)

    @staticmethod
    def _load_model(agent, load_path):
        """
        Load model.
        agent: Agent
            Agent to fill with saved weighs.
        load_path:  str
            Path to pretrained model.
        """
        print(f"Model is pretrained, model loaded from: {load_path}")

        try:
            agent.load(load_path)
        except (IOError, ImportError) as exc:
            print(f"[Runner] Error {exc} when opening {load_path}")
            exit()

    @staticmethod
    def _save_model_init(save_path):
        """
        Initialization of save model.
        save_path:  str
            Path to save model.
        """
        print(f"Project is in training mode, model will be saved to: {save_path}")

        # Create directory for saving model
        os.makedirs(save_path, exist_ok = True)

        # Copy parameters file to folder
        shutil.copy(get_file_path_from_root(ENVIRONMENT_CONFIG_PATH), save_path)
        shutil.copy(get_file_path_from_root(RUNNER_CONFIG_PATH), save_path)

    def _run(self, is_training, pretrained, is_recording, load_path, is_verbose):
        """
        Run the AI process.
        is_training: bool
            Is model training or infering.
        pretrained: bool
            Is model pretrained.
        is_recording: bool
            Is record enabled.
        load_path:  str|None
            Path to pretrained model.
        is_verbose: bool
            Is verbose enabled.
        """
        # Parse runner
        runner_model = RunnerParser(is_training = is_training,
                                    pretrained = pretrained,
                                    is_verbose = is_verbose).content

        # Create environment
        e = Environment(is_training = is_training,
                        is_recording = is_recording,
                        save_path = self.save_path,
                        is_verbose = is_verbose)
        env = e.env
        exploration_max, exploration_min, exploration_decay = runner_model.get_exploration_rate()
        scorer = None

        # Create agent
        agent = Agent(action_dim = env.action_space.n,
                      max_memory_size = runner_model.max_memory_size,
                      learning_rate = runner_model.learning_rate,
                      exploration_max = exploration_max,
                      exploration_min = exploration_min,
                      exploration_decay = exploration_decay,
                      gamma = runner_model.gamma,
                      batch_size = runner_model.batch_size,
                      image_size = e.model.downscale_size,
                      buffer_size = e.model.buffer_size,
                      is_verbose = is_verbose)

        # Load model if necessary
        if pretrained:
            self._load_model(agent = agent, load_path = load_path)

        # Create save elements if in training
        if is_training:
            self._save_model_init(save_path = self.save_path)
            scorer = Scorer.saver(buffer_size = runner_model.score_buffer_size,
                                  filepath = f"{self.save_path}/{RESULTS_SAVE_PATH}")

        env.reset()

        # Only one episode in inference
        episodes_nb = runner_model.num_episodes
        if not is_training:
            episodes_nb = 1

        # Run episodes loop
        for episode in tqdm(range(episodes_nb)):
            episode_reward = 0
            step_nb = 0
            state = env.reset()
            finished = False

            # Run one episode loop
            while not finished:
                step_nb += 1

                # Predict an action
                action = agent.act(state)
                # Make action
                next_state, reward, done, _ = env.step(action)
                # Punish behaviour which does not accumulate reward
                reward -= 1

                if is_training:
                    # Store sequence in replay memory
                    agent.remember(state = state,
                                   action = action,
                                   reward = reward,
                                   next_state = next_state,
                                   done = done)

                state = next_state
                episode_reward += reward

                if (len(agent.memory.buffer) > runner_model.batch_size) and is_training:
                    # Learn from experience
                    agent.replay(batch_size = runner_model.batch_size)

                if done:
                    # Episode is finished
                    if not is_training:
                        print("Reward is {} in {} steps.".format(episode_reward, step_nb))
                    finished = True

                if not is_training and step_nb > 2000:
                    print("\nBlocked !")
                    finished = True

            if is_training:
                # Save score
                scorer.add(episode_num = episode, score = episode_reward, steps = step_nb)

                # Save model every x episodes
                if episode % runner_model.score_buffer_size == 0:
                    agent.save(f"{self.save_path}/{NN_MODEL_SAVE_PATH}")
        env.close()
