import gzip
import importlib.metadata
import json
import logging
import os
import pickle
import re
import sys
import time
import traceback
import uuid
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import asdict, dataclass, field, is_dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional
from urllib.parse import urljoin

import gymnasium as gym
import numpy as np
from PIL import Image
from tqdm import tqdm

from agisdk.REAL.browsergym.core.chat import Chat
from agisdk.REAL.browsergym.utils.obs import flatten_axtree_to_str, flatten_dom_to_str, prune_html

from .agent import Agent
from .utils import count_messages_token, count_tokens

logger = logging.getLogger(__name__)


@dataclass
class EnvArgs:
    task_name: str
    task_seed: int = None
    max_steps: int = None
    headless: bool = True
    record_video: bool = False
    wait_for_user_message: bool = False
    viewport: dict = None  # use default value from BrowserGym
    slow_mo: int = None  # use default value from BrowserGym
    storage_state: Optional[str | Path | dict] = None
    golden_user_data_dir: Optional[str | Path] = None  # use a golden profile directory
    extensions_dir: Optional[str | Path] = None  # directory containing Chrome extensions to load (can be a single extension or a directory of extensions)
    task_kwargs: dict = None  # use default value from BrowserGym

    def make_env(self, action_mapping, exp_dir):
        extra_kwargs = {}
        if self.record_video:
            extra_kwargs["record_video_dir"] = exp_dir
        if self.viewport:
            extra_kwargs["viewport"] = self.viewport
        if self.slow_mo is not None:
            extra_kwargs["slow_mo"] = self.slow_mo
        if self.storage_state:
            extra_kwargs["pw_context_kwargs"] = {"storage_state": self.storage_state}
        if self.golden_user_data_dir:
            extra_kwargs["golden_user_data_dir"] = str(self.golden_user_data_dir)
        if self.extensions_dir:
            extra_kwargs["extensions_dir"] = str(self.extensions_dir)
        if self.task_kwargs is not None:
            extra_kwargs["task_kwargs"] = self.task_kwargs

        return gym.make(
            _get_env_name(self.task_name),
            disable_env_checker=True,
            max_episode_steps=self.max_steps,
            headless=self.headless,
            wait_for_user_message=self.wait_for_user_message,
            action_mapping=action_mapping,  # action mapping is provided by the agent
            **extra_kwargs,
        )


@dataclass
class AbstractAgentArgs(ABC):
    """A template class that defines the required signature of an agent's arguments."""

    agent_name: str = None

    def __post_init__(self):
        if self.agent_name is None:
            self.agent_name = self.__class__.__name__

    def prepare(self):
        """Prepare the agent's LLM models before running the experiment."""
        pass

    def close(self):
        """Close the agent's LLM models after running the experiment."""
        pass

    @abstractmethod
    def make_agent(self) -> Agent:
        """Comply the experiments.loop API for instantiating the agent."""


def save_package_versions(exp_dir: Path):
    """Save the versions of the installed packages in the experiment directory."""
    python_dists = "\n".join(
        sorted(
            [
                f'{dist.metadata["Name"]}=={dist.metadata["Version"]}'
                for dist in importlib.metadata.distributions()
            ]
        )
    )
    (exp_dir / "package_versions.txt").write_text(python_dists)


@dataclass
class ExpArgs:
    """Arguments to run an experiment, i.e. run agent in an environment until done.

    This dataclass is used to store experiments arguments. It contains
    agent_args and env_args which follows the same principle. It contains helper
    functions to prepare and run experiments.

    Attributes:
    -----------
    agent_args: AbstractAgentArgs
        The arguments to instantiate the agent.
    env_args: EnvArgs
        The arguments to instantiate the environment.
    exp_dir: str
        The directory where the experiment will be saved.
    exp_name: str
        The name of the experiment. If None, it will be generated from the
        agent and environment names.
    enable_debug: bool
        If python is running in debug mode and `enable_debug` is True, errors
        will be raised instead of only logged
    error_msg: str
        Error that occured while running the experiment (if any).
    stack_trace: str
        Stack trace of the error (if any).
    order: int (internal)
        The order of the experiment in the batch. It is used to keep track of
        the original order of the experiments in case they are shuffled.
    """

    agent_args: AbstractAgentArgs
    env_args: EnvArgs
    exp_dir: str = None
    exp_name: str = None
    enable_debug: bool = True
    err_msg: str = None
    stack_trace: str = None
    order: int = None  # use to keep the original order the experiments were meant to be launched.
    logging_level: int = logging.INFO
    exp_id: str = None
    depends_on: tuple[str] = ()
    save_screenshot: bool = False
    save_som: bool = False
    save_step_info_pkl: bool = False
    model_name: str = None
    post_run_js_snippet: str = None
    post_run_js_snippet_path: str = None
    post_run_url: str = None
    initial_delay: float = 0

    def prepare(self, exp_root):
        """Prepare the experiment directory and save the experiment arguments.

        This enables inspecting experiments that are not run yet.
        """
        if self.env_args.task_seed is None:
            self.env_args.task_seed = np.random.randint(0, 1000)

        if self.exp_name is None:
            task_name = self.env_args.task_name
            self.exp_name = f"{self.agent_args.agent_name}_on_{task_name}_{self.env_args.task_seed}"

        if self.exp_id is None:  # reuse the same task_id if it's a relaunch
            self.exp_id = str(uuid.uuid4().hex)

        # if exp_dir exists, it means it's a re-run, move the old one
        if self.exp_dir is not None:
            _move_old_exp(self.exp_dir)

        self.exp_date = datetime.now()
        self._make_dir(exp_root)

        self.exp_dir.mkdir(parents=True, exist_ok=True)
        with open(self.exp_dir / "exp_args.pkl", "wb") as f:
            pickle.dump(self, f)

    def _make_dir(self, exp_root):
        """Create a unique directory for the experiment."""
        date_str = self.exp_date.strftime("%Y-%m-%d_%H-%M-%S")

        for i in range(1000):
            if i >= 999:  # make sure we don't loop forever
                raise ValueError("Could not find a unique name for the experiment directory.")

            tag = f"_{i}" if i > 0 else ""
            random_id = str(uuid.uuid4().hex)
            self.exp_dir = Path(exp_root) / f"{date_str}_{self.exp_name}_{random_id}{tag}"
            if not self.exp_dir.exists():
                break

    # TODO distinguish between agent error and environment or system error. e.g.
    # the parsing error of an action should not be re-run.
    def run(self):
        """Run the experiment and save the results"""

        # start writing logs to run logfile
        self._set_logger()

        # log python environment info
        save_package_versions(self.exp_dir)

        episode_info = []
        env, agent, step_info, err_msg, stack_trace = None, None, None, None, None
        post_run_js_result, post_run_js_error = None, None
        post_run_page_url, post_run_page_content = None, None
        post_run_page_html, post_run_page_axtree, post_run_page_error = None, None, None
        
        try:
            logger.info(f"Running experiment {self.exp_name} in:\n  {self.exp_dir}")

            # Check agent type and pass dimensions if needed
            if self.agent_args.__class__.__name__ == "OperatorAgentArgs":
                viewport = self.env_args.viewport
                if viewport and isinstance(viewport, dict) and "width" in viewport and "height" in viewport:
                    browser_dimensions = (viewport["width"], viewport["height"])
                    agent = self.agent_args.make_agent(browser_dimensions=browser_dimensions)
                else:
                    err_msg = f"OperatorAgent requires browser dimensions, but viewport is missing/invalid in env_args: {viewport}"
                    logger.error(err_msg)
                    raise ValueError(err_msg)
            else:
                # Default agent creation for other types
                agent = self.agent_args.make_agent()
            logger.debug(f"Agent created.")

            # Determine action mapping (handle cases where agent might not have action_set, e.g., Operator)
            action_mapping = None
            if hasattr(agent, 'action_set') and agent.action_set is not None:
                 action_mapping = agent.action_set.to_python_code

            env = self.env_args.make_env(
                action_mapping=action_mapping, # Pass mapping or None
                exp_dir=self.exp_dir
            )
            logger.debug(f"Environment created.")

            # Set the agent name on the environment instance
            env.unwrapped.active_agent_name = self.agent_args.agent_name

            step_info = StepInfo(step=0)
            episode_info = [step_info]
            step_info.from_reset(
                env, seed=self.env_args.task_seed, obs_preprocessor=agent.obs_preprocessor
            )
            logger.debug(f"Environment reset.")

            if self.initial_delay > 0:
                logger.info(f"Waiting {self.initial_delay}s for page to fully load before first action.")
                time.sleep(self.initial_delay)
                step_info.obs = env.unwrapped._get_obs()
                if agent.obs_preprocessor:
                    step_info.obs = agent.obs_preprocessor(step_info.obs)

            while not step_info.is_done:  # set a limit
                logger.debug(f"Starting step {step_info.step}.")
                action = step_info.from_action(agent)
                logger.debug(f"Agent chose action:\n {action}")

                if action is None:
                    # will end the episode after saving the step info.
                    step_info.truncated = True

                step_info.save_step_info(
                    self.exp_dir,
                    save_screenshot=self.save_screenshot,
                    save_som=self.save_som,
                    save_pkl=self.save_step_info_pkl,
                )
                logger.debug(f"Step info saved.")

                _send_chat_info(env.unwrapped.chat, action, step_info.agent_info)
                logger.debug(f"Chat info sent.")

                if action is None:
                    logger.debug(f"Agent returned None action. Ending episode.")
                    break

                step_info = StepInfo(step=step_info.step + 1)
                episode_info.append(step_info)

                logger.debug(f"Sending action to environment.")
                step_info.from_step(env, action, obs_preprocessor=agent.obs_preprocessor)
                logger.debug(f"Environment stepped.")

        except Exception as e:
            err_msg = f"Exception uncaught by agent or environment in task {self.env_args.task_name}.\n{type(e).__name__}:\n{e}"
            stack_trace = traceback.format_exc()

            self.err_msg = err_msg
            self.stack_trace = stack_trace

            logger.warning(err_msg + "\n" + stack_trace)
            if _is_debugging() and self.enable_debug:
                raise

        finally:
            try:
                if step_info is not None:
                    step_info.save_step_info(
                        self.exp_dir,
                        save_screenshot=self.save_screenshot,
                        save_som=self.save_som,
                        save_pkl=self.save_step_info_pkl,
                    )
            except Exception as e:
                logger.error(f"Error while saving step info in the finally block: {e}")
            try:
                if env is not None and self.post_run_url:
                    (
                        post_run_page_url,
                        post_run_page_content,
                        post_run_page_html,
                        post_run_page_axtree,
                        post_run_page_error,
                    ) = _capture_post_run_page(
                        env,
                        agent,
                        self.post_run_url,
                    )

                if env is not None and self.post_run_js_snippet:
                    post_run_js_result, post_run_js_error = _evaluate_post_run_javascript(
                        env.page,
                        self.post_run_js_snippet,
                    )
                if (
                    not err_msg
                    and len(episode_info) > 0
                    and not (episode_info[-1].terminated or episode_info[-1].truncated)
                ):
                    e = KeyboardInterrupt("Early termination??")
                    err_msg = f"Exception uncaught by agent or environment in task {self.env_args.task_name}.\n{type(e).__name__}:\n{e}"
                _save_summary_info(
                    episode_info,
                    self.exp_dir,
                    err_msg,
                    stack_trace,
                    post_run_url=self.post_run_url,
                    post_run_js_result=post_run_js_result,
                    post_run_js_error=post_run_js_error,
                    post_run_js_snippet_path=self.post_run_js_snippet_path,
                    post_run_page_url=post_run_page_url,
                    post_run_page_content=post_run_page_content,
                    post_run_page_html=post_run_page_html,
                    post_run_page_axtree=post_run_page_axtree,
                    post_run_page_error=post_run_page_error,
                )
            except Exception as e:
                logger.error(f"Error while saving summary info in the finally block: {e}")
            try:
                if env is not None:
                    env.close()
            except Exception as e:
                logger.error(f"Error while closing the environment in the finally block: {e}")
            try:
                self._unset_logger()  # stop writing logs to run logfile
            except Exception as e:
                logger.error(f"Error while unsetting the logger in the finally block: {e}")

    def _set_logger(self):
        # output logging traces to a log file
        file_handler = logging.FileHandler(self.exp_dir / "experiment.log")
        file_handler.setLevel(self.logging_level)  # same level as console outputs
        formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        file_handler.setFormatter(formatter)
        # setup root logger
        root_logger = logging.getLogger()
        root_logger.setLevel(self.logging_level)
        root_logger.addHandler(file_handler)
        # setup openai logger (don't go below INFO verbosity)
        openai_logger = logging.getLogger("openai._base_client")
        openai_logger.setLevel(max(logging.INFO, self.logging_level))

        self.logging_file_handler = file_handler

    def _unset_logger(self):
        root_logger = logging.getLogger()
        root_logger.removeHandler(self.logging_file_handler)


@dataclass
class StepTimestamps:
    env_start: float = 0
    action_exec_start: float = 0  # to extract begining of visual action from video
    action_exec_stop: float = 0  # to extract end of visual action from video
    action_exect_after_timeout: float = 0
    env_stop: float = 0
    agent_start: float = 0
    agent_stop: float = 0


@dataclass
class StepInfo:
    """Collects information about step that will be saved and reloaded.
    Helper functions only modify the dataclass attributes and helps keeping the
    information organized.

    Attributes:
    -----------
    step: int
        The step number of the episode.
    obs: dict
        The observation of the environment.
    reward: float
        The reward of the step.
    raw_reward: float
        The raw reward of the step.
    terminated: bool
        Whether the episode is terminated i.e. reached a terminal state.
    truncated: bool
        Whether the episode is truncated i.e. reached a maximum number of steps.
    action: Optional[str]
        The action taken by the agent. `None` means the agent stopped without producing an action.
    agent_info: dict
        Additional information from the agent.
    stats: dict
        Extra statistics about the step.
    profiling: StepTimestamps
        Timestamps of the different events during the episode.
    """

    step: int = None
    obs: dict = None
    reward: float = 0
    raw_reward: float = 0
    terminated: bool = None
    truncated: bool = None
    action: Optional[str] = None
    agent_info: dict = field(default_factory=dict)
    stats: dict = None
    profiling: StepTimestamps = field(default_factory=StepTimestamps)
    task_info: dict = None
    model_name: str = None

    def from_step(self, env: gym.Env, action: Optional[str], obs_preprocessor: callable):
        t = self.profiling
        t.env_start = time.time()
        self.obs, self.reward, self.terminated, self.truncated, env_info = env.step(action)
        t.env_stop = time.time()

        self.task_info = env_info.get("task_info", None)

        self.raw_reward = env_info.get("RAW_REWARD_GLOBAL", None)

        t.action_exec_start = env_info["action_exec_start"]  # start
        t.action_exect_after_timeout = env_info["action_exec_stop"]
        t.action_exec_stop = env_info["action_exec_stop"] - env_info["action_exec_timeout"]

        if obs_preprocessor:
            self.obs = obs_preprocessor(self.obs)

    def from_action(self, agent: Agent):
        self.profiling.agent_start = time.time()
        self.action, self.agent_info = agent.get_action(self.obs.copy())
        self.profiling.agent_stop = time.time()

        self.make_stats()

        return self.action

    def from_reset(self, env: gym.Env, seed: int, obs_preprocessor: callable):
        t = self.profiling
        t.env_start = time.time()
        self.obs, env_info = env.reset(seed=seed)
        self.reward, self.terminated, self.truncated = 0, False, False
        t.env_stop = time.time()

        t.action_exec_start = env_info.get("recording_start_time", t.env_start)
        t.action_exect_after_timeout = t.env_stop
        t.action_exec_stop = t.env_stop

        if obs_preprocessor:
            self.obs = obs_preprocessor(self.obs)

    @property
    def is_done(self):
        return self.terminated or self.truncated

    def make_stats(self):

        stats = {
            f"n_token_{key}": count_tokens(val)
            for key, val in self.obs.items()
            if isinstance(val, str)
        }
        stats.update(self.agent_info.pop("stats", {}))

        messages = self.agent_info.get("chat_messages", None)
        if messages is not None:
            stats["n_token_agent_messages"] = count_messages_token(messages)

        t = self.profiling
        stats["step_elapsed"] = t.env_stop - t.env_start
        stats["agent_elapsed"] = t.agent_stop - t.agent_start

        self.stats = stats

    def save_step_info(self, exp_dir, save_json=False, save_screenshot=True, save_som=False, save_pkl=True):

        screenshot = self.obs.pop("screenshot", None)
        screenshot_som = self.obs.pop("screenshot_som", None)
        # Temporarily remove browser object to avoid serialization issues
        browser = self.obs.pop("browser", None) if self.obs and "browser" in self.obs else None

        if save_screenshot and screenshot is not None:
            img = Image.fromarray(screenshot)
            img.save(exp_dir / f"screenshot_step_{self.step}.png")

        if save_som and screenshot_som is not None:
            img = Image.fromarray(screenshot_som)
            img.save(exp_dir / f"screenshot_som_step_{self.step}.png")

        # save goal object (which might contain images) to a separate file to save space
        if self.obs is not None and self.obs.get("goal_object", False):
            # save the goal object only once (goal should never change once setup)
            goal_object_file = Path(exp_dir) / "goal_object.pkl.gz"
            if not goal_object_file.exists():
                with gzip.open(goal_object_file, "wb") as f:
                    pickle.dump(self.obs["goal_object"], f)
            # set goal_object to a special placeholder value, which indicates it should be loaded from a separate file
            self.obs["goal_object"] = None

        if save_pkl:
            with gzip.open(exp_dir / f"step_{self.step}.pkl.gz", "wb") as f:
                # TODO should we pop the screenshots too before this to save space ?
                pickle.dump(self, f)

        if save_json:
            with open(exp_dir / "steps_info.json", "w") as f:
                json.dump(self, f, indent=4, cls=DataclassJSONEncoder)

        # add the screenshots back to the obs
        if screenshot is not None:
            self.obs["screenshot"] = screenshot
        if screenshot_som is not None:
            self.obs["screenshot_som"] = screenshot_som
        # Restore browser object
        if browser is not None:
            self.obs["browser"] = browser


def _extract_err_msg(episode_info: list[StepInfo]):
    """Extract the last error message from the episode info."""
    errors = [(None, None)]
    for step_info in episode_info:
        if step_info.agent_info is None:
            continue
        err_msg = step_info.agent_info.get("err_msg", None)
        if err_msg is not None:
            errors.append((err_msg, step_info.agent_info.get("stack_trace", None)))

    return errors[-1]


def _aggregate_episode_stats(episode_info: list[StepInfo]):
    """Aggregate StepInfo.stats across episodes.

    It will compute the sum and max of each value in the stats dict.
    These two summaries should cover many use cases. If more are needed, the
    user can compute other stats by reloading individual StepInfo.
    """

    stats = defaultdict(list)
    for step_info in episode_info:
        if step_info.stats is not None:
            for key, val in step_info.stats.items():
                if val is None:
                    val = np.nan
                stats[key].append(val)

    aggregated_stats = {"cum_steps": len(episode_info)}  # to be able to compute the mean
    for key, val_list in stats.items():
        aggregated_stats[f"cum_{key}"] = np.nansum(val_list)
        aggregated_stats[f"max_{key}"] = np.nanmax(val_list)

    for key, val in aggregated_stats.items():
        if isinstance(val, np.generic):
            aggregated_stats[key] = val.item()
        if np.isnan(val):
            aggregated_stats[key] = None
    return aggregated_stats


def _save_summary_info(
    episode_info: list[StepInfo],
    exp_dir,
    err_msg,
    stack_trace,
    post_run_url=None,
    post_run_js_result=None,
    post_run_js_error=None,
    post_run_js_snippet_path=None,
    post_run_page_url=None,
    post_run_page_content=None,
    post_run_page_html=None,
    post_run_page_axtree=None,
    post_run_page_error=None,
):
    # bring err from agent_info to the top level
    if err_msg is None:
        err_msg, stack_trace = _extract_err_msg(episode_info)
    else:
        # useful until we get a proper place in agent_xray to view error
        # messages.
        if len(episode_info) == 0:
            episode_info.append(StepInfo())
        episode_info[-1].agent_info["err_msg"] = err_msg
        episode_info[-1].agent_info["stack_trace"] = stack_trace

    # Load existing summary info if it exists (preserving metadata)
    summary_info = {}
    summary_info_path = exp_dir / "summary_info.json"
    if summary_info_path.exists():
        try:
            with open(summary_info_path, "r") as f:
                summary_info = json.load(f)
        except Exception:
            # If we can't load the existing file, start fresh
            pass
    
    last_agent_step = _get_last_agent_step(episode_info)
    agent_response = _extract_agent_response(last_agent_step)
    raw_agent_response = _extract_raw_agent_response(last_agent_step)

    # Extract the full prompt from the first step's agent_info
    full_prompt = None
    if len(episode_info) > 0 and episode_info[0].agent_info:
        full_prompt = episode_info[0].agent_info.get("full_prompt")
    
    # Extract task_id from path if possible
    task_id = None
    exp_dir_str = str(exp_dir)
    # Try to extract the task name/ID from the directory path
    task_match = re.search(r'on_([\w.-]+)_', exp_dir_str)
    if task_match:
        task_id = task_match.group(1)
    
    # Check if there was an error
    had_error = err_msg is not None and len(err_msg) > 0
    
    # Calculate success based on rewards and errors
    success = False
    if len(episode_info) > 0:
        # Consider success if terminal state reached with positive reward and no errors
        success = (episode_info[-1].terminated and 
                  sum([step.reward for step in episode_info]) > 0 and 
                  not had_error)
    
    # Extract finish state if available
    finish_state = {}
    if len(episode_info) > 0 and episode_info[-1].task_info:
        finish_state = episode_info[-1].task_info

    capture_post_run_artifacts = bool(post_run_js_snippet_path or post_run_page_url)
    final_page_html = None
    final_page_axtree = None
    final_page_content = None
    if not capture_post_run_artifacts and len(episode_info) > 0 and episode_info[-1].obs:
        final_page_html = episode_info[-1].obs.get("pruned_html")
        final_page_axtree = episode_info[-1].obs.get("axtree_txt")
        final_page_content = final_page_html or final_page_axtree
    
    # Get configuration from the task info if available
    config = {}
    if len(episode_info) > 0 and episode_info[-1].obs and "config" in episode_info[-1].obs:
        config = episode_info[-1].obs["config"]
    elif finish_state and "config" in finish_state:
        config = finish_state["config"]
    
    # Update with new results data
    summary_info.update({
        # Legacy fields (keep for backward compatibility)
        "n_steps": len(episode_info) - 1,
        "cum_reward": sum([step.reward for step in episode_info]),
        "cum_raw_reward": sum([step.raw_reward for step in episode_info if step.raw_reward]),
        "err_msg": err_msg,
        "stack_trace": stack_trace,
        "experiment_status": "completed",
        
        # New fields matching the desired format
        "completed": True,
        "success": success,
        "error": had_error,
        "score": float(sum([step.raw_reward for step in episode_info if step.raw_reward]) or 0.0),
        "task_id": task_id or "",
        "full_prompt": full_prompt,
        "agent_response": agent_response,
        "raw_agent_response": raw_agent_response,
        "finish_state": finish_state,
        "finish_page_content": final_page_content,
        "finish_page_html": final_page_html,
        "finish_page_axtree": final_page_axtree,
        "post_run_url": post_run_url,
        "post_run_js_snippet_path": post_run_js_snippet_path,
        "post_run_js_result": post_run_js_result,
        "post_run_js_error": post_run_js_error,
        "post_run_page_url": post_run_page_url,
        "post_run_page_content": post_run_page_content,
        "post_run_page_html": post_run_page_html,
        "post_run_page_axtree": post_run_page_axtree,
        "post_run_page_error": post_run_page_error,
        "eval_results": [],  # This would need to be populated by an evaluation system
        "env_setup_error": err_msg if "Executable doesn't exist" in str(err_msg) or "playwright" in str(err_msg) else None,
    })
    
    # Add stats
    for key, val in _aggregate_episode_stats(episode_info).items():
        summary_info[f"stats.{key}"] = val

    if len(episode_info) > 0:
        summary_info["terminated"] = episode_info[-1].terminated
        summary_info["truncated"] = episode_info[-1].truncated

    agent_outputs_path, agent_output_text_path = _save_agent_outputs(
        exp_dir=exp_dir,
        episode_info=episode_info,
        agent_response=agent_response,
        raw_agent_response=raw_agent_response,
        post_run_url=post_run_url,
        post_run_js_result=post_run_js_result,
        post_run_js_error=post_run_js_error,
        post_run_js_snippet_path=post_run_js_snippet_path,
        post_run_page_url=post_run_page_url,
        post_run_page_content=post_run_page_content,
        post_run_page_html=post_run_page_html,
        post_run_page_axtree=post_run_page_axtree,
        post_run_page_error=post_run_page_error,
    )
    summary_info["agent_outputs_path"] = str(agent_outputs_path.resolve())
    summary_info["agent_output_text_path"] = str(agent_output_text_path.resolve())

    # Write updated summary info
    with open(summary_info_path, "w") as f:
        json.dump(summary_info, f, indent=4)


def _get_last_agent_step(episode_info: list[StepInfo]) -> Optional[StepInfo]:
    for step_info in reversed(episode_info):
        agent_info = step_info.agent_info or {}
        if step_info.action:
            return step_info
        if (
            agent_info.get("model_response")
            or agent_info.get("raw_model_response")
            or agent_info.get("chat_messages")
        ):
            return step_info
    return episode_info[-1] if episode_info else None


def _extract_agent_response(step_info: Optional[StepInfo]) -> str:
    if step_info is None:
        return ""

    agent_info = step_info.agent_info or {}

    chat_messages = agent_info.get("chat_messages") or []
    for msg in reversed(chat_messages):
        if msg.get("role") == "assistant":
            return msg.get("message", "")

    model_response = agent_info.get("model_response")
    if model_response:
        return model_response

    if step_info.action and "send_msg_to_user" in step_info.action:
        match = re.search(r'send_msg_to_user\("(.+?)"\)', step_info.action)
        if match:
            return match.group(1)

    task_info = step_info.task_info or {}
    for criterion in task_info.get("criteria", []):
        if criterion.get("model_response"):
            return criterion.get("model_response")

    return step_info.action or ""


def _extract_raw_agent_response(step_info: Optional[StepInfo]) -> str:
    if step_info is None:
        return ""

    agent_info = step_info.agent_info or {}

    raw_model_response = agent_info.get("raw_model_response")
    if raw_model_response:
        return raw_model_response

    return _extract_agent_response(step_info)


def _build_agent_outputs_payload(
    episode_info: list[StepInfo],
    agent_response: str,
    raw_agent_response: str,
    post_run_url,
    post_run_js_result,
    post_run_js_error,
    post_run_js_snippet_path,
    post_run_page_url,
    post_run_page_content,
    post_run_page_html,
    post_run_page_axtree,
    post_run_page_error,
):
    steps = []
    for step_info in episode_info:
        agent_info = step_info.agent_info or {}
        if not (
            step_info.action
            or agent_info.get("model_response")
            or agent_info.get("raw_model_response")
            or agent_info.get("chat_messages")
            or agent_info.get("err_msg")
        ):
            continue

        steps.append(
            {
                "step": step_info.step,
                "action": step_info.action,
                "model_response": agent_info.get("model_response"),
                "raw_model_response": agent_info.get("raw_model_response"),
                "chat_messages": agent_info.get("chat_messages"),
                "err_msg": agent_info.get("err_msg"),
                "stack_trace": agent_info.get("stack_trace"),
                "terminated": step_info.terminated,
                "truncated": step_info.truncated,
            }
        )

    primary_output = post_run_js_result
    if primary_output in (None, ""):
        primary_output = agent_response
    if primary_output in (None, "") and steps:
        primary_output = steps[-1].get("model_response") or steps[-1].get("action") or ""

    return {
        "primary_output": primary_output,
        "agent_response": agent_response,
        "raw_agent_response": raw_agent_response,
        "post_run_url": post_run_url,
        "post_run_js_result": post_run_js_result,
        "post_run_js_error": post_run_js_error,
        "post_run_js_snippet_path": post_run_js_snippet_path,
        "post_run_page_url": post_run_page_url,
        "post_run_page_content": post_run_page_content,
        "post_run_page_html": post_run_page_html,
        "post_run_page_axtree": post_run_page_axtree,
        "post_run_page_error": post_run_page_error,
        "steps": steps,
    }


def _save_agent_outputs(
    exp_dir,
    episode_info: list[StepInfo],
    agent_response: str,
    raw_agent_response: str,
    post_run_url,
    post_run_js_result,
    post_run_js_error,
    post_run_js_snippet_path,
    post_run_page_url,
    post_run_page_content,
    post_run_page_html,
    post_run_page_axtree,
    post_run_page_error,
):
    payload = _build_agent_outputs_payload(
        episode_info=episode_info,
        agent_response=agent_response,
        raw_agent_response=raw_agent_response,
        post_run_url=post_run_url,
        post_run_js_result=post_run_js_result,
        post_run_js_error=post_run_js_error,
        post_run_js_snippet_path=post_run_js_snippet_path,
        post_run_page_url=post_run_page_url,
        post_run_page_content=post_run_page_content,
        post_run_page_html=post_run_page_html,
        post_run_page_axtree=post_run_page_axtree,
        post_run_page_error=post_run_page_error,
    )

    agent_outputs_path = Path(exp_dir) / "agent_outputs.json"
    with open(agent_outputs_path, "w") as f:
        json.dump(payload, f, indent=4, cls=DataclassJSONEncoder)

    agent_output_text_path = Path(exp_dir) / "agent_output.txt"
    primary_output = payload.get("primary_output")
    if primary_output is None:
        primary_output = ""
    elif not isinstance(primary_output, str):
        primary_output = json.dumps(primary_output, indent=2)
    agent_output_text_path.write_text(primary_output)

    return agent_outputs_path, agent_output_text_path


def _make_json_safe(value):
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, list):
        return [_make_json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [_make_json_safe(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _make_json_safe(val) for key, val in value.items()}
    return repr(value)


def _extract_post_run_capture_result(capture):
    prompts = capture.get("prompts") or []
    if len(prompts) == 1:
        prompt_value = prompts[0].get("defaultValue")
        if prompt_value is not None:
            return prompt_value
        return prompts[0].get("message")
    if len(prompts) > 1:
        return [
            prompt.get("defaultValue")
            if prompt.get("defaultValue") is not None
            else prompt.get("message")
            for prompt in prompts
        ]

    clipboard_writes = capture.get("clipboardWrites") or []
    if len(clipboard_writes) == 1:
        return clipboard_writes[0]
    if len(clipboard_writes) > 1:
        return clipboard_writes

    alerts = capture.get("alerts") or []
    confirms = capture.get("confirms") or []
    if alerts or confirms:
        return capture

    return None


def _extract_post_run_page_artifacts(post_run_obs: dict):
    post_run_page_html = post_run_obs.get("pruned_html")
    post_run_page_axtree = post_run_obs.get("axtree_txt")

    if post_run_page_html is None and post_run_obs.get("dom_object") is not None:
        post_run_page_html = prune_html(flatten_dom_to_str(post_run_obs["dom_object"]))
    if post_run_page_axtree is None and post_run_obs.get("axtree_object") is not None:
        post_run_page_axtree = flatten_axtree_to_str(post_run_obs["axtree_object"])

    post_run_page_content = post_run_page_html or post_run_page_axtree
    return post_run_page_content, post_run_page_html, post_run_page_axtree


def _normalize_post_run_javascript_result(raw_result):
    if not isinstance(raw_result, dict) or "__agisdk_has_value" not in raw_result:
        return _make_json_safe(raw_result)

    if raw_result.get("__agisdk_has_value"):
        value_box = raw_result.get("__agisdk_value_box")
        if isinstance(value_box, dict) and value_box.get("type") == "undefined":
            return "undefined"
        if isinstance(value_box, dict) and value_box.get("type") == "value":
            return _make_json_safe(value_box.get("value"))
        return _make_json_safe(raw_result.get("__agisdk_value"))

    capture = raw_result.get("__agisdk_capture") or {}
    inferred_result = _extract_post_run_capture_result(capture)
    if inferred_result is not None:
        return _make_json_safe(inferred_result)

    return None


def _build_post_run_wrapper(script_body: str, expect_value: bool):
    has_value_literal = "true" if expect_value else "false"
    return f"""async () => {{
        const __agisdk_boxValue = (value) => {{
            if (value === undefined) {{
                return {{ type: "undefined" }};
            }}
            return {{ type: "value", value }};
        }};
        const __agisdk_capture = {{
            prompts: [],
            alerts: [],
            confirms: [],
            clipboardWrites: [],
        }};
        const __agisdk_originalPrompt = window.prompt;
        const __agisdk_originalAlert = window.alert;
        const __agisdk_originalConfirm = window.confirm;
        const __agisdk_originalClipboardWriteText =
            navigator.clipboard && navigator.clipboard.writeText
                ? navigator.clipboard.writeText.bind(navigator.clipboard)
                : null;
        let __agisdk_clipboardOverridden = false;

        window.prompt = (message = "", defaultValue = "") => {{
            __agisdk_capture.prompts.push({{
                message: String(message),
                defaultValue: defaultValue == null ? null : String(defaultValue),
            }});
            return defaultValue ?? "";
        }};
        window.alert = (message = "") => {{
            __agisdk_capture.alerts.push(String(message));
        }};
        window.confirm = (message = "") => {{
            __agisdk_capture.confirms.push(String(message));
            return true;
        }};
        if (__agisdk_originalClipboardWriteText) {{
            try {{
                navigator.clipboard.writeText = async (text) => {{
                    __agisdk_capture.clipboardWrites.push(text == null ? null : String(text));
                    return __agisdk_originalClipboardWriteText(text);
                }};
                __agisdk_clipboardOverridden = true;
            }} catch (error) {{
                __agisdk_clipboardOverridden = false;
            }}
        }}

        try {{
            let __agisdk_value;
            let __agisdk_has_value = {has_value_literal};
            {script_body}
            return {{
                __agisdk_has_value: __agisdk_has_value,
                __agisdk_value: __agisdk_value,
                __agisdk_value_box: __agisdk_boxValue(__agisdk_value),
                __agisdk_capture: __agisdk_capture,
            }};
        }} finally {{
            window.prompt = __agisdk_originalPrompt;
            window.alert = __agisdk_originalAlert;
            window.confirm = __agisdk_originalConfirm;
            if (__agisdk_clipboardOverridden) {{
                navigator.clipboard.writeText = __agisdk_originalClipboardWriteText;
            }}
        }}
    }}"""


def _evaluate_post_run_javascript(page, script_source: str):
    script_source = script_source.strip()
    if not script_source:
        return None, None

    candidate_scripts = []
    candidate_scripts.append(
        _build_post_run_wrapper(
            f"__agisdk_value = await (0, eval)({json.dumps(script_source)});",
            expect_value=True,
        )
    )
    if (
        script_source.startswith("(")
        or script_source.startswith("async ")
        or script_source.startswith("function")
    ):
        candidate_scripts.append(
            _build_post_run_wrapper(
                f"""
            __agisdk_value = await (async () => {{
                const __agisdk_candidate = {script_source};
                if (typeof __agisdk_candidate === "function") {{
                    return await __agisdk_candidate();
                }}
                return await __agisdk_candidate;
            }})();
                """,
                expect_value=True,
            )
        )
    else:
        candidate_scripts.append(
            _build_post_run_wrapper(
                f"__agisdk_value = await (async () => ({script_source}))();",
                expect_value=True,
            )
        )
        candidate_scripts.append(
            _build_post_run_wrapper(
                f"""
            __agisdk_has_value = false;
            await (async () => {{
{script_source}
            }})();
                """,
                expect_value=False,
            )
        )

    last_error = None
    for candidate in candidate_scripts:
        try:
            result = page.evaluate(candidate)
            return _normalize_post_run_javascript_result(result), None
        except Exception as exc:
            last_error = exc

    return None, repr(last_error) if last_error else None


def _capture_post_run_page(env, agent, post_run_url: str):
    resolved_url = urljoin(env.page.url, post_run_url)
    capture_error = None

    try:
        env.page.goto(resolved_url, wait_until="domcontentloaded")
    except Exception as exc:
        capture_error = repr(exc)

    try:
        env.page.wait_for_load_state("domcontentloaded", timeout=5000)
    except Exception as exc:
        if capture_error is None:
            capture_error = repr(exc)

    try:
        post_run_obs = env._get_obs()
        if agent is not None and getattr(agent, "obs_preprocessor", None):
            post_run_obs = agent.obs_preprocessor(post_run_obs)

        post_run_page_content, post_run_page_html, post_run_page_axtree = (
            _extract_post_run_page_artifacts(post_run_obs)
        )

        return (
            resolved_url,
            post_run_page_content,
            post_run_page_html,
            post_run_page_axtree,
            capture_error,
        )
    except Exception as exc:
        if capture_error is None:
            capture_error = repr(exc)

    try:
        page_html = env.page.content()
        page_text = None
        try:
            page_text = env.page.locator("body").inner_text(timeout=1000)
        except Exception:
            pass
        return resolved_url, page_text or page_html, page_html, None, capture_error
    except Exception as exc:
        if capture_error is None:
            capture_error = repr(exc)

    return resolved_url, None, None, None, capture_error


def _is_debugging():
    """Tells you if your code is currently running in debug mode."""
    return sys.gettrace() is not None


class ExpResult:
    """Helper class to load and visualize the results of an experiment.

    attributes are loaded lazily.

    Attributes (lazily loaded):
        exp_args: ExpArgs, the arguments of the experiment.
        steps_info: list[StepInfo], the information of each steps so far
        summary_info: dict, the summary of the experiment.
        screenshots: list[Image], the screenshots of each step.
        screenshots_som: list[Image], the screenshots of each step with set of
            marks inprinted.
        flat_exp_args: dict, the flattened version of exp_args.
        chat_video_path: Path, the path to the chat video. (if record_video=True)
        task_video_path: Path, the path to the task video. (if record_video=True)
        combined_video_path: Path, the path to the combined video. (if video was
            combined)
    """

    def __init__(self, exp_dir) -> None:
        self.exp_dir = Path(exp_dir)
        self._exp_args = None
        self._steps_info = {}
        self._summary_info = None
        self._screenshots = {}
        self._flat_exp_args = None
        self._logs = None

    @property
    def exp_args(self) -> ExpArgs:
        if self._exp_args is None:
            with open(self.exp_dir / "exp_args.pkl", "rb") as f:
                self._exp_args = pickle.load(f)
                # in case experiments were moved
                self._exp_args.exp_dir = self.exp_dir
        return self._exp_args

    def get_step_info(self, step: int) -> StepInfo:
        """Load the step info from the file and return it."""
        if self._steps_info.get(step, None) is None:
            with gzip.open(self.exp_dir / f"step_{step}.pkl.gz", "rb") as f:
                self._steps_info[step] = pickle.load(f)
            if "screenshot" not in self._steps_info[step].obs:
                try:
                    self._steps_info[step].obs["screenshot"] = np.array(
                        self.get_screenshot(step), dtype=np.uint8
                    )
                except FileNotFoundError:
                    pass
            if "screenshot_som" not in self._steps_info[step].obs:
                try:
                    self._steps_info[step].obs["screenshot_som"] = np.array(
                        self.get_screenshot(step, som=True), dtype=np.uint8
                    )
                except FileNotFoundError:
                    pass
        # if goal_object is set to None, it indicates it has been saved into a separate file
        if (
            self._steps_info[step].obs
            and "goal_object" in self._steps_info[step].obs
            and self._steps_info[step].obs["goal_object"] is None
        ):
            with gzip.open(self.exp_dir / "goal_object.pkl.gz", "rb") as f:
                goal_object = pickle.load(f)
                self._steps_info[step].obs["goal_object"] = goal_object

        return self._steps_info[step]

    @property
    def steps_info(self) -> list[StepInfo]:
        step_files = list(self.exp_dir.glob("step_*.pkl.gz"))
        for file in step_files:
            step = int(file.name.split("_")[-1].split(".")[0])
            self.get_step_info(step)

        return [self._steps_info[i] for i in range(len(self._steps_info))]

    @property
    def summary_info(self) -> dict:
        if self._summary_info is None:
            with open(self.exp_dir / "summary_info.json", "r") as f:
                # if length is zero raise file not found error
                if os.fstat(f.fileno()).st_size == 0:
                    raise FileNotFoundError(f"summary_info.json is empty.")
                self._summary_info = json.load(f)
        return self._summary_info

    def get_screenshot(self, step: int, som=False) -> Image:
        key = (step, som)
        if self._screenshots.get(key, None) is None:
            file_name = f"screenshot_{'som_' if som else ''}step_{step}"
            try:
                with Image.open(self.exp_dir / (file_name + ".png")) as img:
                    self._screenshots[key] = img.copy()
            except FileNotFoundError:
                with Image.open(self.exp_dir / (file_name + ".jpg")) as img:
                    self._screenshots[key] = img.copy()
        return self._screenshots[key]

    def get_screenshots(self, som=False):
        files = list(self.exp_dir.glob("screenshot_step_*"))
        max_step = 0
        for file in files:
            step = int(file.name.split("_")[-1].split(".")[0])
            self.get_screenshot(step, som=som)
            max_step = max(max_step, step)
        return [self._screenshots.get((i, som), None) for i in range(max_step + 1)]

    @property
    def screenshots(self):
        return self.get_screenshots(som=False)

    @property
    def screenshots_som(self):
        return self.get_screenshots(som=True)

    @property
    def flat_exp_args(self) -> dict:
        """Return a dict with exp_args flattened."""
        if self._flat_exp_args is None:
            exp_args = asdict(self.exp_args)
            # this will flatten nested dicts
            self._flat_exp_args = _flatten_dict(exp_args)
        return self._flat_exp_args

    def get_exp_record(self) -> dict:
        """Return a dict with exp_args flattened and summary_info."""
        record = {"exp_dir": self.exp_dir}
        try:
            record.update(self.flat_exp_args)
        except FileNotFoundError:
            pass
        try:
            record.update(self.summary_info)
        except FileNotFoundError:
            pass
        return record

    @property
    def chat_video_path(self) -> Path:
        try:
            return next(self.exp_dir.glob("chat_video/*.webm"))
        except StopIteration:
            raise FileNotFoundError(f"No chat_video found in {self.exp_dir}")

    @property
    def task_video_path(self) -> Path:
        try:
            return next(self.exp_dir.glob("task_video/*.webm"))
        except StopIteration:
            raise FileNotFoundError(f"No task_video found in {self.exp_dir}")

    @property
    def combined_video_path(self) -> Path:
        return self.exp_dir / "combined_video.mp4"

    @property
    def logs(self):
        if self._logs is None:
            self._logs = (self.exp_dir / "experiment.log").read_text()
        return self._logs


EXP_RESULT_CACHE = {}


def get_exp_result(exp_dir) -> ExpResult:
    """Keep a cache of pre-loaded exp_results for faster loading"""
    exp_dir = str(exp_dir)  # make sure it's not a Path
    exp_result = EXP_RESULT_CACHE.get(exp_dir, None)
    if exp_result is None:
        exp_result = ExpResult(exp_dir)
        EXP_RESULT_CACHE[exp_dir] = exp_result
    return exp_result


def yield_all_exp_results(
    savedir_base: str | Path, progress_fn=tqdm, load_hidden=False, use_cache=True
):
    """Recursively find all experiments from savedir_base folder.

    This will ignore all experiments that start with "_" or ".". use
    `load_hidden=True` to load them anyway.
    """

    if not isinstance(savedir_base, list):
        savedir_base = [savedir_base]

    exp_args_paths = []
    for exp_dir in savedir_base:
        exp_args_paths.extend(list(Path(exp_dir).glob("**/exp_args.pkl")))

    if progress_fn is not None:
        exp_args_paths = progress_fn(exp_args_paths, desc="Searching experiments directories.")

    for exp_args_path in exp_args_paths:
        exp_dir = exp_args_path.parent
        if not load_hidden:
            if exp_dir.name.startswith("_") or exp_dir.name.startswith("."):
                continue
        if use_cache:
            yield get_exp_result(exp_dir)
        else:
            yield ExpResult(exp_dir)


class DataclassJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if is_dataclass(obj):
            return asdict(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def _move_old_exp(exp_dir):
    """Move the old experiment directory to a new name."""
    exp_dir = Path(exp_dir)
    if exp_dir.exists():
        exp_dir.rename(exp_dir.with_name("_" + exp_dir.name))


def _get_env_name(task_name: str):
    """Register tasks if needed (lazy import) and return environment name."""

    # lazy benchmark import
    # if task_name.startswith("miniwob"):
    #     import browsergym.miniwob
    # elif task_name.startswith("workarena"):
    #     import browsergym.workarena
    # elif task_name.startswith("webarena"):
    #     import browsergym.webarena
    # elif task_name.startswith("visualwebarena"):
    #     import browsergym.visualwebarena
    if task_name.startswith("webclones"):
        import agisdk.REAL.browsergym.webclones
    # Adding exception for 'eval' tasks we registered.
    elif not task_name.startswith("eval"):
        raise ValueError(
            f"Task {task_name} not found. Please register the task in browsergym."
        )

    return f"browsergym/{task_name}"


def _send_chat_info(chat: Chat, action: Optional[str], agent_info: dict):
    """Send the think and action info to the chat."""
    msg = ""
    if "think" in agent_info:
        msg += f"""\
{agent_info["think"]}

"""

    msg += f"""\
action:
{action}
"""

    logger.info(msg)
    chat.add_message(role="info", msg=msg)


def _flatten_dict(d, parent_key="", sep="."):
    """Recursively flatten a nested dictionary."""
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, dict):
            items.extend(_flatten_dict(v, new_key, sep).items())
        elif isinstance(v, Path):
            items.append((new_key, str(v)))
        else:
            items.append((new_key, v))
    return dict(items)
