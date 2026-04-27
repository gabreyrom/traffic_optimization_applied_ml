from __future__ import annotations

import logging
import random
import time
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

from traffic_opt_experiment import (
    TrafficOptimizationExperiment,
    animate_routes_three_way_light,
    fraction_reached_agentspecific,
    plot_heatmaps_three_way,
    plot_routes_three_way,
    run_ff_expert_coin_rl_experiment,
)

from .config import AppConfig
from .models import ArtifactPaths, ExperimentRunResult, PipelineOutput, ScenarioResult


class TrafficOptimizationPipeline:
    """
    Orchestrates the full run with config objects and structured logging.
    """

    def __init__(self, config: AppConfig):
        self.config = config
        self.logger = logging.getLogger(config.logging.logger_name)
        self._configure_logging(config.logging.level)
        self.exp = TrafficOptimizationExperiment(
            shapefile_path=config.network.shapefile_path,
            seed=config.network.seed,
        )

    def _configure_logging(self, level: str) -> None:
        if not logging.getLogger().handlers:
            logging.basicConfig(
                level=getattr(logging, level.upper(), logging.INFO),
                format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
            )
        self.logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    def _resolve_fixed_od(self) -> Optional[Tuple[tuple[float, float], tuple[float, float]]]:
        if self.config.network.od_mode != "fixed":
            return None
        if self.config.network.fixed_od is not None:
            return self.config.network.fixed_od
        if self.exp.origin is None or self.exp.destination is None:
            self.exp.pick_fixed_od()
        return self.exp.origin, self.exp.destination

    def _prepare(self) -> None:
        random.seed(self.config.network.seed)
        np.random.seed(self.config.network.seed)
        if self.exp.gdf is None:
            self.logger.info("Loading shapefile: %s", self.config.network.shapefile_path)
            self.exp.load_data()
        if self.exp.network is None:
            self.logger.info("Building traffic network")
            self.exp.build_network()
        if self.config.network.od_mode == "fixed":
            _ = self._resolve_fixed_od()
            self.logger.info("Using fixed OD: origin=%s destination=%s", self.exp.origin, self.exp.destination)

    def run_experiment(self) -> ExperimentRunResult:
        self._prepare()
        assert self.exp.network is not None

        e = self.config.experiment
        num_agents_expert = e.num_agents_expert if e.num_agents_expert is not None else max(e.num_agents // 3, 33)
        fixed_od = self._resolve_fixed_od()

        self.logger.info(
            "Running experiment: agents=%s, episodes=%s, od_mode=%s",
            e.num_agents,
            e.num_episodes_rl,
            self.config.network.od_mode,
        )
        t0 = time.time()
        (
            agents_spa,
            flows_spa,
            g_spa,
            agents_ff,
            flows_ff,
            g_ff,
            agents_ff_rl,
            flows_ff_rl,
            g_ff_rl,
            q_ff_rl,
        ) = run_ff_expert_coin_rl_experiment(
            self.exp.network,
            alpha=e.alpha,
            gamma=e.gamma,
            temp_start=e.temp_start,
            temp_end=e.temp_end,
            epsilon_eval=0.0,
            temperature=e.temperature,
            lambda_live_cost=e.lambda_live_cost,
            visited_penalty_val=e.visited_penalty_val,
            lambda_dr=e.lambda_dr,
            lambda_prog=e.lambda_prog,
            lambda_exp=e.lambda_exp,
            lambda_step=e.lambda_step,
            visited_penalty_train=e.visited_penalty_train,
            lambda_time_train=e.lambda_time_train,
            time_scale_s=e.time_scale_s,
            t_cap_s=e.t_cap_s,
            tau=e.tau,
            time_scale=e.time_scale,
            lambda_term=e.lambda_term,
            lambda_miss=e.lambda_miss,
            early_stop_enabled=e.early_stop_enabled,
            early_stop_patience=e.early_stop_patience,
            early_stop_min_delta=e.early_stop_min_delta,
            early_stop_min_episodes=e.early_stop_min_episodes,
            early_stop_target_reach=e.early_stop_target_reach,
            early_stop_low_reach_threshold=e.early_stop_low_reach_threshold,
            early_stop_stagnation_window=e.early_stop_stagnation_window,
            num_agents_baseline=e.num_agents,
            num_agents_expert=num_agents_expert,
            num_agents_eval_rl=e.num_agents,
            od_mode=self.config.network.od_mode,
            fixed_od=fixed_od,
            num_episodes_rl=e.num_episodes_rl,
            steps_per_agent=e.steps_per_agent,
            horizon_s=e.horizon_s,
            debug=e.debug,
        )
        elapsed = time.time() - t0

        spa = ScenarioResult(
            agents=agents_spa,
            flows=flows_spa,
            world_utility_s=g_spa,
            reached_fraction=fraction_reached_agentspecific(agents_spa),
        )
        ff = ScenarioResult(
            agents=agents_ff,
            flows=flows_ff,
            world_utility_s=g_ff,
            reached_fraction=fraction_reached_agentspecific(agents_ff),
        )
        ff_rl = ScenarioResult(
            agents=agents_ff_rl,
            flows=flows_ff_rl,
            world_utility_s=g_ff_rl,
            reached_fraction=fraction_reached_agentspecific(agents_ff_rl),
        )

        self.logger.info(
            "Run done in %.1fs | G(hrs): SPA=%.2f FF=%.2f FF-RL=%.2f",
            elapsed,
            spa.world_utility_s / 3600.0,
            ff.world_utility_s / 3600.0,
            ff_rl.world_utility_s / 3600.0,
        )
        return ExperimentRunResult(
            spa=spa,
            ff=ff,
            ff_rl=ff_rl,
            q_table=q_ff_rl,
            origin=self.exp.origin,
            destination=self.exp.destination,
            duration_s=elapsed,
            metadata={
                "seed": self.config.network.seed,
                "od_mode": self.config.network.od_mode,
                "horizon_s": e.horizon_s,
                "num_agents": e.num_agents,
                "num_episodes_rl": e.num_episodes_rl,
            },
        )

    def save_artifacts(self, run: ExperimentRunResult) -> ArtifactPaths:
        if self.exp.gdf is None or self.exp.network is None:
            raise RuntimeError("save_artifacts requires prepared data/network")

        a = self.config.artifacts
        e = self.config.experiment
        out_dir = Path(a.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        paths = ArtifactPaths()

        if a.save_routes_plot:
            p = out_dir / "routes_three_way.png"
            self.logger.info("Saving routes plot: %s", p)
            plot_routes_three_way(
                self.exp.gdf,
                run.spa.agents,
                run.spa.world_utility_s,
                run.ff.agents,
                run.ff.world_utility_s,
                run.ff_rl.agents,
                run.ff_rl.world_utility_s,
                num_agents=e.num_agents,
                horizon_s=e.horizon_s,
                out_path=str(p),
            )
            paths.routes_plot = str(p)

        if a.save_heatmap_plot:
            p = out_dir / "heatmaps_three_way.png"
            self.logger.info("Saving heatmap plot: %s", p)
            plot_heatmaps_three_way(
                self.exp.gdf,
                self.exp.network,
                run.spa.flows,
                run.ff.flows,
                run.ff_rl.flows,
                num_agents=e.num_agents,
                horizon_s=e.horizon_s,
                out_path=str(p),
            )
            paths.heatmap_plot = str(p)

        if a.save_animation_mp4:
            p = out_dir / "routes_three_way.mp4"
            self.logger.info("Saving animation MP4: %s", p)
            animate_routes_three_way_light(
                self.exp.gdf,
                run.spa.agents,
                run.ff.agents,
                run.ff_rl.agents,
                num_agents=e.num_agents,
                out_path=str(p),
                max_frames=a.animation_frames,
                fps=a.animation_fps,
            )
            paths.animation_mp4 = str(p)

        if a.save_animation_gif:
            p = out_dir / "routes_three_way.gif"
            self.logger.info("Saving animation GIF: %s", p)
            animate_routes_three_way_light(
                self.exp.gdf,
                run.spa.agents,
                run.ff.agents,
                run.ff_rl.agents,
                num_agents=e.num_agents,
                out_path=str(p),
                max_frames=a.animation_frames,
                fps=a.animation_fps,
            )
            paths.animation_gif = str(p)

        return paths

    def run_all(self) -> PipelineOutput:
        run = self.run_experiment()
        artifacts = self.save_artifacts(run)
        return PipelineOutput(experiment=run, artifacts=artifacts)
