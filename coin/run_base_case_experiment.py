from __future__ import annotations

import argparse

from traffic_opt import TrafficOptimizationPipeline, load_config_yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run traffic optimization pipeline from YAML config and save artifacts."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.base.yaml",
        help="Path to YAML config file.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config_yaml(args.config)
    pipeline = TrafficOptimizationPipeline(cfg)

    out = pipeline.run_all()  # One call: run experiment + save artifacts

    print("Run complete.")
    print(f"Origin: {out.experiment.origin}")
    print(f"Destination: {out.experiment.destination}")
    print(f"SPA reached: {out.experiment.spa.reached_fraction:.3f}")
    print(f"FF reached: {out.experiment.ff.reached_fraction:.3f}")
    print(f"FF-RL reached: {out.experiment.ff_rl.reached_fraction:.3f}")
    print(f"G SPA (hrs): {out.experiment.spa.world_utility_s / 3600.0:.2f}")
    print(f"G FF (hrs): {out.experiment.ff.world_utility_s / 3600.0:.2f}")
    print(f"G FF-RL (hrs): {out.experiment.ff_rl.world_utility_s / 3600.0:.2f}")
    if out.artifacts.routes_plot:
        print(f"Routes plot: {out.artifacts.routes_plot}")
    if out.artifacts.heatmap_plot:
        print(f"Heatmap plot: {out.artifacts.heatmap_plot}")
    if out.artifacts.animation_mp4:
        print(f"Animation MP4: {out.artifacts.animation_mp4}")
    if out.artifacts.animation_gif:
        print(f"Animation GIF: {out.artifacts.animation_gif}")


if __name__ == "__main__":
    main()
