"""Test-only fixtures. Not importable from production code, not
reachable from any config.

`toy_restless_world.py` was `stage2_env/world.py`: a restless-arm
simulator parameterised from CrossCheck summary statistics. It is kept
because it exercises the environment and policy mechanics cheaply at
scale, and removed from the production package because the project now
replays the actual recorded trajectories
(`stage2_env/replay_world.py`) and a calibrated stand-in for data we
already hold can only produce mechanics validation that reads as
evidence.
"""
