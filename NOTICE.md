# Source and external dependencies

This repository extracts the simulation implementation developed for the user's
BEHAVIOR active-perception experiments into a standalone package. It does not
contain the planner, model services, model weights, demonstrations, or proprietary
scene assets.

The simulator requires a separately installed and licensed BEHAVIOR-1K /
OmniGibson environment and NVIDIA Isaac Sim. Their code and assets retain their
respective licenses. Runtime versions used during extraction are recorded in
`behavior_ap_sim/configs/simulator-source.json`. The R1Pro configuration preserves
the experiment's original physical controllers and assisted-grasp setting.

The rendering and bounded video-encoding design was informed by
https://github.com/rascal-yang/Isaac-sim-renderer at commit
`eccdecde583b2e4fc74ef8bb2cc84ba9d6dc6d02`. Its G1 serializer, model, and assets are
not incorporated. The implementation here uses OmniGibson native observations
and retains physics on the simulator's owning thread.

No license for NVIDIA or BEHAVIOR assets is granted by cloning this repository.
