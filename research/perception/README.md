# Research: learned floor-plan perception

Not used by the engine. Results: `docs/LEARNED_PERCEPTION_EXPERIMENT.md`.

Training uses its own environment (torch), kept outside the repository:

    python3 -m venv <scratch>/mlvenv && <scratch>/mlvenv/bin/pip install torch numpy opencv-python-headless
    <scratch>/mlvenv/bin/python research/perception/synth_data.py --count 600 --start 0     # x4 in parallel
    <scratch>/mlvenv/bin/python research/perception/train.py --iters 5000
    .venv/bin/python research/perception/prepare_real.py                                   # engine venv
    <scratch>/mlvenv/bin/python research/perception/infer.py --manifest output/perception/real/manifest.json
    .venv/bin/python research/perception/eval_real.py --variant norm --save output/perception/eval_norm.json
    .venv/bin/python research/perception/eval_zones.py

Data, weights and evaluation outputs go to output/perception/ (not versioned). Synthetic data
comes only from benchmark.generator (commercially clean).
