# NeuroPose

3D human pose estimation pipeline for clinical movement research, built on
[MeTRAbs](https://github.com/isarandi/metrabs). Developed by the Shu Lab at
Brown University.

!!! warning "Pre-alpha software"
    NeuroPose is under active development at version `0.1.0.dev0`. APIs,
    schemas, and the command-line interface may change without notice
    between commits until the first tagged release. This is research
    software and **must not** be used for clinical decision-making.

## What NeuroPose does

NeuroPose takes a video (or a directory of videos organised into "jobs"),
runs the MeTRAbs 3D pose-estimation model on every frame, and produces a
validated JSON output containing per-frame 3D and 2D joint positions and
the original video's metadata (frame count, fps, resolution). The output
schema is designed to be loaded back into Python, numpy, or any downstream
analysis pipeline without ambiguity.

Three core components:

- **`neuropose.estimator`** — the per-video inference worker. Streams
  frames from an input video, runs MeTRAbs on each one, and returns a
  validated `VideoPredictions` object. No filesystem or job-queue
  semantics.
- **`neuropose.interfacer`** — a filesystem-polling daemon that watches an
  input directory for new job subdirectories, dispatches each to the
  estimator, and manages the status-file lifecycle.
- **`neuropose.analyzer`** — a post-processing subpackage for motion
  analysis and classification (FastDTW, joint-angle features, sktime).
  *(Pending the rewrite in commit 10.)*

## Where to go next

<div class="grid cards" markdown>

- :material-rocket-launch: **[Getting Started](getting-started.md)** —
  install, run your first job, understand the output.

- :material-cube-outline: **[Architecture](architecture.md)** — how the
  pieces fit together and why.

- :material-api: **[API Reference](api/config.md)** — auto-generated from
  the source docstrings.

- :material-tools: **[Development](development.md)** — contributing,
  testing, and the release workflow.

- :material-server: **[Deployment](deployment.md)** — running the daemon
  in production.

</div>

## Intended use

NeuroPose is built for:

- Clinical gait and movement-assessment research
- Biomechanics work using standard RGB video
- Research reproducibility — the output schema carries enough metadata
  (frame count, fps, resolution) to recover real time from frame indices
  without needing access to the original video.

It is **not** intended for:

- Clinical diagnosis or treatment decisions.
- General-purpose motion capture outside the research use cases actively
  supported by the Shu Lab.

## Citing NeuroPose

If you use NeuroPose in academic work, please cite it using the metadata
in [`CITATION.cff`](https://git.levineuwirth.org/neuwirth/neuropose/src/branch/main/CITATION.cff).
A DOI and a manuscript citation will be added once the first paper is
submitted.

## License and attribution

NeuroPose is distributed under the MIT License. It builds on MeTRAbs
(Copyright &copy; 2020 István Sárándi), also distributed under MIT. Full
attribution lives in [`AUTHORS.md`](https://git.levineuwirth.org/neuwirth/neuropose/src/branch/main/AUTHORS.md).
