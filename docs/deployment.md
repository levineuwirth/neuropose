# Deployment

This page covers running NeuroPose in production — on a research
server, in a container, or as a managed system service. The target
audience is whoever is actually setting up the pipeline for a study.

!!! warning "Data handling policy"
    Before deploying NeuroPose against subject data, read the (pending)
    `docs/data-policy.md` — it describes the IRB constraints on
    retention, sharing, and derived-data handling. If you are reading
    this before the data policy has landed, **pause and ask the project
    lead** before proceeding.

## Choosing a deployment mode

| Mode | Use when | Notes |
|---|---|---|
| Local (bare) | Developer machine, one-off experiments | Fastest feedback loop. Use `neuropose process`. |
| Systemd service | Single-host lab server | Recommended for study runs. Auto-restart, log capture, clean shutdown. |
| Docker | Shared infra, CI pipelines, reproducible runs | Image build is pending commit 12. |
| Kubernetes | Multi-study labs with shared GPU pools | Not currently supported; would layer on top of the Docker image. |

## Local (bare-metal)

For one-off processing, the CLI is enough:

```bash
neuropose --config ./config.yaml process path/to/video.mp4
```

For batch mode, run the daemon in a `tmux` or `screen` session:

```bash
tmux new -s neuropose
neuropose --config ./config.yaml --verbose watch
# Ctrl-B D to detach
```

## Systemd user service

A systemd *user* unit (not a root-privileged one) is the right way to
run the daemon on a research server where the researcher owns the job
queue.

Create `~/.config/systemd/user/neuropose.service`:

```ini
[Unit]
Description=NeuroPose job daemon
After=network-online.target

[Service]
Type=simple
WorkingDirectory=%h/neuropose
Environment=XDG_DATA_HOME=%h/.local/share
ExecStart=%h/neuropose/.venv/bin/neuropose --config %h/neuropose/config.yaml watch
Restart=on-failure
RestartSec=10

[Install]
WantedBy=default.target
```

Enable it:

```bash
systemctl --user daemon-reload
systemctl --user enable --now neuropose.service
journalctl --user -u neuropose.service -f
```

The interfacer's `fcntl`-based lock file prevents a second daemon from
starting if systemd restarts it before the first instance has fully
released the lock.

## Docker

*Pending commit 12.* The plan is to ship two Dockerfiles:

- `Dockerfile` — CPU base, suitable for small studies.
- `Dockerfile.gpu` — CUDA base derived from `tensorflow/tensorflow:<pinned>-gpu`.

Both images will have the `neuropose` command as their `ENTRYPOINT` so
they can be invoked as:

```bash
docker run --rm \
  -v /srv/neuropose:/data \
  -e NEUROPOSE_DATA_DIR=/data/jobs \
  -e NEUROPOSE_MODEL_CACHE_DIR=/data/models \
  ghcr.io/.../neuropose:latest \
  watch
```

## GPU considerations

- NeuroPose delegates device selection to TensorFlow via the
  `device` field in `Settings` (`"/CPU:0"` or `"/GPU:0"`). No multi-GPU
  dispatch yet — a single daemon instance uses a single device.
- If you need to run inference on multiple GPUs in parallel, run one
  daemon per GPU with distinct `data_dir` values and divide jobs
  between them. The fcntl lock is keyed on the data directory, so
  separate daemons on separate data dirs do not conflict.
- The first call to `Estimator.process_video` triggers MeTRAbs model
  load, which in turn initializes the TensorFlow GPU runtime. Expect
  a one-time startup delay of several seconds.

## Log management

The daemon writes to stdlib `logging`. Under systemd, logs land in the
user journal. For other deployment modes, redirect stdout/stderr to
your log collector of choice — NeuroPose writes one line per event with
a structured `%(asctime)s %(levelname)-8s %(name)s: %(message)s`
format, which any log aggregator can parse.

Log verbosity is controlled via the CLI:

```bash
neuropose --verbose watch   # DEBUG
neuropose watch             # INFO (default)
neuropose --quiet watch     # WARNING
```

## Monitoring

The canonical state of the daemon lives in
`$data_dir/out/status.json`, which is a JSON object keyed by job name.
A tiny Prometheus exporter or a nightly cron that tails the file is
enough to alert on stuck jobs. A richer monitoring story is out of
scope for v0.1.

## Backups and retention

Two things are worth backing up:

1. `$data_dir/out/*/results.json` — the aggregated predictions for each
   job. These are the outputs of the research process.
2. `$data_dir/out/status.json` — the daemon's record of which jobs ran
   when, which failed, and why.

**Do not back up `$data_dir/in/` or `$data_dir/failed/` indiscriminately.**
These contain source video files that may be IRB-protected subject data,
and your backup store may not be covered by the same data-handling
agreement as the primary server. Consult the (pending)
`docs/data-policy.md` before designing a retention plan.
