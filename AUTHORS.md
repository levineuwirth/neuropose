# Authors

NeuroPose is developed by the Shu Lab at Brown University and collaborators.

## Core team

| Role | Name | Contact |
|---|---|---|
| Principal investigator | **Dr. Liqi Shu** | [liqi_shu@brown.edu](mailto:liqi_shu@brown.edu) |
| Project lead, pose estimation, infrastructure | **Levi Neuwirth** | [ln@levineuwirth.org](mailto:ln@levineuwirth.org) |
| Analysis design | **David Man** | [david_man@brown.edu](mailto:david_man@brown.edu) |
| Front-end and technical documentation | **Praneeth Tummala** | [praneeth_tummala@brown.edu](mailto:praneeth_tummala@brown.edu) |

Levi Neuwirth is the current maintainer and leads day-to-day development of
this rewrite.

## Upstream and third-party acknowledgment

NeuroPose wraps and depends on [**MeTRAbs**](https://github.com/isarandi/metrabs)
by István Sárándi ([istvan.sarandi@uni-tuebingen.de](mailto:istvan.sarandi@uni-tuebingen.de)),
distributed under the MIT License (Copyright (c) 2020). The core 3D pose
estimation capability of NeuroPose is entirely attributable to MeTRAbs;
NeuroPose contributes the pipeline, configuration, batching, analysis, and
deployment layers around it.

Additional open-source dependencies are enumerated in
[`pyproject.toml`](pyproject.toml) and their licenses are retained in the
installed package metadata.

## Contributing

See [`CONTRIBUTING.md`](CONTRIBUTING.md) (to be added in a later commit) for
how to propose changes. New contributors will be listed here once their first
change is merged.
