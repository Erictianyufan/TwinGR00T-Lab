# Contributing to TwinGR00T Lab

Contributions are welcome when they preserve the repository's evidence-first
standard and keep verified replay results separate from experimental policy
claims.

## Development setup

Dependency-light modules and tests do not require Isaac Sim:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[dev]'
```

Install `.[data]` only when working on HDF5/Parquet/video conversion and
`.[bridge]` only when working on the policy transport. Follow
[`docs/deployment.md`](docs/deployment.md) for simulator or GR00T work; do not
merge their Python environments.

## Quality gates

Before opening a pull request, run:

```bash
ruff check .
pytest
git diff --check
```

Changes that require Isaac Sim should also include the exact command, relevant
runtime versions, and a compact JSON manifest. Contact-rich demo changes should
retain the trace needed to review object pose, finger contact, lift, relative
drift, release, and settling.

## Documentation and claim labels

Use one of these labels whenever reporting project state:

- **Verified:** acceptance checks passed and review artifacts exist.
- **Implemented:** code and dependency-light tests exist, but no accepted
  simulator or policy result is claimed.
- **Experimental:** the path is runnable or under active study; results are not
  part of the accepted showcase.
- **Planned:** design only.

Every demo description must identify whether actions came from a recorded
RoboTwin trajectory, a scripted controller, or a named learned policy. Do not
use “GR00T demo” for recorded-action replay merely because GR00T integration
code exists in the same repository.

## Pull requests

Keep each pull request focused. Include:

1. the problem and intended outcome;
2. the affected task, runtime track, and controller;
3. tests and simulator commands that were run;
4. quantitative acceptance results and artifact paths;
5. any known limitations or non-reproducible dependencies.

Generated datasets, checkpoints, full-resolution videos, upstream repositories,
and provider assets should not be committed. Add only compact documentation media
that is licensed for redistribution and useful to reviewers.

## Security and credentials

Never include SSH passwords, Hugging Face tokens, cloud credentials, or device
authorization codes in commits, logs, manifests, screenshots, or issues. If a
secret is exposed, rotate it first and follow [`SECURITY.md`](SECURITY.md).

