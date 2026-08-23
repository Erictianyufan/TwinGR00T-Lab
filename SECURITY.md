# Security policy

## Reporting a vulnerability

Do not open a public issue for leaked credentials, unsafe command execution,
arbitrary file access, or another exploitable security problem. Use GitHub's
private security-advisory workflow for this repository so the maintainer can
review the report before disclosure.

Include the affected commit, reproduction steps, expected impact, and any safe
mitigation you have already tested. Do not attach live tokens, passwords, private
keys, or confidential datasets.

## Credential exposure

If a credential appears in a commit, terminal recording, issue, or artifact:

1. revoke or rotate it at the provider immediately;
2. remove it from active files and generated artifacts;
3. audit recent access and automation logs;
4. report the exposure privately;
5. rewrite Git history only after coordinating with the maintainer.

Deleting a visible file does not invalidate a credential and does not remove it
from repository history.

## Robot and simulator safety

This repository is a research prototype. Simulation acceptance does not certify
a policy for physical deployment. Real-robot use requires independent workspace
limits, collision checks, speed and force limits, emergency-stop hardware,
supervised commissioning, and a task-specific risk assessment.

