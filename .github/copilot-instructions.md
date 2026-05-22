**Prompt:**
Always ensure that any Python command is run inside an activated virtual environment (`.venv`).
Before running a Python command, activate the virtual environment using:

```bash
source .venv/bin/activate
```

Use `uv` to run installations, tests, and other Python commands to ensure they are executed within the virtual environment.

For example, to install a package, use:

```bash
uv pip install <package-name>
```

Additionally, treat this repository as a **learning-focused, small-scale ML project**:

- Prioritize documenting key real-world ML concepts in **structured markdown documents** (for example under `docs/`), and keep them up to date alongside the code.
- Assume this is **not a production-grade ML system**: use **industry-standard libraries and patterns**, but keep the dataset size and system scale small and avoid unnecessary production complexity.
- Develop the project **incrementally in clearly defined phases** (as outlined in `docs/archive/PRD.md`), where **each phase produces a concrete, working milestone** (for example: project skeleton, key concept docs, dataset exploration, training loop, model evaluation, web UI), and later phases add features/complexity on top.
 - After **each completed development phase**, update `README.md` to reflect the current capabilities, limitations, and how to run the system at that phase.
 - When planning or making non-trivial changes, **always consult** `docs/archive/PRD.md` and other available planning docs to stay aligned with the agreed scope, requirements, and phase milestones.