# Bootstrap Plan: versionable

The rename from `pyserializable` is complete. All source, tests, docs, and config files have been copied and renamed.
This plan covers remaining setup tasks.

## Verification (do this first)

```bash
cd /Users/hendrickmelo/Documents/projects/versionable
pixi install
pixi run cleanup
pixi run pytest
```

Fix any issues that come up. The rename script handled all identifier and import renames, but linting/formatting may
need a pass.

## Remaining Tasks

### 1. GitHub repo

```bash
gh repo create hendrickmelo/versionable --public --source=. --push
```

### 2. New logo/images

The `docs/images/` directory was intentionally not copied — it contained pyserializable branding. Create new images for
versionable and place them in `docs/images/`.

Update references in:

- `docs/conf.py` (if it references logo files)
- `docs/index.md` (if it embeds images)
- `README.md` (if it embeds images)

### 3. Update CLAUDE.md

Review `.claude/CLAUDE.md` — the rename script updated paths and class names, but you may want to revise:

- The project description/overview section
- Any shortcuts or workflow references specific to the old repo
- Architecture bullet points

### 4. Clean up docs

- `docs/skills.md` and `docs/skills-user.md` — review for coherence after rename
- `README.md` — review the full content, especially the comparison table and feature descriptions
- Consider updating the module docstring in `__init__.py` to emphasize "versioned persistence" over "serialization
  framework"

### 5. Initial commit

```bash
git add -A
git commit -m "Initial commit: versionable — versioned persistence for Python dataclasses"
```

### 6. Archive old repo

Once versionable is confirmed working, archive `hendrickmelo/pyserializable` on GitHub (Settings → Archive).
