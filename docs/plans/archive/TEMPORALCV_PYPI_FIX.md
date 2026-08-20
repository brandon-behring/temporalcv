# temporalcv PyPI repair — your runbook (2026-06-12)

**Status**: ARCHIVED 2026-08-20 (RESOLVED) — confirmed via PyPI's own JSON API (`1.0.0`/`1.0.0rc1`
yanked, `2.0.0`→`2.3.0` live under the correct repo), a live `pip download` of the `2.3.0` wheel, and
`double_ml_time_series`'s own CHANGELOG `[1.1.1] - 2026-06-12` independently documenting the fix date
plus a 71/71 + 339-test parity check before flipping its pin to `temporalcv>=2.0.0,<3`. No fleet
consumer still works around the old bug. Moved here from the fleet root
(`~/Claude/TEMPORALCV_PYPI_FIX.md`), where it had sat as a misleadingly "open runbook."

Everything you personally need to do for Track C. Two web-UI tasks: **fix the
trusted publisher now (= Step 0 of the plan)**, and **yank the stale releases
later** (I'll remind you). Both are on pypi.org; neither has an API/CLI path,
which is why they're yours.

---

## 1. Why this is broken (30-second version)

1. `temporalcv 1.0.0` was published to PyPI in April 2026 **from the old dev
   mirror repo** (`brandonmbehring-dev/temporalcv`). PyPI's trusted-publisher
   entry was created to match *that* repo.
2. The project then migrated to canonical **`brandon-behring/temporalcv`**.
   Trusted publishing matches the exact OIDC claim tuple
   (owner / repo / workflow / environment) — the migration silently severed it.
3. Every publish since has been refused with **`invalid-publisher`**:
   - `v1.0.1` (run 25137515461, 2026-04-29) — so PyPI's 1.0.0 also lacks the
     "trust-repair patch".
   - `v2.0.0` (run 27268242838, 2026-06-10) — the current release.
   In both cases the **build succeeded**; only PyPI's token exchange refused.
4. Result: PyPI serves stale, known-flawed 1.0.0 while the real release (2.0.0)
   exists only as a git tag.

---
continue
## 2. TASK A = PLAN STEP 0 — Fix the trusted publisher (~2 min, do now)

### Prerequisites
- Log in to <https://pypi.org> as the account that owns `temporalcv`
  (package author: Brandon Behring). PyPI requires 2FA — have your second
  factor ready.

### Steps
1. Top-right menu → **Your projects** → **temporalcv** → **Manage**.
2. Left sidebar → **Publishing**.
3. Under **Manage current publishers** you should see a GitHub publisher
   pointing at the old mirror (`brandonmbehring-dev/temporalcv`).
   **Remove it** (recommended — it's stale; nothing publishes from there).
4. Under **Add a new publisher**, GitHub tab, enter EXACTLY:

   | Field | Value |
   |---|---|
   | Owner | `brandon-behring` |
   | Repository name | `temporalcv` |


   | Workflow name | `publish.yml` |
   | Environment name | `pypi` |

5. Click **Add**. The publisher list should now show
   `brandon-behring/temporalcv` / `publish.yml` / environment `pypi`.

### Gotchas
- **Workflow name is the bare filename** — `publish.yml`, NOT
  `.github/workflows/publish.yml`.
- **Environment name is required and must be `pypi`** (lowercase). The
  workflow's publish job runs with `environment: pypi`, so that claim is part
  of the token; leaving the field blank on PyPI's side fails the match too.
- Use the **project's** Publishing page, not the account-level "pending
  publishers" page — that one is only for projects that don't exist yet.
- These values are not guesses — they're copied from the OIDC claims the
  failed run actually presented (see §4).

### Then
Tell Claude **"done"** or **"continue"**. Everything after is automated:
rerun publish run `27268242838` → verify 2.0.0 on PyPI → prove the wheel
against the 71/71 golden suite → dml_ts pin-flip PR (`temporalcv>=2.0.0,<3`)
→ v1.1.1 tag + release.

---

## 3. TASK B — Yank stale releases (~1 min, AFTER 2.0.0 is live)

Don't do this until Claude confirms 2.0.0 is on PyPI.

1. pypi.org → **temporalcv** → **Manage** → **Releases**.
2. For **`1.0.0`**: Options ▾ → **Yank** → reason:
   `Superseded by 2.0.0; predates the v1.0.1 trust-repair patch.`
3. Repeat for **`1.0.0rc1`**.

Notes:
- Yank ≠ delete. Yanked releases stay installable for anyone who pins
  `temporalcv==1.0.0` exactly, but resolvers stop choosing them. Fully
  reversible from the same menu.
- Why: with `temporalcv<2` style constraints, pip would otherwise resolve to
  1.0.0 — a build that's missing a fix that never managed to publish.

---

## 4. If the rerun STILL fails (contingency — Claude debugs, you re-edit)

Locked decision: we stay on OIDC trusted publishing; **no API-token / manual
twine fallback**. If `invalid-publisher` recurs, Claude pulls the rendered
claims from the new failed run and diffs them against your publisher entry.
For reference, the claims the v2.0.0 run presented were:

```
sub:              repo:brandon-behring/temporalcv:environment:pypi
repository:       brandon-behring/temporalcv
repository_owner: brandon-behring
workflow_ref:     brandon-behring/temporalcv/.github/workflows/publish.yml@refs/tags/v2.0.0
environment:      pypi
```

Every field in the §2 table must match its claim character-for-character.
The usual culprits: a typo in the owner (hyphen placement), a path instead of
a filename in "Workflow name", or a blank/mismatched environment.

Troubleshooting reference: <https://docs.pypi.org/trusted-publishers/troubleshooting/>

---

## 5. State of the world (so this file is self-contained)

- PyPI `temporalcv`: **1.0.0** + 1.0.0rc1 only; metadata still points at the
  mirror repo. Target: **2.0.0** from the immutable git tag `v2.0.0`
  (commit `3e43849`).
- dml_ts: at v1.1.0, functionally fine — it pins the git tag directly
  (`temporalcv @ git+…@v2.0.0`, proven by 13 green PRs + 71/71 goldens). The
  PyPI repair is about (a) outside users getting a stale package and (b)
  removing the direct-URL dependency that blocks any future dml_ts PyPI
  upload.
- Full plan + 11 locked decisions:
  `~/.claude/plans/use-the-following-handoff-mighty-phoenix.md`
