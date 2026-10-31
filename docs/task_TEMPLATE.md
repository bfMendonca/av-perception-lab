# Task DDDD: Descriptive Task Title

## Description

Explain the problem/opportunity and what this task does. Why the change is
needed and the expected result. May include a list of the implications of the
current state when that helps justify the task.

## Scope

### Files Changed

| File | Change |
|------|--------|
| `path/to/file.py` | What changes in this file |
| `other/file.js` | What changes in this file |

<!-- Alternative: numbered list with line-by-line detail when already known -->
<!--
1. **`path/file.py`** (298 lines → ~90 lines)
   - REMOVE: function X (lines 23-45)
   - KEEP: function Y (lines 48-90)
-->

### Files NOT Changed (safety confirmation)

| File | Reason |
|------|--------|
| `path/untouched.py` | Why it stays unchanged / why it is safe |

## Planning

### Step 1: <action>

Concrete description of the step.

### Step 2: <action>

Concrete description of the step.

## Impact Analysis

| Risk | Likelihood | Mitigation |
|------|------------|------------|
| Description of the risk | Low / Medium / High | How the risk is mitigated |

## Verification

```bash
# Automated tests
<project test command>

# Lint / formatting
<project lint command>
```

Manual verification: see Test notebook (added after implementation).

## Dependencies

- **None** — this task is independent.
<!-- or: -->
- Depends on: `task_0003` (reason).
- Tasks that depend on this one: `task_0007` (reason).

<!-- On completion add `(COMPLETED)` to the title, then the sections
`## Implementation details` (what was done, deviations from the plan, notes)
and `## Test notebook` (manual checklist, `- [ ]` items). -->
