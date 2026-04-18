# Scene `oneshot.py` Patch Guide

This repo includes a reproducible patch script for the `scene` environment's `jsqlsim` low-cost missile tracking and lead prediction.

## Target file

Default patch targets:

```text
C:\Users\Tk\.conda\envs\scene\Lib\site-packages\jsqlsim\world\policy\oneshot.py
C:\Users\Tk\.conda\envs\scene\Lib\site-packages\jsqlsim\world\game_state.py
```

## Apply the patch

Run from this repo root:

```powershell
& 'C:\Users\Tk\.conda\envs\scene\python.exe' .\patch_scene_oneshot.py
```

What it does:

- creates timestamped backups next to `oneshot.py` and `game_state.py`
- patches low-cost intercept prediction in `oneshot.py`
- patches detector-history recording in `game_state.py` so stale tracks do not keep polluting `GLOBAL_HISTORY`
- runs `py_compile` on the patched files
- writes patch metadata to `test/archive/scene_patches/`

## Optional debug logging

To inspect the new low-cost intercept timing during a run:

```powershell
$env:JSQLSIM_ONESHOT_DEBUG="1"
```

Expected log fields:

- `weapon`
- `target_id`
- `distance_m`
- `raw_delta_t`
- `intercept_t`
- `history_points`
- `track_age`
- `raw_target_pos`
- `predicted_target_pos`

## Restore the original file

If you need to roll back manually, restore the latest backups created beside `oneshot.py` and `game_state.py`:

```powershell
Copy-Item 'C:\Users\Tk\.conda\envs\scene\Lib\site-packages\jsqlsim\world\policy\oneshot.py.bak.YYYYMMDD_HHMMSS' 'C:\Users\Tk\.conda\envs\scene\Lib\site-packages\jsqlsim\world\policy\oneshot.py' -Force
Copy-Item 'C:\Users\Tk\.conda\envs\scene\Lib\site-packages\jsqlsim\world\game_state.py.bak.YYYYMMDD_HHMMSS' 'C:\Users\Tk\.conda\envs\scene\Lib\site-packages\jsqlsim\world\game_state.py' -Force
```

Then optionally re-run:

```powershell
& 'C:\Users\Tk\.conda\envs\scene\python.exe' -m py_compile 'C:\Users\Tk\.conda\envs\scene\Lib\site-packages\jsqlsim\world\policy\oneshot.py'
& 'C:\Users\Tk\.conda\envs\scene\python.exe' -m py_compile 'C:\Users\Tk\.conda\envs\scene\Lib\site-packages\jsqlsim\world\game_state.py'
```

## Notes

- This patch changes only `oneshot.py` and `game_state.py` inside the `scene` environment.
- High-cost missiles, cruise missiles, intercept missiles, and repo-side RED logic are left untouched.
- The patch expands `GLOBAL_HISTORY` capacity at runtime from 4 to 6, collapses duplicate same-tick samples, compensates for stale-track age, stops writing obviously stale repeated detector positions back into history each frame, and adds a recent-velocity + linear-intercept blend for low-cost missile prediction.
