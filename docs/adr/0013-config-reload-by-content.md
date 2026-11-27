# 0013: Detect config changes by content, not file events

- **Status:** accepted
- **Date:** 2026-11-27

## Context

PLAN.md §4 planned the routing-config file watch with `watchfiles` (inotify). While
testing it on the development machine (WSL2), two problems showed up:

- On WSL, `watchfiles` always uses its polling backend. That backend compares file
  modification times at one-second resolution, so a second save within the same second
  as the previous one is never seen. In the component test, the original write and an
  edit straight after it in the same second were missed outright. For a config file,
  that means a gateway quietly running an older version than the file on disk.
- The file the gateway reads is usually not a plain local file:
  - in Docker it is a bind mount, which on some platforms delivers no inotify events;
  - in Kubernetes it is a ConfigMap, which swaps a `..data` symlink rather than writing
    the file.

  Each of these needs its own workaround in an event-based design.

## Decision

The gateway reads the file every `SWITCHBOARD_CONFIG_WATCH_SECONDS` (default 1 s; 0
turns it off) and reloads when the SHA-256 of its bytes differs from the last one seen.

- The read runs in a worker thread.
- A file that is briefly missing, mid-replace by an editor, is skipped for that tick.
- The comparison is against the last bytes seen, not the active config, so a broken file
  is reported once rather than every second.
- `SIGHUP` (`make reload`) reloads immediately. `watchfiles` is no longer imported.

## Consequences

- Every change is seen, on every file system, within a second. A one-second delay is
  fine for routing changes.
- The cost is one small file read and hash per second per replica.
- One reload path for files, symlink swaps and bind mounts, with nothing
  platform-specific in it.
- Revisit if configs grow large enough for the read to matter, which would take
  megabytes. A stat-based pre-check (`st_mtime_ns`, size, inode) before reading would
  then be the cheap fix.
