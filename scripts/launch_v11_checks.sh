#!/usr/bin/env bash
# Start, watch or stop the v1.1 on-box checks (scripts/run_v11_checks.sh) as a detached
# systemd user unit, so the run survives closing the terminal, the editor or an assistant
# session — a plain background job dies with whatever started it.
#
#   scripts/launch_v11_checks.sh check     # seconds: preflight + which steps are done
#   scripts/launch_v11_checks.sh           # start the full run (~10 h)
#   scripts/launch_v11_checks.sh smoke     # start the ~40-minute end-to-end smoke test
#   scripts/launch_v11_checks.sh status    # is it running? + the step log so far
#   scripts/launch_v11_checks.sh stop      # stop it (resumable: just start again)
#
# The run skips every step whose output already exists, so `start` after a stop, a crash
# or a reboot resumes where it left off; a stop costs at most the step in progress. If
# the date has changed since the first start, pass the original one so the run
# directories line up:
#
#   V11_DATE=2026-11-17 scripts/launch_v11_checks.sh
#
# Extra arguments after `start` go to the runner (e.g. `start --stages transfer,probe-l1`).
# Stay logged in while it runs (a screen lock is fine): user units stop at logout.
set -euo pipefail
cd "$(dirname "$0")/.."

CMD="${1:-start}"; [ $# -gt 0 ] && shift
UNITS=(v11-checks v11-checks-smoke)
LOGS=(runs/v11-checks-overnight.log runs/v11c-smoke.log)
case "$CMD" in
  start) UNIT=v11-checks;       ARGS=("$@");         LOG=runs/v11-checks-overnight.log ;;
  smoke) UNIT=v11-checks-smoke; ARGS=(--smoke "$@"); LOG=runs/v11c-smoke.log ;;
  status|stop) ;;
  check) exec bash scripts/run_v11_checks.sh --check "$@" ;;
  *) echo "usage: $0 [check|start|smoke|status|stop] [runner args]"; exit 2 ;;
esac

if [ "$CMD" = status ] || [ "$CMD" = stop ]; then
  for u in "${UNITS[@]}"; do
    state=$(systemctl --user is-active "$u" 2>/dev/null || true)
    echo "$u: ${state:-inactive}"
    [ "$CMD" = stop ] && [ "$state" = active ] && systemctl --user stop "$u" && echo "  stopped"
  done
  for log in "${LOGS[@]}"; do
    [ -f "$log" ] && { echo "--- $log (steps):"; grep -a '^=== 20' "$log" | tail -16; }
  done
  exit 0
fi

for u in "${UNITS[@]}" v11-seed-sweep v11-seed-sweep-smoke; do
  if systemctl --user is-active --quiet "$u"; then
    echo "$u is already running — '$0 status' to watch, '$0 stop' to stop"; exit 1
  fi
done
systemctl --user reset-failed "$UNIT" 2>/dev/null || true

# The machine must not suspend under the run.
if command -v gsettings >/dev/null 2>&1; then
  ac=$(gsettings get org.gnome.settings-daemon.plugins.power sleep-inactive-ac-type 2>/dev/null || true)
  [ "$ac" = "'nothing'" ] || echo "WARNING: automatic suspend on AC is '$ac' — disable it while this runs (Settings > Power)"
fi

ENVS=(--setenv=PATH="$PATH" --setenv=HOME="$HOME")
[ -n "${V11_DATE:-}" ] && ENVS+=(--setenv=V11_DATE="$V11_DATE")
systemd-run --user --quiet --collect --unit="$UNIT" --working-directory="$PWD" \
  "${ENVS[@]}" bash scripts/run_v11_checks.sh "${ARGS[@]}"

echo "started $UNIT — log: $LOG"
echo "  watch:  tail -f $LOG        (or: $0 status)"
echo "  stop:   $0 stop             (resumable)"
