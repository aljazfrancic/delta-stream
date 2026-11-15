#!/usr/bin/env bash
# Start, watch or stop the v1.1 seed sweep (scripts/run_v11_seeds.sh) as a detached
# systemd user unit, so it survives closing the terminal, the editor or an assistant
# session — a plain background job dies with whatever started it.
#
#   scripts/launch_v11_overnight.sh           # start the full sweep (overnight)
#   scripts/launch_v11_overnight.sh smoke     # start the few-minute smoke test instead
#   scripts/launch_v11_overnight.sh status    # is it running? + the step log so far
#   scripts/launch_v11_overnight.sh stop      # stop it (resumable: just start again)
#
# The sweep skips every step whose output already exists, so `start` after a stop, a
# crash or a reboot resumes where it left off. If the date has changed since the first
# start, pass the original one so the run directories line up:
#
#   V11_DATE=2026-11-15 scripts/launch_v11_overnight.sh
#
# Stay logged in while it runs (a screen lock is fine): user units stop at logout.
set -euo pipefail
cd "$(dirname "$0")/.."

CMD="${1:-start}"
case "$CMD" in
  start) UNIT=v11-seed-sweep;       ARGS=();        LOG=runs/v11-seeds-overnight.log ;;
  smoke) UNIT=v11-seed-sweep-smoke; ARGS=(--smoke); LOG=runs/v11-smoke.log ;;
  status|stop) ;;
  *) echo "usage: $0 [start|smoke|status|stop]"; exit 2 ;;
esac

if [ "$CMD" = status ] || [ "$CMD" = stop ]; then
  for u in v11-seed-sweep v11-seed-sweep-smoke; do
    state=$(systemctl --user is-active "$u" 2>/dev/null || true)
    echo "$u: ${state:-inactive}"
    [ "$CMD" = stop ] && [ "$state" = active ] && systemctl --user stop "$u" && echo "  stopped"
  done
  for log in runs/v11-seeds-overnight.log runs/v11-smoke.log; do
    [ -f "$log" ] && { echo "--- $log (steps):"; grep -a '^=== 20' "$log" | tail -12; }
  done
  exit 0
fi

for u in v11-seed-sweep v11-seed-sweep-smoke; do
  if systemctl --user is-active --quiet "$u"; then
    echo "$u is already running — '$0 status' to watch, '$0 stop' to stop"; exit 1
  fi
done
systemctl --user reset-failed "$UNIT" 2>/dev/null || true

# The machine must not suspend under the run.
if command -v gsettings >/dev/null 2>&1; then
  ac=$(gsettings get org.gnome.settings-daemon.plugins.power sleep-inactive-ac-type 2>/dev/null || true)
  [ "$ac" = "'nothing'" ] || echo "WARNING: automatic suspend on AC is '$ac' — disable it for tonight (Settings > Power)"
fi

ENVS=(--setenv=PATH="$PATH" --setenv=HOME="$HOME")
[ -n "${V11_DATE:-}" ] && ENVS+=(--setenv=V11_DATE="$V11_DATE")
systemd-run --user --quiet --collect --unit="$UNIT" --working-directory="$PWD" \
  "${ENVS[@]}" bash scripts/run_v11_seeds.sh "${ARGS[@]}"

echo "started $UNIT (detached)"
echo "  watch:  tail -f $LOG        # one START/DONE line per step"
echo "  status: $0 status"
echo "  stop:   $0 stop             # resumable"
