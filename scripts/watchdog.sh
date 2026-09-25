#!/usr/bin/env bash
# Unattended supervisor for the overnight comparison.
#
# Every INTERVAL seconds: snapshot each run's progress, and relaunch any run whose
# trainer has died prematurely. "Prematurely" means the process is gone but the run
# never reached the wall-clock deadline, which is the only legitimate way to stop.
#
# Restarts are capped per run so a reproducible crash cannot loop all night.
set -uo pipefail
cd "$(dirname "$0")/.."

INTERVAL="${INTERVAL:-600}"
MAX_RESTARTS="${MAX_RESTARTS:-2}"
DEADLINE_EPOCH="${DEADLINE_EPOCH:-0}"   # stop supervising after this time
LOG=logs/watchdog.log

# Read the launcher's manifest rather than hardcoding run names, so this supervises
# whatever set of runs is currently in flight.
MANIFEST="${MANIFEST:-logs/runs.manifest}"
NAMES=(); CONFIGS=(); DEVICES=(); PORTS=(); OVERRIDES=()
while read -r name config device port rest; do
    [ -z "${name:-}" ] && continue
    NAMES+=("$name"); CONFIGS+=("$config"); DEVICES+=("$device"); PORTS+=("$port")
    OVERRIDES+=("${rest:-}")
done < "$MANIFEST"
declare -A RESTARTS

log () { echo "[$(date -u +%H:%M:%S)] $*" >> "$LOG"; }

log "watchdog started, interval ${INTERVAL}s"

while true; do
    if [ "$DEADLINE_EPOCH" -gt 0 ] && [ "$(date +%s)" -gt "$DEADLINE_EPOCH" ]; then
        log "past deadline, watchdog exiting"
        break
    fi

    alive_count=0
    for i in "${!NAMES[@]}"; do
        name="${NAMES[$i]}"
        steps=$(wc -l < "runs/${name}/metrics.jsonl" 2>/dev/null || echo 0)

        if pgrep -f "output_dir=runs/${name}" > /dev/null 2>&1; then
            alive_count=$((alive_count + 1))
            log "  ${name}: alive, ${steps} records"
            continue
        fi

        # Finished cleanly if the trainer reported hitting the time limit.
        if grep -q "wall-clock limit" "logs/${name}_train.log" 2>/dev/null; then
            log "  ${name}: finished at deadline, ${steps} records"
            continue
        fi

        n="${RESTARTS[$name]:-0}"
        if [ "$n" -ge "$MAX_RESTARTS" ]; then
            log "  ${name}: DEAD, restart budget exhausted (${steps} records)"
            continue
        fi

        remaining_h=1
        if [ "$DEADLINE_EPOCH" -gt 0 ]; then
            remaining_h=$(( (DEADLINE_EPOCH - $(date +%s)) / 3600 ))
        fi
        if [ "$remaining_h" -lt 1 ]; then
            log "  ${name}: DEAD but under an hour left, not restarting"
            continue
        fi

        RESTARTS[$name]=$((n + 1))
        log "  ${name}: DEAD at ${steps} records, restart #${RESTARTS[$name]} for ${remaining_h}h"
        pkill -f "port ${PORTS[$i]}" 2>/dev/null || true
        sleep 20
        MAX_LEN="${MAX_LEN:-7168}" MAX_NUM_SEQS="${MAX_NUM_SEQS:-256}" \
        MODEL="${MODEL:-Qwen/Qwen3.5-9B-Base}" \
        setsid nohup ./scripts/launch_run.sh "$name" "${CONFIGS[$i]}" "${DEVICES[$i]}" "${PORTS[$i]}" \
            "$remaining_h" total_steps=1000 eval.every_steps=20 eval.avg_at_k=8 \
            save_every_steps=1000000 ${OVERRIDES[$i]} \
            >> "logs/${name}_launch.log" 2>&1 < /dev/null &
    done

    log "${alive_count}/8 runs alive"
    sleep "$INTERVAL"
done
