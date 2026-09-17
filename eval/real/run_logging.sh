# Sourced by the real launchers. Keep one tee across the guidance -> obs66 exec.
record_requested=0
for argument in "$@"; do
    case "${argument}" in
        --record) record_requested=1 ;;
        *) echo "Unknown argument: ${argument}; use --record to save run logs" >&2; exit 2 ;;
    esac
done
# Consume launcher flags; the active marker carries recording through exec.
set --
if [[ "${_DEX_REAL_LOG_ACTIVE:-0}" != "1" && "${record_requested}" != "1" ]]; then
    # A configured path alone must not enable recording for this run.
    unset REAL_DEBUG_LOG
    echo "[real] recording disabled (use --record to enable)"
    return 0
fi

if [[ "${_DEX_REAL_LOG_ACTIVE:-0}" != "1" ]]; then
    RUN_LOG="${RUN_LOG:-${SCRIPT_DIR}/real/latest_run.txt}"
    DEBUG_RECORDING="${DEBUG_RECORDING:-1}"
    [[ "${DEBUG_RECORDING}" =~ ^[01]$ ]] || { echo "DEBUG_RECORDING must be 0 or 1" >&2; exit 2; }
    mkdir -p -- "$(dirname -- "${RUN_LOG}")"
    # Preserve every run, including paired structured records. Rotate only once
    # when the guidance launcher execs the shared obs66 launcher.
    log_max_age=0
    for log_archive in "${RUN_LOG%.txt}".*.txt "${RUN_LOG%.txt}".*.jsonl; do
        [[ -f "${log_archive}" ]] || continue
        log_suffix="${log_archive#"${RUN_LOG%.txt}."}"
        log_suffix="${log_suffix%.*}"
        if [[ "${log_suffix}" =~ ^[1-9][0-9]*$ ]] && (( log_suffix > log_max_age )); then
            log_max_age=${log_suffix}
        fi
    done
    for ((log_age = log_max_age + 1; log_age >= 1; log_age--)); do
        if ((log_age == 1)); then
            log_source="${RUN_LOG}"
        else
            log_source="${RUN_LOG%.txt}.$((log_age - 1)).txt"
        fi
        if [[ -f "${log_source}" ]]; then
            mv -f -- "${log_source}" "${RUN_LOG%.txt}.${log_age}.txt"
        fi
        debug_source="${log_source%.txt}.jsonl"
        debug_destination="${RUN_LOG%.txt}.${log_age}.jsonl"
        if [[ -f "${debug_source}" ]]; then
            mv -f -- "${debug_source}" "${debug_destination}"
        elif [[ -f "${debug_destination}" ]]; then
            # An older TXT-only run must not inherit a different run's sidecar.
            rm -- "${debug_destination}"
        fi
    done
    # Fail synchronously if the log cannot be opened, before hardware startup.
    : > "${RUN_LOG}"
    REAL_DEBUG_LOG="${RUN_LOG%.txt}.jsonl"
    if [[ "${DEBUG_RECORDING}" == "0" ]]; then
        REAL_DEBUG_LOG=""
    else
        : > "${REAL_DEBUG_LOG}"
    fi
    export REAL_DEBUG_LOG
    export RUN_LOG _DEX_REAL_LOG_ACTIVE=1
    # Ignore Ctrl-C in tee so Python's final cleanup output is still recorded.
    exec > >(tee -i -a -- "${RUN_LOG}") 2>&1
    echo "[real] log: ${RUN_LOG} (keeping all runs; no retention limit)"
    echo "[real] structured debug: ${REAL_DEBUG_LOG:-disabled}"
    echo "[real] started: $(date -Is)"
    if [[ -n "${EXPERIMENT_DESCRIPTION:-}" ]]; then
        echo "[experiment] ${EXPERIMENT_DESCRIPTION}"
    fi
fi
