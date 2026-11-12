# shellcheck shell=bash
#
# Helpers for scripts/skyrl/launch_skyrl.sh, which sources this file: parse the arguments, read .env,
# pick the GPUs, start the retrieval server and Ray, and kill every process of the run when the
# launcher exits. The launcher calls launch_setup, start_retriever and start_ray, in that order.

#######################################
# Parse the launcher's arguments, read .env and pick the GPUs.
# Globals:
#   Sets REPO, MODE, ENCODER, DATA_DIR, TRAINER_ARGS, RETRIEVER_ARGS, RUN_ID, RAY_TMPDIR,
#   NUM_GPUS (for Ray: CUDA_VISIBLE_DEVICES, else the whole node) and TRAINER_GPUS (for the
#   trainer; empty for an eval without CUDA_VISIBLE_DEVICES, so eval.yaml's single GPU applies).
# Arguments:
#   <train|eval> <e5|qwen3-4b> <data_dir> [args]...
#######################################
launch_setup() {
    if [ $# -lt 3 ]; then
        usage
        exit 1
    fi
    MODE=$1
    ENCODER=$2
    DATA_DIR=${3%/}
    shift 3
    case $MODE in
        train | eval) ;;
        *) echo "Unknown mode '$MODE' (expected train or eval)"; usage; exit 1 ;;
    esac

    REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
    cd "$REPO"
    if [ -f .env ]; then
        set -a
        source .env
        set +a
    fi

    # Unique per launch (short: Ray caps socket path length)
    RUN_ID=${SLURM_JOB_ID:-local}-$$
    # Ray workers run `uv run` from their own working dir
    export UV_PROJECT_ENVIRONMENT=${UV_PROJECT_ENVIRONMENT:-$REPO/.venv}
    # Concurrent runs on a node must not share Ray state
    export RAY_TMPDIR=/tmp/ray-$RUN_ID

    RETRIEVER_ARGS=()
    TRAINER_ARGS=()
    local arg
    for arg in "$@"; do
        case $arg in
            retriever.*) RETRIEVER_ARGS+=("${arg#retriever.}") ;;
            *) TRAINER_ARGS+=("$arg") ;;
        esac
    done

    TRAINER_GPUS=""
    if [ -z "${CUDA_VISIBLE_DEVICES:-}" ]; then
        NUM_GPUS=$(nvidia-smi --query-gpu=count --format=csv,noheader,nounits | head -n 1)
        CUDA_VISIBLE_DEVICES=$(seq -s, 0 $((NUM_GPUS - 1)))
        export CUDA_VISIBLE_DEVICES
        if [ "$MODE" = train ]; then
            TRAINER_GPUS=$NUM_GPUS
        fi
    else
        NUM_GPUS=$(echo "$CUDA_VISIBLE_DEVICES" | tr ',' '\n' | wc -l)
        TRAINER_GPUS=$NUM_GPUS
    fi

    mkdir -p logs "$RAY_TMPDIR"
    RETRIEVER_LOG="logs/retriever-$RUN_ID.log"
    PORT_FILE=$(mktemp)
    trap cleanup EXIT
    trap 'exit 130' INT
    trap 'exit 143' TERM

    echo "-------------------------------"
    echo "PhantomEnvs $MODE with $(basename "$0") (run $RUN_ID)"
    echo "  data dir:   $DATA_DIR"
    echo "  retriever:  $ENCODER ${RETRIEVER_ARGS[*]:-} (log: $RETRIEVER_LOG)"
    echo "  GPUs:       $CUDA_VISIBLE_DEVICES (trainer: ${TRAINER_GPUS:-as in the eval config})"
    echo "  args:       ${TRAINER_ARGS[*]:-}"
    echo "-------------------------------"
}

#######################################
# Signal every process of this run (they share its RAY_TMPDIR).
# Arguments:
#   The signal, e.g. TERM.
#######################################
signal_run() {
    local pid
    for pid in $(pgrep -u "$USER"); do
        if [ "$pid" != "$$" ] && { tr '\0' '\n' < "/proc/$pid/environ"; } 2>/dev/null | grep -qx "RAY_TMPDIR=$RAY_TMPDIR"; then
            kill "-$1" "$pid" 2>/dev/null || true
        fi
    done
}

#######################################
# Kill the run's processes and remove its temp files (the EXIT trap).
#######################################
cleanup() {
    signal_run TERM
    sleep 5
    signal_run KILL  # Ray's GCS and monitor can outlive SIGTERM
    rm -rf "$RAY_TMPDIR" "$PORT_FILE"
}

#######################################
# Start the retrieval server over DATA_DIR and wait until it is healthy.
# Globals:
#   Sets SEARCH_URL.
#######################################
start_retriever() {
    # FAISS from the faiss-cpu wheel, or with retriever.faiss_gpu=true (Search-R1's 21M documents)
    # from the conda env in FAISS_GPU_ENV
    local retriever_env=() retriever_uv=(--with faiss-cpu==1.12.0) arg
    for arg in "${RETRIEVER_ARGS[@]}"; do
        if [ "${arg,,}" = "faiss_gpu=true" ]; then
            if [ -z "${FAISS_GPU_ENV:-}" ]; then
                echo "Error: retriever.faiss_gpu=true needs FAISS_GPU_ENV in .env (see the README)"
                exit 1
            fi
            retriever_env=(env "PYTHONPATH=$FAISS_GPU_ENV/lib/python3.12/site-packages${PYTHONPATH:+:$PYTHONPATH}"
                "LD_LIBRARY_PATH=$FAISS_GPU_ENV/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}")
            retriever_uv=()
        fi
    done

    # port=0: the server picks a free port and writes it to PORT_FILE
    "${retriever_env[@]}" uv run --no-sync "${retriever_uv[@]}" python -m phantom_envs.retriever.retrieval_server \
        --encoder "$ENCODER" --data_dir "$DATA_DIR" port=0 "port_file=$PORT_FILE" "${RETRIEVER_ARGS[@]}" \
        > "$RETRIEVER_LOG" 2>&1 &
    local pid=$! elapsed

    # A large index takes minutes to load (Search-R1's: ~64 GB)
    echo "Waiting for the retrieval server ..."
    for elapsed in $(seq 0 2 300); do
        if ! kill -0 "$pid" 2>/dev/null; then
            echo "Error: the retrieval server exited; see $RETRIEVER_LOG"
            exit 1
        fi
        if [ -s "$PORT_FILE" ] && curl -sf "http://127.0.0.1:$(cat "$PORT_FILE")/health" > /dev/null; then
            break
        fi
        if [ "$elapsed" -ge 300 ]; then
            echo "Error: the retrieval server was not ready after ${elapsed}s; see $RETRIEVER_LOG"
            exit 1
        fi
        sleep 2
    done
    SEARCH_URL="http://127.0.0.1:$(cat "$PORT_FILE")/retrieve"
    echo "Retrieval server ready at $SEARCH_URL"
}

#######################################
# Pick a random TCP port that nothing listens on.
# Arguments:
#   The lowest and highest allowed port.
# Outputs:
#   Writes the port to stdout.
#######################################
free_port() {
    local port
    for _ in $(seq 100); do
        port=$(shuf -i "$1-$2" -n 1)
        if [ -z "$(ss -Htln "sport = :$port")" ]; then
            echo "$port"
            return 0
        fi
    done
    echo "Error: no free port in $1-$2" >&2
    return 1
}

#######################################
# Start this run's Ray head on a free port.
# Globals:
#   Sets RAY_ADDRESS.
# Arguments:
#   The lowest and highest allowed port, then the ray command (e.g. uv run --no-sync ray).
#######################################
start_ray() {
    local port
    port=$(free_port "$1" "$2")
    shift 2
    # Workers take kernel-assigned ports: Ray's default range is the same for every cluster on a node
    "$@" start --head --port="$port" --min-worker-port=0 --max-worker-port=0 --num-cpus=16 --num-gpus="$NUM_GPUS"
    export RAY_ADDRESS=127.0.0.1:$port
}
