#!/usr/bin/env bash

# print warning/error
warn() {
    echo "$*" >&2
}

# kill the script with an error message
die() {
    warn "$*"
    exit 1
}

# global variable to enable verbosity
__VERBOSE=0

print_usage() {
    warn "A helper script for CP interactions."
    warn
    warn "Usage: ${SCRIPT_FILE} [OPTIONS] build|run_pov|custom|make-cpsrc-prepare"
    warn
    warn "OPTIONS:"
    warn "  -h    Print this help menu"
    warn "  -v    Turn on verbose debug messages"
    warn
    warn "Subcommands:"
    warn "  build [<patch_file> <source>]       Build the CP"
    warn "  run_pov <blob_file> <harness_name>  Run the binary data blob against specified harness"
    warn "  custom <arbitrary cmd ...>          Run an arbitrary command in the docker container"
    warn "  make-cpsrc-prepare                  Prepare docker container"
    die
}

# echo/log message if verbosity is on
verbose() {
    [[ ${__VERBOSE} -gt 0 ]] && echo "<DEBUG> $*"
}

# prepare docker container once
make-cpsrc-prepare() {
    verbose "start build process"

    # directly install sudo, bear and its dependencies in Dockerfile
    # use awk with tempfile to insert AIxCC-style dependencie installation
    verbose "modify Dockerfile to install bear"
    # Check if the Dockerfile already contains the command to install bear
    if ! grep -q 'bear' "$DOCKERFILE_PATH"; then
        tempfile=$(mktemp)
        # Insert commands after the FROM line using awk for more control
        awk '
        /^FROM .*/ {
            print $0
            print "RUN set -eux; \\"
            print "    DEBIAN_FRONTEND=noninteractive apt-get update && \\"
            print "    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \\"
            print "        gettext-base \\"
            print "        gosu \\"
            print "        sudo \\"
            print "        bear \\"
            print "        && \\"
            print "    apt-get autoremove -y && \\"
            print "    rm -rf /var/lib/apt/lists/* && \\"
            print "    gosu nobody true"
            next
        }
        { print $0 }
        ' "$DOCKERFILE_PATH" > "$tempfile"
        # Replace the original Dockerfile with the modified one
        mv "$tempfile" "$DOCKERFILE_PATH"
    fi

    # build container
    verbose "building docker container"
    docker build --no-cache -t $IMAGE_NAME --file $DOCKERFILE_PATH $BUILD_CONTEXT
}


# "build" command handler -> compile src code
# splittung up docker build and complation process to keep persistence
build(){
    # mount out and work and start build.sh
    # src cannot be mounted yet, it first has to be copied or else it would be overwritten
    # we need to overwrite the entrypoint in order to use bear

    if [[ ! -f "$STATE_FILE" ]]; then
        verbose "compiling project for the first time"
        docker run --rm --privileged --shm-size=2g --platform linux/amd64 -e FUZZING_ENGINE=libfuzzer -e SANITIZER=address -e ARCHITECTURE=x86_64 -e PROJECT_NAME="NAME_TO_BE_REPLACED" -e HELPER=True -e FUZZING_LANGUAGE=c++ --env-file "$DOCKER_ENV_FILE" -v "$OUT":/out -v "$WORK":/work -t -v "$INIT_SCRIPT":/init_CRS.sh --entrypoint /init_CRS.sh "$IMAGE_NAME"

        verbose "copying SRC to expose it"
        docker run --rm "$IMAGE_NAME" sh -c 'tar cf - /src' | tar xf - -C "$PROJECT_DIR"

        echo "called" > "$STATE_FILE"
    else
        verbose "compiling project"
        docker run --rm --privileged --shm-size=2g --platform linux/amd64 -e FUZZING_ENGINE=libfuzzer -e SANITIZER=address -e ARCHITECTURE=x86_64 -e PROJECT_NAME="NAME_TO_BE_REPLACED" -e HELPER=True -e FUZZING_LANGUAGE=c++ --env-file "$DOCKER_ENV_FILE" -v "$OUT":/out -v "$WORK":/work -v "$SRC":/src -v "$INIT_SCRIPT":/init_CRS.sh --entrypoint /init_CRS.sh -t "$IMAGE_NAME"
    fi

    verbose "build process completed"
}

# "run_pov" command handler
run_pov() {
    shift

    # check validity of arguments for run_pov command
    BLOB_FILE=$1
    HARNESS_NAME=$2
    [[ -n "${BLOB_FILE}" ]] || die "Missing blob file argument"
    [[ -f "${BLOB_FILE}" ]] || die "Blob file not found: ${BLOB_FILE}"
    [[ -n "${HARNESS_NAME}" ]] || die "Missing harness argument"

    # Copy blob file to work directory
    cp "$BLOB_FILE" "${WORK}/tmp_blob" || die "No blob file found!"

    # execute 100 times for flaky and non-deterministic crashes
    verbose "Testing $(realpath "$BLOB_FILE") with harness $HARNESS_NAME"
    docker run --rm --privileged --shm-size=2g --platform linux/amd64 --rm -e HELPER=True -e ARCHITECTURE=x86_64 --env-file "$DOCKER_ENV_FILE" -v "$OUT":/out -v "$WORK":/work -v "$SRC":/src --mount type=bind,source="$(realpath "$BLOB_FILE")",target=/testcase -t gcr.io/oss-fuzz-base/base-runner reproduce "$HARNESS_NAME" -runs=100
}

# choose your own adventure with an arbitrary docker run command invocation
custom() {
    shift
    verbose "Executing in /bin/bash: $*"
    docker run --rm --privileged --shm-size=2g --platform linux/amd64 --rm --env-file "$DOCKER_ENV_FILE" -v "$OUT":/out -v "$WORK":/work -v "$SRC":/src -t "$IMAGE_NAME" /bin/bash -c "$*"
}


# array of top-level command handlers
declare -A MAIN_COMMANDS=(
    [help]=print_usage
    [build]=build
    [run_pov]=run_pov
    [custom]=custom
    [make-cpsrc-prepare]=make-cpsrc-prepare
)

# look for needed commands/dependencies
REQUIRED_COMMANDS="git docker"
for c in ${REQUIRED_COMMANDS}; do
    command -v "${c}" >/dev/null || warn "WARNING: needed executable (${c}) not found in PATH"
done

# these directories are shared with the docker container
# cannot use oss-fuzz/build because it would override repos that the Dockerfile pulls and build.sh wants to use later
PROJECT_DIR=$(realpath .)
WORK="${PROJECT_DIR}/work"
OUT="${PROJECT_DIR}/out/"
SRC="${PROJECT_DIR}/src/"
INIT_SCRIPT="${PROJECT_DIR}/init_CRS.sh"

# file to track wehther project was already built once
# if it was already built, we can mount src
STATE_FILE="${PROJECT_DIR}/.was_built_tracker"

# create dir if does not exist already
mkdir -p "$WORK" "$OUT" "$SRC"

IMAGE_NAME="DOCKER_IMAGE_TO_BE_REPLACED"
DOCKERFILE_PATH="Dockerfile"
BUILD_CONTEXT=$PROJECT_DIR
DOCKER_ENV_FILE="$PROJECT_DIR/.env.docker"
DOCKER_ENV_VARS=""

# parse entry/global options
while getopts ":hxv" opt; do
    case ${opt} in
        h) print_usage;; # help
        x) __RETURN_DOCKER_EXIT_CODE=1;; # exit script with docker run exit code
        v) __VERBOSE=1; verbose "invoked as: ${SCRIPT_FILE} $*";; # turn on verbosity
        ?) warn "Invalid option: -${OPTARG}"; print_usage;; # error, print usage
    esac
done

shift "$((OPTIND-1))"

# call subcommand function from declared array of handlers (default to help)
"${MAIN_COMMANDS[${1:-help}]:-${MAIN_COMMANDS[help]}}" "$@"
