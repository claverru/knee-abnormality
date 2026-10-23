set shell := ["bash", "-euo", "pipefail", "-c"]
set positional-arguments

export KNEE_PROJECT_ROOT := justfile_directory()

python := "uv run --locked python -m"

[private]
default:
    @uv run --locked just --list

setup:
    uv sync --locked

prepare-codecs *args:
    {{python}} scripts.tools.prepare_codecs "$@"

config *args:
    {{python}} knee_abnormality.config "$@"

validation *args:
    {{python}} scripts.tools.validation "$@"

train *args:
    {{python}} scripts.pipeline.train "$@"

predict *args:
    {{python}} scripts.pipeline.predict "$@"

export *args:
    {{python}} scripts.pipeline.export "$@"

infer bundle data_root="data" output="outputs/submission/submission.csv" *args:
    bundle_path="$(realpath -e -- "$1")"; \
    data_path="$(realpath -e -- "$2")"; \
    output_path="$(realpath -m -- "$3")"; \
    cd "$bundle_path"; \
    env PYTHONPATH="$bundle_path/code" "$KNEE_PROJECT_ROOT/.venv/bin/python" -m scripts.pipeline.predict "$bundle_path/config.json" --bundle "$bundle_path" --data-root "$data_path" --output "$output_path" "${@:4}"

submit *args:
    {{python}} scripts.pipeline.submit "$@"

submission-job *args:
    {{python}} scripts.tools.submission_job "$@"

submissions:
    uv run --locked kaggle competitions submissions rsna-knee-abnormality-detection -v

autoresearch *args:
    {{python}} scripts.tools.autoresearch "$@"

autoresearch-watch *args:
    {{python}} scripts.tools.autoresearch_watch "$@"
