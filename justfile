set shell := ["bash", "-euo", "pipefail", "-c"]
set positional-arguments

export KNEE_PROJECT_ROOT := justfile_directory()

[private]
default:
    @rtk proxy uv run --locked just --list

setup:
    rtk proxy uv sync --locked

prepare-codecs output="outputs/dependencies/kaggle-v170" python_version="3.12" platform="manylinux2014_x86_64":
    rtk proxy uvx --from pip==25.1.1 pip download --no-deps --only-binary=:all: --python-version "$2" --platform "$3" --dest "$1" pylibjpeg==2.1.0 pylibjpeg-libjpeg==2.4.0 pylibjpeg-openjpeg==2.5.0

prepare-folds config="configs/baseline.yaml":
    rtk proxy uv run --locked python -m scripts.tools.validation prepare --config "$1"

train run_id config="configs/baseline.yaml" folds="0":
    rtk proxy uv run --locked python -m scripts.pipeline.train "$2" --folds "$3" --run-id "$1"

resume run_id checkpoint config fold="0":
    rtk proxy uv run --locked python -m scripts.pipeline.train "$3" --folds "$4" --run-id "$1" --resume "$2"

validate config checkpoint output fold="0" image_multiplier="" batch_size="":
    args=(); \
    if [[ -n "$5" ]]; then args+=(--image-multiplier "$5"); fi; \
    if [[ -n "$6" ]]; then args+=(--batch-size "$6"); fi; \
    rtk proxy uv run --locked python -m scripts.pipeline.predict "$1" --checkpoint "$2" --output "$3" --split validation --fold "$4" "${args[@]}"

export checkpoint output image_multiplier="2" batch_size="2" wheel_dir="outputs/dependencies/kaggle-v170" target_python="3.12" target_platform="manylinux_x86_64":
    rtk proxy uv run --locked python -m scripts.pipeline.export --checkpoint "$1" --output "$2" --image-multiplier "$3" --inference-batch-size "$4" --wheel-dir "$5" --target-python "$6" --target-platform "$7"

infer bundle data_root="data" output="outputs/submission.csv" device="auto" precision="auto" batch_size="" num_workers="2":
    bundle_path="$(rtk proxy realpath -e -- "$1")"; \
    data_path="$(rtk proxy realpath -e -- "$2")"; \
    output_path="$(rtk proxy realpath -m -- "$3")"; \
    args=(); \
    if [[ -n "$6" ]]; then args+=(--batch-size "$6"); fi; \
    cd "$bundle_path"; \
    rtk proxy env PYTHONPATH="$bundle_path/code" "$KNEE_PROJECT_ROOT/.venv/bin/python" -m scripts.pipeline.predict "$bundle_path/config.json" --bundle "$bundle_path" --data-root "$data_path" --output "$output_path" --device "$4" --precision "$5" --num-workers "$7" "${args[@]}"

submit bundle dataset notebook record message="Fold 0 inference bundle":
    rtk proxy uv run --locked python -m scripts.tools.submission_job start --bundle "$1" --dataset "$2" --notebook "$3" --record "$4" --message "$5"

submit-foreground bundle dataset notebook record message="Fold 0 inference bundle":
    rtk proxy uv run --locked python -m scripts.pipeline.submit --bundle "$1" --dataset "$2" --notebook "$3" --record "$4" --message "$5"

submit-status record:
    rtk proxy uv run --locked python -m scripts.tools.submission_job status --record "$1"

submit-logs record lines="60":
    rtk proxy uv run --locked python -m scripts.tools.submission_job logs --record "$1" --lines "$2"

submit-follow record lines="60":
    rtk proxy uv run --locked python -m scripts.tools.submission_job logs --record "$1" --lines "$2" --follow

submissions:
    rtk proxy uv run --locked kaggle competitions submissions rsna-knee-abnormality-detection -v
