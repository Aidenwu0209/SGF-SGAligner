#!/usr/bin/dash
# Hash-bound active fixed4 child.  Production is entered only through the
# code-pinned wrapper and a signed, hash-bound execution manifest.
set -eu
umask 077
stage= task_id= task= task_sha256= preflight= preflight_sha256=
authorization= authorization_sha256= task_manifest= task_manifest_sha256=
runner_output= fixture_input= fixture_input_sha256=
repo= output_root= execution_manifest= execution_manifest_sha256=
production_manifest_commit= production_manifest_commit_sha256=
production_python= production_python_sha256= production_wrapper=
production_wrapper_sha256= runner_source_sha256=
while [ "$#" -gt 0 ]; do
    case "$1" in
        --stage) stage=$2; shift 2 ;;
        --task-id) task_id=$2; shift 2 ;;
        --task) task=$2; shift 2 ;;
        --task-sha256) task_sha256=$2; shift 2 ;;
        --preflight) preflight=$2; shift 2 ;;
        --preflight-sha256) preflight_sha256=$2; shift 2 ;;
        --authorization) authorization=$2; shift 2 ;;
        --authorization-sha256) authorization_sha256=$2; shift 2 ;;
        --task-manifest) task_manifest=$2; shift 2 ;;
        --task-manifest-sha256) task_manifest_sha256=$2; shift 2 ;;
        --runner-output) runner_output=$2; shift 2 ;;
        --fixture-input) fixture_input=$2; shift 2 ;;
        --fixture-input-sha256) fixture_input_sha256=$2; shift 2 ;;
        --repo) repo=$2; shift 2 ;;
        --output-root) output_root=$2; shift 2 ;;
        --execution-manifest) execution_manifest=$2; shift 2 ;;
        --execution-manifest-sha256) execution_manifest_sha256=$2; shift 2 ;;
        --production-manifest-commit) production_manifest_commit=$2; shift 2 ;;
        --production-manifest-commit-sha256) production_manifest_commit_sha256=$2; shift 2 ;;
        --production-python) production_python=$2; shift 2 ;;
        --production-python-sha256) production_python_sha256=$2; shift 2 ;;
        --production-wrapper) production_wrapper=$2; shift 2 ;;
        --production-wrapper-sha256) production_wrapper_sha256=$2; shift 2 ;;
        --runner-source-sha256) runner_source_sha256=$2; shift 2 ;;
        *) exit 64 ;;
    esac
done
case "$stage" in
    colorpcr_direction|bidirectional_multi_solver_pilot|v16_pair_hypothesis_cluster|fixed4_aggregate|contract_fixture) ;;
    *) exit 64 ;;
esac
case "$task_id" in *[!A-Za-z0-9._-]*|'') exit 64 ;; esac
verify_file() {
    expected=$1 path=$2
    [ -n "$expected" ] && [ -n "$path" ]
    printf '%s  %s\n' "$expected" "$path" | /usr/bin/sha256sum --check --status
}
verify_file "$task_sha256" "$task"
verify_file "$preflight_sha256" "$preflight"
verify_file "$authorization_sha256" "$authorization"
verify_file "$task_manifest_sha256" "$task_manifest"
[ -n "$runner_output" ] && [ ! -e "$runner_output" ]
set -C
if [ "$stage" = contract_fixture ]; then
    verify_file "$fixture_input_sha256" "$fixture_input"
    /usr/bin/cat "$fixture_input" > "$runner_output"
    exit 0
fi
if [ -z "$execution_manifest" ]; then
    printf '%s\n' "{\"schema\":\"v16-b716-fixed4-active-runner-refusal-v1\",\"runner_mode\":\"active\",\"stage\":\"$stage\",\"status\":\"adapter_unavailable\",\"failure_type\":\"PRODUCTION_STAGE_ADAPTER_UNAVAILABLE\",\"task_id\":\"$task_id\",\"task_sha256\":\"$task_sha256\",\"operational_result_emitted\":false}" > "$runner_output"
    exit 69
fi
verify_file "$execution_manifest_sha256" "$execution_manifest"
verify_file "$production_manifest_commit_sha256" "$production_manifest_commit"
verify_file "$production_python_sha256" "$production_python"
verify_file "$production_wrapper_sha256" "$production_wrapper"
[ -n "$repo" ] && [ -n "$output_root" ] && [ -n "$runner_source_sha256" ]
unset PYTHONPATH PYTHONHOME PYTHONSTARTUP PYTHONUSERBASE
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export PYTHONPYCACHEPREFIX=/proc/v16-b716-fixed4-no-pyc
export PYTHONHASHSEED=0 CUDA_CACHE_DISABLE=1
exec "$production_python" -I -S -s -B \
    -X pycache_prefix=/proc/v16-b716-fixed4-no-pyc \
    "$production_wrapper" \
    --repo "$repo" --task "$task" --task-sha256 "$task_sha256" \
    --execution-manifest "$execution_manifest" \
    --execution-manifest-sha256 "$execution_manifest_sha256" \
    --production-manifest-commit "$production_manifest_commit" \
    --production-manifest-commit-sha256 "$production_manifest_commit_sha256" \
    --output-root "$output_root" --runner-source-sha256 "$runner_source_sha256" \
    --output "$runner_output"
