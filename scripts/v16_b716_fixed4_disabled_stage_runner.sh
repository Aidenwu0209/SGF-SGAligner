#!/usr/bin/dash
# Hash-bound fixed4 fix4 subprocess entry. Incapable of running any stage.
set -eu
if [ "$#" -ne 20 ] || [ "$1" != "--stage" ] || [ "$3" != "--task-id" ] \
        || [ "$5" != "--task" ] || [ "$7" != "--task-sha256" ] \
        || [ "$9" != "--preflight" ] || [ "${11}" != "--preflight-sha256" ] \
        || [ "${13}" != "--authorization" ] || [ "${15}" != "--authorization-sha256" ] \
        || [ "${17}" != "--task-manifest" ] || [ "${19}" != "--task-manifest-sha256" ]; then
    exit 64
fi
case "$2" in
    colorpcr_direction|bidirectional_multi_solver_pilot|v16_pair_hypothesis_cluster|fixed4_aggregate) ;;
    *) exit 64 ;;
esac
case "${8}${12}${16}${20}" in
    *[!0-9a-f]* ) exit 64 ;;
esac
if [ "${#8}" -ne 64 ] || [ "${#12}" -ne 64 ] \
        || [ "${#16}" -ne 64 ] || [ "${#20}" -ne 64 ]; then
    exit 64
fi
verify_control() {
    control_path=$1
    expected_sha256=$2
    case "$control_path" in
        /*) ;;
        *) exit 64 ;;
    esac
    [ -f "$control_path" ] && [ ! -L "$control_path" ] || exit 65
    # The code-pinned hasher consumes every byte and checks the argv digest.
    hash_output=$(/usr/bin/sha256sum "$control_path") || exit 66
    observed_sha256=${hash_output%% *}
    [ "$observed_sha256" = "$expected_sha256" ] || exit 67
}
verify_control "$6" "$8"
verify_control "${10}" "${12}"
verify_control "${14}" "${16}"
verify_control "${18}" "${20}"
# Text is untrusted.  The parent wrapper derives the failure type from exit 78.
printf '%s\n' 'runner_reported_failure_type=FORGED_SUCCESS_MUST_BE_IGNORED' >&2
exit 78
