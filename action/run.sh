#!/usr/bin/env bash
# The prompteval GitHub Action (action.yml at the repository root runs this).
#
# compare mode (default): runs the PR base's version of INPUT_PROMPT and the
#   head version on the same eval, compares them, and posts the report.
# sweep mode (INPUT_MODELS set): runs INPUT_PROMPT on each model and posts
#   which model is cheapest without losing quality.
#
# The report is written to the job summary and, on a pull request, to ONE
# comment that is updated on every push (found by a hidden marker). The step
# fails when the --fail-on gate does, but only after the report is posted.
set -euo pipefail

prompt="${INPUT_PROMPT:?the 'prompt' input is required}"
eval_file="${INPUT_EVAL_FILE:-evals/eval.py}"
work="${RUNNER_TEMP:-/tmp}/prompteval-action"
mkdir -p "$work"
report="$work/report.md"
status=0

if [[ -n "${INPUT_MODELS:-}" ]]; then
  args=(sweep --prompt "$prompt" --models "$INPUT_MODELS" --eval-file "$eval_file" --markdown "$report")
  [[ -n "${INPUT_MARGIN:-}" ]] && args+=(--margin "$INPUT_MARGIN")
  prompteval "${args[@]}" || status=$?
else
  base="${INPUT_BASELINE_REF:-}"
  if [[ -z "$base" ]]; then
    base="${PR_BASE_SHA:-origin/${DEFAULT_BRANCH:-main}}"
  fi
  git fetch --quiet --depth=1 origin "$base" 2>/dev/null || true
  if ! git show "$base:$prompt" > "$work/baseline.txt" 2>/dev/null; then
    printf '### prompteval\n\n`%s` does not exist at the base (`%s`), so there is nothing to compare against yet. It will be compared on the next change.\n' \
      "$prompt" "$base" > "$report"
  elif cmp -s "$work/baseline.txt" "$prompt"; then
    printf '### prompteval\n\n`%s` is unchanged from the base, so no runs were made (and nothing was spent).\n' \
      "$prompt" > "$report"
  else
    prompteval run --prompt "$work/baseline.txt" --tag baseline --eval-file "$eval_file"
    prompteval run --prompt "$prompt" --tag candidate --eval-file "$eval_file"
    args=(compare baseline candidate --markdown "$report")
    [[ -n "${INPUT_FAIL_ON:-}" ]] && args+=(--fail-on "$INPUT_FAIL_ON")
    [[ -n "${INPUT_MARGIN:-}" ]] && args+=(--margin "$INPUT_MARGIN")
    prompteval "${args[@]}" || status=$?
  fi
fi

if [[ ! -s "$report" ]]; then
  printf '### prompteval\n\nThe run failed before producing a report; see the job log.\n' > "$report"
  [[ "$status" -eq 0 ]] && status=1
fi

if [[ -n "${GITHUB_STEP_SUMMARY:-}" ]]; then
  cat "$report" >> "$GITHUB_STEP_SUMMARY"
fi

if [[ "${INPUT_COMMENT:-true}" == "true" && -n "${PR_NUMBER:-}" ]]; then
  marker="<!-- prompteval-action:${prompt} -->"
  body="$(printf '%s\n\n%s\n' "$marker" "$(cat "$report")")"
  existing="$(gh api "repos/${GITHUB_REPOSITORY}/issues/${PR_NUMBER}/comments" --paginate \
    --jq ".[] | select(.body | startswith(\"${marker}\")) | .id" | head -n1 || true)"
  if [[ -n "$existing" ]]; then
    gh api --method PATCH "repos/${GITHUB_REPOSITORY}/issues/comments/${existing}" -f body="$body" > /dev/null
  else
    gh api --method POST "repos/${GITHUB_REPOSITORY}/issues/${PR_NUMBER}/comments" -f body="$body" > /dev/null
  fi
fi

exit "$status"
