#!/bin/bash
# Extract code from HumanEval samples and run functional-correctness evaluation.
#
# Evaluates the Raspberry Pi results (results_rpi/<TAG>) and the previous envelope
# results (llm_energy_exp/results/E*). Outputs for the envelopes go to
# results_analysis/human_eval/<E>/ so that llm_energy_exp/ stays untouched.
#
# Environment variables:
#   TAG     experiment label used in results_rpi/<TAG>/...  (default: RPi)
#   PYTHON  python interpreter with human-eval installed (default: python)
set -uo pipefail

cd "$(dirname "$0")/.."

TAG="${TAG:-RPi}"
PYTHON="${PYTHON:-python}"
result_dir="results_rpi/${TAG}"

shopt -s nullglob
for records_file in "${result_dir}"/human_eval/*/human_eval_*_records.jsonl; do
    "$PYTHON" HE_utils/extract_code.py --input_file "$records_file" \
        --output_file "${records_file%_records.jsonl}_samples_extracted.jsonl"
done

"$PYTHON" HE_utils/evaluate_human_eval_envelops.py "$result_dir"

for env_dir in llm_energy_exp/results/E*; do
    env="$(basename "$env_dir")"
    out_dir="results_analysis/human_eval/${env}"
    for sample_file in "${env_dir}"/human_eval/*/human_eval_*_samples.jsonl; do
        model_dir="$(basename "$(dirname "$sample_file")")"
        mkdir -p "${out_dir}/human_eval/${model_dir}"
        "$PYTHON" HE_utils/extract_code.py --input_file "$sample_file" \
            --output_file "${out_dir}/human_eval/${model_dir}/human_eval_${model_dir}_samples_extracted.jsonl"
    done
    "$PYTHON" HE_utils/evaluate_human_eval_envelops.py "$out_dir"
done
#python correct_ptme.py
