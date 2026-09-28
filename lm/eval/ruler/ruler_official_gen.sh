#!/usr/bin/env bash
# Generate RULER NIAH data with NVIDIA's official scripts + Llama-2 tokenizer. Usage: bash lm/eval/ruler/ruler_official_gen.sh [N=500]
set -u; N=${1:-500}; R=${R:-$(cd "$(dirname "$0")/../../.." && pwd)}; S=$R/external/RULER/scripts/data/synthetic; OUT=$R/external/RULER/data_official
T='Some special magic {type_needle_v} are hidden within the following text. Make sure to memorize it. I will quiz you about the {type_needle_v} afterwards.
{context}
What are all the special magic {type_needle_v} for {query} mentioned in the provided text? The special magic {type_needle_v} for {query} mentioned in the provided text are'
gen(){ local name=$1 L=$2; shift 2
  [ -s $OUT/${name}_$L/validation.jsonl ] && { echo "have $name $L"; return; }
  (cd $S && LD_LIBRARY_PATH= PYTHONPATH=$R/external/RULER/scripts/data $R/python niah.py --save_dir $OUT --save_name ${name}_$L --subset validation \
     --tokenizer_path $R/data/llama2_tok --tokenizer_type hf --max_seq_length $L --tokens_to_generate 128 --num_samples $N --random_seed 42 \
     --template "$T" "$@" > $R/analysis/ruler_official_gen_${name}_$L.log 2>&1) && echo "done $name $L ($(wc -l < $OUT/${name}_$L/validation.jsonl) rows)" || echo "FAIL $name $L"; }
for L in 1024 2048 4096 8192; do gen niah_single_1 $L --type_haystack noise --type_needle_k words --type_needle_v numbers; done &
for L in 1024 2048 4096 8192; do gen niah_single_2 $L --type_haystack essay --type_needle_k words --type_needle_v numbers; done &
for L in 1024 2048 4096; do gen niah_single_3 $L --type_haystack essay --type_needle_k words --type_needle_v uuids; done &
for L in 1024 2048 4096; do gen niah_multikey_1 $L --type_haystack essay --type_needle_k words --type_needle_v numbers --num_needle_k 4; done &
for L in 1024 2048 4096; do gen niah_multikey_2 $L --type_haystack needle --type_needle_k words --type_needle_v numbers; done &
for L in 1024 2048 4096; do gen niah_multikey_3 $L --type_haystack needle --type_needle_k uuids --type_needle_v uuids; done &
wait; echo ALL-DONE
