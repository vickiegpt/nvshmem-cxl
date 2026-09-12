#!/usr/bin/env bash
# Size sweep + conventional-transport baseline for the NVSHMEM-CXL RL rollout.
# Writes results/rl_rollout_cxl.tsv and rl_rollout_vram.tsv.  Every number in
# the README comes from this script, nothing transcribed by hand.
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p results

run_case() {
    local mode=$1 mib=$2
    ./run.sh --mode "$mode" --size-mib "$mib" --rounds 6 --steps 2 --tokens 16 2>/dev/null \
        | grep -E "^\[rollout\] round|^\[trainer\] step|^\[trainer\] done" \
        | python3 -c '
import sys

mode, mib = sys.argv[1], sys.argv[2]
gpus, tps, bws, pubs, opts, stages = [], [], [], [], [], []
stale = ""
for line in sys.stdin:
    f = line.split()
    if f[0] == "[rollout]":
        for tok in f:
            if tok.startswith("gpu="):
                gpus.append(float(tok[4:].rstrip("ms")))
            elif tok.startswith("tok/s="):
                tps.append(float(tok[6:]))
            elif tok.startswith("weight-read="):
                bws.append(float(tok[12:].rstrip("GB/s")))
            elif tok.startswith("stage="):
                v = float(tok[6:].rstrip("ms"))
                if v > 0:
                    stages.append(v)
    elif f[0:2] == ["[trainer]", "step"]:
        for tok in f:
            if tok.startswith("publish="):
                pubs.append(float(tok[8:].rstrip("ns")))
            elif tok.startswith("opt="):
                opts.append(float(tok[4:].rstrip("ms")))
    elif f[0:2] == ["[trainer]", "done"]:
        for tok in f:
            if tok.startswith("stale="):
                stale = tok[6:]

if gpus:
    n = len(gpus)
    pub = sum(pubs) / len(pubs) if pubs else 0
    stage = sum(stages) / len(stages) if stages else 0
    k = (stage * 1e6 / pub) if (stage > 0 and pub > 0) else 0  # copies-worth of publishes
    print(f"{mode}\t{mib}\t{sum(gpus)/n:.2f}\t{sum(tps)/n:.1f}\t{sum(bws)/n:.1f}\t"
          f"{pub:.0f}\t{stage:.2f}\t{k:.0f}\t{(sum(opts)/len(opts)) if opts else 0:.2f}\t{stale}")
' "$mode" "$mib"
}

for mode in cxl rdma vram; do
    out="results/rl_rollout_${mode}.tsv"
    printf "mode\tweights_mib\tgpu_ms\ttok_s\tweight_read_gbps\tpublish_ns\tstage_ms\tpublishes_per_copy\topt_ms\tstale\n" > "$out"
    for mib in 16 64 256; do
        echo "running $mode $mib MiB..." >&2
        run_case "$mode" "$mib" >> "$out" || true
    done
done

for mode in cxl rdma vram; do
    echo "== $mode =="
    column -t -s$'\t' "results/rl_rollout_${mode}.tsv" 2>/dev/null || cat "results/rl_rollout_${mode}.tsv"
done
