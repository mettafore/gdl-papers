#!/usr/bin/env bash
# CO2 isotherm in RUBTAK: one GCMC run per pressure, lowest first.
# Usage (from this folder): ./run_isotherm.sh            -> all pressures below
#                           ./run_isotherm.sh 1 5        -> only these (bar)
# Each run: p<bar>bar.yaml, p<bar>bar.h5, p<bar>bar.log; wall times go to timings.tsv.
set -euo pipefail
cd "$(dirname "$0")"

pressures=("$@")
[ ${#pressures[@]} -eq 0 ] && pressures=(0.01 0.1 0.5 1 2 5 10)

# Prefixes every line (and every \r progress redraw) with its first argument, streaming.
tag_lines='$|=1;$t=shift;$a=1;while(sysread(STDIN,$b,4096)){$b=~s/([\r\n])(?=[^\r\n])/$1$t/g;$b=$t.$b if $a&&$b=~/^[^\r\n]/;$a=($b=~/[\r\n]$/);print $b}'

i=0
for p in "${pressures[@]}"; do
  i=$((i + 1))  # run number, for the "[2/7]" counter
  pa=$(python3 -c "print(int(round($p * 1e5)))")
  name="p${p}bar"
  sed -e "s/PRESSURE_PA/${pa}/" -e "s/OUT_FILE/${name}.h5/" template.yaml > "${name}.yaml"
  echo "=== [${i}/${#pressures[@]}] ${p} bar (${pa} Pa) started $(date '+%H:%M:%S') ==="
  start=$(date +%s)
  # caffeinate: Mac stays awake. PYTHONUNBUFFERED: print immediately, not in chunks.
  # perl: put "[2/7 0.1 bar]" at the start of every line, progress-bar redraws (\r) included.
  # tee: show on screen and save to the log.
  PYTHONUNBUFFERED=1 caffeinate -i uv run kups_mcmc_rigid "${name}.yaml" 2>&1 \
    | perl -e "$tag_lines" "[${i}/${#pressures[@]} ${p} bar] " \
    | tee "${name}.log"
  secs=$(( $(date +%s) - start ))
  printf "%s\t%s\n" "$p" "$secs" >> timings.tsv
  echo "=== [${i}/${#pressures[@]}] ${p} bar done in $((secs / 60)) min $((secs % 60)) s ==="
done
