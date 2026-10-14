#!/usr/bin/env bash
# Run, in dependency order, every pending GPU check of tasks 0002-0007 (and
# the task_0008 GPU smoke) ON the GPU host, from the repository root, after
# `utils/sync_gpu_host.sh push` on the laptop. Each step logs to
# _testbed/outputs/gpu_pending_<stamp>/NN-<id>.log and one line of
# summary.tsv (id, task, PASS|FAIL|SKIP|MANUAL, exit code, seconds); pytest
# temporaries go to the sibling gpu_pending_<stamp>_pytest/, so pulling the
# logs never pulls run outputs. Exit code 1 when a step failed.
# A failing command is logged and its step goes on (FAIL at the end), so
# one missing piece does not hide the rest; runs needing the task_0004
# weights are skipped while they are missing (rfdetr would download them).
# Writes only inside the repository; uv from $UV (default ~/.local/bin/uv),
# always with --extra infer. Weights are never downloaded without
# --allow-downloads AND an interactive [s/N] per item; task_0004 downloads
# (caches outside the repo) are only printed (MANUAL). Fetch the logs with
# `utils/sync_gpu_host.sh pull gpu_pending_<stamp>`; run outputs stay here.
#
# Usage:
#   utils/gpu_pending.sh list
#   utils/gpu_pending.sh run [--from ID] [--only ID] [--skip-long] [--dry-run]
#                            [--allow-downloads] [--sam-hq base|huge] [--with-optional]
# ID is the full step id (05-gpu) or its number (05). Long runs (on by
# default) are skipped with --skip-long; use tmux for a full run.
# From the laptop:
#   utils/sync_gpu_host.sh push
#   ssh -t gpu-host 'cd ~/dev/det-depth-distance && utils/gpu_pending.sh run'
#   utils/sync_gpu_host.sh pull gpu_pending_<stamp>
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
UV="${UV:-$HOME/.local/bin/uv}"
VIDEOS=(IMG_4021 IMG_4022)
YOLO_DIR="$HOME/dev/ultralytics/_testbed/weights"
RF_DIR="${RF_HOME:-$HOME/.roboflow/models}"
# Exit codes of a step besides 0 (PASS) and failures: outside the range of
# the commands' own codes (pytest uses 1-5, signals 129-192).
SKIP_RC=200
MANUAL_RC=201

# id|task|kind|description, in dependency order. Kinds: abort (stop the run
# on failure), test, run, long (--skip-long), manual (printed only), download.
STEPS=(
  "01-sync|0002-0007|abort|uv sync --extra infer"
  "02-env|-|abort|disk, uv and CUDA facts"
  "03-unit|0007a.2,0008|test|whole unit suite with the infer extra (smoke, plots)"
  "04-regression|0002,0004.1,0005.2,0006.1|test|task_0001 regression before any download"
  "05-gpu|0002,0003.2,0006.1,0007a.2,0008|test|pytest -m gpu (contracts, label maps, GPU smoke)"
  "06-plots|0007a.2|test|evaluation plots (matplotlib from the infer extra)"
  "07-inspect|0007a.3|run|calibrate inspect on the test videos"
  "08-nocal|0007a.4|run|rfdetr_da3, 60 frames, without calibration"
  "09-fusion|0003.3|run|concat/nms/wbf x videos, 300 frames"
  "10-track|0006.2|long|rfdetr_da3_track on the whole videos"
  "11-track-variants|0006.3-0006.5|run|stride 3/5, ema/median, several detectors"
  "12-dl-0004|0004.2|manual|YOLO26-seg and RF-DETR Seg downloads (outside the repo)"
  "13-vocab-seg|0004.3|test|vocabulary of the objv1-seg checkpoints"
  "14-gpu-seg|0004.4|test|pytest -m gpu with the segmentation weights"
  "15-boxmask|0004.5|run|box x mask runs and the erosion sweep"
  "16-dl-0005|0005.3|download|refiner and SegFormer weights into weights/hf"
  "17-gpu-refiners|0005.4|test|refiner and segmenter GPU tests"
  "18-refiner-runs|0005.5|run|refiner validations A-E on both videos"
  "19-calibrated|0007a.5|run|calibrated runs (needs the real calibration file)"
)

DRY_RUN=0
SKIP_LONG=0
ALLOW_DOWNLOADS=0
WITH_OPTIONAL=0
SAM_HQ=""
FROM=""
ONLY=""
LOG=""  # log folder of this run (created only without --dry-run)
STEP_RC=0  # first failing exit code of the running step (set by x, in its subshell)

# The "Usage:" block of the header comment.
usage() {
  sed -n '/^# Usage:/,/^[^#]/{/^#/s/^# \{0,1\}//p}' "$0"
}

# Print a command; run it unless --dry-run. A failure is recorded in STEP_RC
# and the step goes on, so one missing piece does not hide the remaining
# validations of a step (errexit cannot be relied upon here: bash ignores it
# in functions called from a condition).
x() {
  printf '+'
  printf ' %q' "$@"
  printf '\n'
  if [[ $DRY_RUN -eq 1 ]]; then
    return 0
  fi
  local rc=0
  "$@" || rc=$?
  if [[ $rc -ne 0 ]]; then
    echo "FAILED (exit $rc): $*" >&2
    if [[ $STEP_RC -eq 0 ]]; then STEP_RC=$rc; fi
  fi
  return 0
}

uvr() { x "$UV" run --extra infer "$@"; }
ddd() { uvr python -m det_depth_distance "$@"; }
pytest_gpu() { uvr pytest --basetemp="${LOG}_pytest/$1" "${@:2}"; }

skip() {
  echo "SKIP: $*"
  exit "$SKIP_RC"
}

manual() {
  echo "MANUAL: $*"
  exit "$MANUAL_RC"
}

# Ask [s/N] on the terminal; "no" without a TTY.
confirm() {
  if [[ ! -t 0 ]]; then
    echo "no TTY: '$1' answered no"
    return 1
  fi
  local answer
  read -r -p "$1 [s/N] " answer
  [[ $answer == [sS] ]]
}

need_files() {
  local f
  for f in "$@"; do
    [[ -e $f ]] || skip "missing $f"
  done
}

step_01_sync() { x "$UV" sync --extra infer; }

step_02_env() {
  x df -h "$HOME" /dev/shm
  x "$UV" --version
  uvr python -c 'import torch; ok = torch.cuda.is_available(); print(torch.__version__, ok, torch.cuda.mem_get_info() if ok else None)'
}

step_03_unit() { pytest_gpu unit -q; }

step_04_regression() { pytest_gpu regression -m gpu tests/gpu/test_regression_task0001.py -v; }

step_05_gpu() { pytest_gpu gpu -m gpu -v -rs; }

step_06_plots() { pytest_gpu plots tests/test_eval_report.py -v -rs; }

step_07_inspect() {
  local v
  for v in "${VIDEOS[@]}"; do
    uvr det-depth-distance-calibrate inspect "_testbed/data/$v.MOV"
  done
}

step_08_nocal() {
  ddd --config configs/rfdetr_da3.yaml _testbed/data/IMG_4021.MOV \
    --set source.max_frames=60 --set output.tag=t7nocal
}

step_09_fusion() {
  local v agg tag
  for v in "${VIDEOS[@]}"; do
    for agg in "{type: concat}" "{type: nms, iou_thr: 0.5}" "{type: wbf, iou_thr: 0.55}"; do
      tag="$(echo "$agg" | cut -d' ' -f2 | tr -d ',}')"
      ddd --config configs/multi_detector_fused.yaml "_testbed/data/$v.MOV" \
        --set "aggregation=$agg" --set source.max_frames=300 --set "output.tag=$tag"
    done
  done
}

step_10_track() {
  [[ $SKIP_LONG -eq 0 ]] || skip "--skip-long"
  local v
  for v in "${VIDEOS[@]}"; do
    ddd --config configs/rfdetr_da3_track.yaml "_testbed/data/$v.MOV" --set output.tag=track
  done
}

step_11_track_variants() {
  local s f c video=_testbed/data/IMG_4021.MOV
  for s in 3 5; do
    ddd --config configs/rfdetr_da3_track.yaml "$video" \
      --set "source.stride=$s" --set "output.tag=track_s$s"
  done
  for f in ema median; do
    ddd --config configs/rfdetr_da3_track.yaml "$video" \
      --set "smoothing.filter=$f" --set "output.tag=track_$f"
  done
  for c in multi_detector multi_detector_fused; do
    ddd --config "configs/$c.yaml" "$video" --set "tracking={type: iou}" \
      --set smoothing.filter=kalman --set "output.tag=track_$c"
  done
}

step_12_dl_0004() {
  cat <<EOF
Run by hand after confirming the space (they write outside the repository):
  df -h ~ && cd $YOLO_DIR && for n in yolo26n-seg yolo26l-seg yolo26n-objv1-seg yolo26l-objv1-seg; do curl -fL -o \$n.pt https://github.com/ultralytics/assets/releases/download/v8.4.0/\$n.pt && stat -c "%n %s" \$n.pt; done
    expected bytes: yolo26n-seg 6719965, yolo26l-seg 63700037, yolo26n-objv1-seg 7185309, yolo26l-objv1-seg 64758405
  cd $ROOT && $UV run --extra infer python -c "import rfdetr; rfdetr.RFDETRSegNano(); rfdetr.RFDETRSegLarge()" && ls -l $RF_DIR/rf-detr-seg-*.pt
    expected bytes: rf-detr-seg-nano.pt 134545398, rf-detr-seg-large.pt 145055866
EOF
  manual "task_0004 downloads are never run by this script"
}

VOCAB_PY='import os
from pathlib import Path
import det_depth_distance.adapters.detectors.ultralytics_yolo  # sets YOLO_* env
Path(os.environ["YOLO_CONFIG_DIR"]).mkdir(parents=True, exist_ok=True)
from ultralytics import YOLO
from det_depth_distance.core.labels import identify_vocabulary
d = Path("~/dev/ultralytics/_testbed/weights").expanduser()
for w in ("yolo26n-objv1-seg", "yolo26l-objv1-seg"):
    m = YOLO(str(d / f"{w}.pt"))
    v = identify_vocabulary([m.names[i] for i in sorted(m.names)])
    print(w, m.task, len(m.names), v.name if v else None)'

step_13_vocab_seg() {
  need_files "$YOLO_DIR/yolo26n-objv1-seg.pt" "$YOLO_DIR/yolo26l-objv1-seg.pt"
  uvr python -c "$VOCAB_PY"
  echo "expected: 'segment 365 objects365' on both lines"
}

step_14_gpu_seg() {
  need_files "$YOLO_DIR/yolo26n-seg.pt"
  pytest_gpu gpu_seg -m gpu -v -rs
}

step_15_boxmask() {
  local v c e weights ran=0
  declare -A needs=(
    [rfdetr_seg_da3]="$RF_DIR/rf-detr-seg-large.pt"
    [yolo_seg_da3]="$YOLO_DIR/yolo26l-seg.pt"
    [yolo_o365_seg_da3]="$YOLO_DIR/yolo26l-objv1-seg.pt"
  )
  for c in rfdetr_seg_da3 yolo_seg_da3 yolo_o365_seg_da3; do
    weights="${needs[$c]}"
    if [[ ! -e $weights ]]; then
      echo "skipping $c: missing $weights"
      continue
    fi
    ran=1
    for v in "${VIDEOS[@]}"; do
      ddd --config "configs/$c.yaml" "_testbed/data/$v.MOV" \
        --set source.max_frames=300 --set output.tag=both
    done
  done
  # The erosion sweep needs the RF-DETR Seg weights; SKIP only when nothing ran
  # (a SKIP after some runs would hide their failures).
  if [[ ! -e ${needs[rfdetr_seg_da3]} ]]; then
    [[ $ran -eq 1 ]] || skip "missing every task_0004 Seg checkpoint"
    echo "skipping the erosion sweep: missing ${needs[rfdetr_seg_da3]}"
    return 0
  fi
  for v in "${VIDEOS[@]}"; do
    for e in 0 1 2; do
      ddd --config configs/rfdetr_seg_da3.yaml "_testbed/data/$v.MOV" \
        --set source.max_frames=300 --set "distance.mask_erode=$e" --set "output.tag=erode$e"
    done
  done
}

# Download one HF repo into weights/hf (filtered: never the duplicate .pt/.bin).
hf_download() {
  local repo="$1" bytes="$2" pattern="$3"
  if [[ $DRY_RUN -eq 0 ]] && ! confirm "download $repo ($bytes bytes) into weights/hf?"; then
    echo "not downloaded: $repo"
    return 0
  fi
  uvr python -c "from huggingface_hub import snapshot_download as d; d('$repo', cache_dir='weights/hf', allow_patterns=['*.json', '$pattern'])"
}

step_16_dl_0005() {
  [[ $ALLOW_DOWNLOADS -eq 1 ]] || skip "downloads need --allow-downloads (and a [s/N] per item)"
  x df -h "$ROOT"
  hf_download facebook/sam2.1-hiera-large 897897416 model.safetensors
  hf_download yonigozlan/EdgeTAM-hf 55912824 model.safetensors
  case "$SAM_HQ" in
    base) hf_download syscv-community/sam-hq-vit-base 379266384 model.safetensors ;;
    huge) hf_download syscv-community/sam-hq-vit-huge 2570816384 model.safetensors ;;
    *) echo "sam_hq not downloaded: choose --sam-hq base|huge (user decision)" ;;
  esac
  hf_download nvidia/segformer-b5-finetuned-cityscapes-1024-1024 338896453 pytorch_model.bin
  if [[ $WITH_OPTIONAL -eq 1 ]]; then
    hf_download facebook/sam-vit-base 374979480 model.safetensors
    if [[ -n ${HF_TOKEN:-} ]]; then
      hf_download facebook/sam3 3439938512 model.safetensors
    else
      echo "facebook/sam3 not downloaded: gated, needs HF_TOKEN and the accepted license"
    fi
    echo "MobileSAM (40728226 bytes, outside the repo) by hand:"
    echo "  curl -fL -o $YOLO_DIR/mobile_sam.pt https://github.com/ultralytics/assets/releases/download/v8.4.0/mobile_sam.pt"
  fi
  x du -sb weights/hf
}

step_17_gpu_refiners() {
  pytest_gpu refiners -m gpu -v -rs
  pytest_gpu refiner_contracts -m gpu tests/gpu/test_adapter_contracts.py -v \
    -k "refiner or segformer or twinlitenet or segmenters" -rs -s
}

step_18_refiner_runs() {
  local v video c t hq refiners=("sam2" "edgetam")
  # E: sam_hq is the SAM-HQ the user downloaded (base or huge, a task_0005
  # decision), never assumed
  hq="$(sam_hq_model)"
  if [[ -n $hq ]]; then
    refiners+=("sam_hq, model: $hq")
  else
    echo "sam_hq skipped: no syscv-community/sam-hq-vit-{base,huge} in weights/hf"
  fi
  for v in "${VIDEOS[@]}"; do
    video="_testbed/data/$v.MOV"
    # A: box x SAM2 mask
    for c in rfdetr_sam2_da3 yolo_o365_sam2_da3; do
      ddd --config "configs/$c.yaml" "$video" --set source.max_frames=300 --set output.tag=sam2
    done
    # B: SAM2 x native masks, only with the task_0004 Seg weights (rfdetr
    # would download them on its own)
    if [[ -e $RF_DIR/rf-detr-seg-large.pt ]]; then
      ddd --config configs/rfdetr_seg_da3.yaml "$video" --set source.max_frames=300 \
        --set "refiners=[{type: sam2, hf_cache: weights/hf, targets: all}]" --set output.tag=sam2_all
      ddd --config configs/rfdetr_seg_da3.yaml "$video" --set source.max_frames=300 --set output.tag=native
    else
      echo "skipping B: missing $RF_DIR/rf-detr-seg-large.pt (task_0004 download)"
    fi
    # C: semantic exclusion
    ddd --config configs/rfdetr_semantic_da3.yaml "$video" --set source.max_frames=300 --set output.tag=excl
    ddd --config configs/rfdetr_semantic_da3.yaml "$video" --set source.max_frames=300 \
      --set "distance.exclude_semantic=[]" --set output.tag=noexcl
    ddd --config configs/rfdetr_semantic_da3.yaml "$video" --set source.max_frames=300 \
      --set "semantic=[{name: lanes, type: twinlitenet}]" \
      --set "distance.exclude_semantic=[lanes.drivable,lanes.lane]" --set output.tag=lanes
    ddd --config configs/rfdetr_lanes_scene_da3.yaml "$video" --set source.max_frames=300 \
      --set output.save_depth=true --set output.tag=multi
    ddd --config configs/rfdetr_lanes_scene_da3.yaml "$video" --set source.max_frames=300 \
      --set "distance.exclude_semantic=[]" --set output.tag=multi_noexcl
    # E2/E3: cost (against the YOLO-seg native masks, task_0004 weights) and
    # distance effect per refiner
    for t in "${refiners[@]}"; do
      if [[ -e $YOLO_DIR/yolo26l-seg.pt ]]; then
        ddd --config configs/yolo_seg_da3.yaml "$video" --set source.max_frames=300 \
          --set "refiners=[{type: $t, hf_cache: weights/hf, targets: all}]" --set "output.tag=cmp_${t%%,*}"
      else
        echo "skipping cmp_${t%%,*}: missing $YOLO_DIR/yolo26l-seg.pt (task_0004 download)"
      fi
      ddd --config configs/rfdetr_sam2_da3.yaml "$video" --set source.max_frames=300 \
        --set "refiners=[{type: $t, hf_cache: weights/hf}]" --set "output.tag=dist_${t%%,*}"
    done
  done
}

# The SAM-HQ checkpoint in weights/hf: the --sam-hq choice, else whichever
# was downloaded (huge first, the code default); empty when none is.
sam_hq_model() {
  local v
  for v in ${SAM_HQ:-huge base}; do
    if [[ -d weights/hf/models--syscv-community--sam-hq-vit-$v ]]; then
      echo "syscv-community/sam-hq-vit-$v"
      return 0
    fi
  done
}

step_19_calibrated() {
  local calibration=configs/calibration/iphone14promax_main_1080p30.yaml v
  [[ -f $calibration ]] || skip "missing $calibration (calibration recording, task_0007 b.1)"
  for v in "${VIDEOS[@]}"; do
    ddd --config configs/rfdetr_da3_calibrated.yaml "_testbed/data/$v.MOV" --set output.tag=calib
    ddd --config configs/rfdetr_da3_calibrated.yaml "_testbed/data/$v.MOV" \
      --set camera.calibration.undistort=true --set output.tag=calib_undist
  done
}

# Whether step id $1 is selected by $2 (full id or its number).
matches() { [[ $1 == "$2" || ${1%%-*} == "$2" ]]; }

# Whether $1 selects some step.
known_step() {
  local entry
  for entry in "${STEPS[@]}"; do
    if matches "${entry%%|*}" "$1"; then return 0; fi
  done
  return 1
}

list_steps() {
  local entry id task kind desc
  for entry in "${STEPS[@]}"; do
    IFS='|' read -r id task kind desc <<<"$entry"
    printf '%-18s %-32s %-9s %s\n' "$id" "$task" "$kind" "$desc"
  done
}

# Run one step: log, time, status line in summary.tsv. Returns 1 when an
# "abort" step failed.
run_step() {
  local id="$1" task="$2" kind="$3" fn rc status start seconds
  fn="step_${id//-/_}"
  echo "=== $id ($task, $kind)"
  if [[ $DRY_RUN -eq 1 ]]; then
    rc=0
    ("$fn") || rc=$?
    case $rc in 0 | "$SKIP_RC" | "$MANUAL_RC") ;; *) echo "dry run of $id failed ($rc)" ;; esac
    return 0
  fi
  start=$SECONDS
  # The step's exit code: skip/manual exit directly, otherwise the first
  # failure recorded by x (or the function's own failure).
  set +e
  (
    "$fn" || STEP_RC=$?
    exit "$STEP_RC"
  ) 2>&1 | tee "$LOG/$id.log"
  rc=${PIPESTATUS[0]}
  set -e
  seconds=$((SECONDS - start))
  case $rc in
    0) status=PASS ;;
    "$SKIP_RC") status=SKIP ;;
    "$MANUAL_RC") status=MANUAL ;;
    *) status=FAIL ;;
  esac
  printf '%s\t%s\t%s\t%s\t%s\n' "$id" "$task" "$status" "$rc" "$seconds" >>"$LOG/summary.tsv"
  echo "--- $id: $status (exit $rc, ${seconds} s)"
  [[ $status != FAIL || $kind != abort ]]
}

run_steps() {
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --from) FROM="${2:?--from needs an ID}"; shift ;;
      --only) ONLY="${2:?--only needs an ID}"; shift ;;
      --skip-long) SKIP_LONG=1 ;;
      --dry-run) DRY_RUN=1 ;;
      --allow-downloads) ALLOW_DOWNLOADS=1 ;;
      --with-optional) WITH_OPTIONAL=1 ;;
      --sam-hq)
        SAM_HQ="${2:?--sam-hq needs base or huge}"
        [[ $SAM_HQ == base || $SAM_HQ == huge ]] || { echo "--sam-hq: base or huge" >&2; exit 2; }
        shift
        ;;
      *) echo "unknown option: $1" >&2; usage >&2; exit 2 ;;
    esac
    shift
  done
  local entry id task kind desc started=1 failed=0 stamp sel
  for sel in "$FROM" "$ONLY"; do
    if [[ -n $sel ]] && ! known_step "$sel"; then
      echo "unknown step: $sel (see: $0 list)" >&2
      exit 2
    fi
  done
  [[ -z $FROM ]] || started=0
  stamp="$(date +%Y%m%d_%H%M%S)"
  LOG="$ROOT/_testbed/outputs/gpu_pending_$stamp"
  if [[ $DRY_RUN -eq 0 ]]; then
    mkdir -p "$LOG"
    printf 'id\ttask\tstatus\texit\tseconds\n' >"$LOG/summary.tsv"
    echo "logs: $LOG"
  fi
  for entry in "${STEPS[@]}"; do
    IFS='|' read -r id task kind desc <<<"$entry"
    if [[ $started -eq 0 ]] && matches "$id" "$FROM"; then started=1; fi
    [[ $started -eq 1 ]] || continue
    if [[ -n $ONLY ]] && ! matches "$id" "$ONLY"; then continue; fi
    if ! run_step "$id" "$task" "$kind"; then
      echo "aborting: $id failed" >&2
      failed=1
      break
    fi
  done
  if [[ $DRY_RUN -eq 0 ]]; then
    column -t -s $'\t' "$LOG/summary.tsv" || cat "$LOG/summary.tsv"
    echo "On the laptop: utils/sync_gpu_host.sh pull gpu_pending_$stamp   (logs only)"
    grep -q $'\tFAIL\t' "$LOG/summary.tsv" && failed=1
  fi
  return "$failed"
}

case "${1:-}" in
  list) list_steps ;;
  run) shift; run_steps "$@" ;;
  *) usage; exit 1 ;;
esac
