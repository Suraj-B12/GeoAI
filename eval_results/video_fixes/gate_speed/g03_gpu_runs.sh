#!/bin/bash
# Everything that needs the GPU, under the shared lock: the full test suite of the isolated after-tree,
# then the timed extraction runs (before x2, after x2 interleaved, after-default x1). Writes out/gpu_runs.log.
set -u
LOCK=/c/GeoAI/gpu.lock
PY=/c/Users/PRO-LAB-3/Documents/GeoAI_Video/pipeline/.venv/Scripts/python
BEF=C:/GeoAI/tmp/gate_speed_before/pipeline
AFT=C:/GeoAI/tmp/gate_speed_after/pipeline
F=/c/Users/PRO-LAB-3/Documents/GeoAI_Video/field
EV=$F/runs_eval
S="$F/sessions/20261026T114332Z_SM-G991B-F0D1_f428 $F/sessions/20261026T114907Z_SM-G991B-F0D1_293d $F/sessions/20261027T050257Z_SM-G991B-F0D1_d47c"
until mkdir $LOCK 2>/dev/null; do echo "$(date -u +%FT%TZ) waiting for gpu.lock: $(cat $LOCK/owner.txt 2>/dev/null)"; sleep 30; done
trap 'rm -rf $LOCK' EXIT
printf 'purpose: gate_speed study (GeoAI_Video segment-mode extraction timing + pipeline test suite), agent gate/speed\nstart: %s\n' "$(date -u +%FT%TZ)" > $LOCK/owner.txt
echo "lock acquired $(date -u +%FT%TZ)"
nvidia-smi --query-gpu=name,memory.used,utilization.gpu --format=csv
geo() { local tree=$1; shift; "$PY" -c "import sys; sys.path.insert(0, r'$tree'); import geoai; print('geoai from', geoai.__file__, flush=True); from geoai.cli import main; sys.exit(main(sys.argv[1:]))" "$@"; }
if [ "${SKIP_TESTS:-0}" != 1 ]; then
  echo "== tests $(date -u +%FT%TZ)"
  (cd $AFT && "$PY" -c "import sys, pytest; sys.path.insert(0, '.'); sys.exit(pytest.main(['-q', '-m', 'slow or not slow', '-p', 'no:cacheprovider']))") 2>&1 | tail -40
fi
echo "== warm cache $(date -u +%FT%TZ)"
cat $F/sessions/*_f428/video/*.mp4 $F/sessions/*_293d/video/*.mp4 $F/sessions/*_d47c/video/*.mp4 > /dev/null
mkdir -p $EV
for step in before:1 after:1 before:2 after:2 afterdefault:1; do
  v=${step%%:*}; n=${step##*:}; out=$EV/${v}_rep$n
  echo "== $v rep $n $(date -u +%FT%TZ)"
  case $v in
    before) geo $BEF process $S --out $out ;;
    after) geo $AFT --config $AFT/calibration/segments_probe_gate.json process $S --out $out ;;
    afterdefault) geo $AFT process $S --out $out ;;
  esac
  echo "exit $?"
done
echo "done $(date -u +%FT%TZ)"
