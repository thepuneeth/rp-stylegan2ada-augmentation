#!/usr/bin/env bash
# Rent a 4090, run pod_run.sh, pull results, ALWAYS delete the pod.
set -uo pipefail
S=/private/tmp/claude-501/-Users-puneeth-Documents-Research-GAN/0b3ee4d9-223d-4998-8ab6-2169cc3c3f6d/scratchpad
OUT=$S/results; mkdir -p "$OUT"
KEY=$HOME/.runpod/ssh/runpodctl-ssh-key
SSHOPT="-i $KEY -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR -o ServerAliveInterval=30"
PID=""

teardown() {
  if [ -n "$PID" ]; then
    echo "=== TEARDOWN: deleting $PID ==="
    runpodctl pod delete "$PID" 2>&1 | tail -2
    sleep 5
    echo "=== pods remaining: $(runpodctl pod list 2>&1 | tr -d '\n' | head -c 200) ==="
  fi
}
trap teardown EXIT INT TERM

TERM_AT=$(python3 -c "import datetime;print((datetime.datetime.now(datetime.timezone.utc)+datetime.timedelta(hours=4)).strftime('%Y-%m-%dT%H:%M:%SZ'))")
echo "=== CREATE (hard backstop terminate-after $TERM_AT) ==="
CREATE=$(runpodctl pod create --name vs-infer3 \
  --template-id runpod-torch-v240 \
  --gpu-id "NVIDIA GeForce RTX 4090" \
  --cloud-type SECURE --gpu-count 1 \
  --container-disk-in-gb 30 --volume-in-gb 40 \
  --volume-mount-path /workspace \
  --terminate-after "$TERM_AT" 2>&1)
echo "$CREATE" | head -c 600; echo
PID=$(echo "$CREATE" | python3 -c "import json,sys;d=json.load(sys.stdin);print(d.get('id') or d.get('pod',{}).get('id',''))" 2>/dev/null)
[ -z "$PID" ] && { echo "CREATE FAILED"; exit 1; }
echo "POD_ID=$PID"

echo "=== WAIT for RUNNING ==="
for i in $(seq 1 60); do
  ST=$(runpodctl pod get "$PID" 2>/dev/null | python3 -c "import json,sys;d=json.load(sys.stdin);print(d.get('desiredStatus',''))" 2>/dev/null)
  echo "  [$i] status=$ST"
  [ "$ST" = "RUNNING" ] && break
  sleep 10
done

echo "=== WAIT for SSH ==="
HOST=""; PORT=""
for i in $(seq 1 60); do
  INFO=$(runpodctl ssh info "$PID" 2>/dev/null)
  HOST=$(echo "$INFO" | python3 -c "import json,sys;d=json.load(sys.stdin);print(d.get('host') or d.get('ip') or '')" 2>/dev/null)
  PORT=$(echo "$INFO" | python3 -c "import json,sys;d=json.load(sys.stdin);print(d.get('port') or '')" 2>/dev/null)
  if [ -n "$HOST" ] && [ -n "$PORT" ]; then
    if ssh $SSHOPT -p "$PORT" root@"$HOST" 'echo SSH_OK' 2>/dev/null | grep -q SSH_OK; then
      echo "  ssh up: $HOST:$PORT"; break
    fi
  fi
  echo "  [$i] ssh not ready (host=$HOST port=$PORT)"; sleep 10
done
[ -z "$HOST" ] && { echo "NO SSH"; exit 1; }
SSHC="ssh $SSHOPT -p $PORT root@$HOST"

echo "=== UPLOAD ==="
$SSHC 'mkdir -p /workspace/pkl /workspace/data /workspace/out'
scp $SSHOPT -P "$PORT" $S/stage/code.tgz $S/stage/clean_111.zip $S/stage/leak_111.zip root@"$HOST":/workspace/data/ || exit 1
$SSHC 'cd /workspace && tar -xzf data/code.tgz && ls stylegan2-ada-pytorch-main | head -3'
# winners first: if anything dies later, the F1 fix and the pools are already possible
for f in clean-002822-Gema.pkl leak-003000-Gema.pkl clean-000403-Gema.pkl clean-000806-Gema.pkl clean-001209-Gema.pkl clean-001612-Gema.pkl; do
  echo "  uploading $f"
  scp $SSHOPT -P "$PORT" "$S/stage/$f" root@"$HOST":/workspace/pkl/ || echo "  UPLOAD FAILED $f"
done
scp $SSHOPT -P "$PORT" $S/pod_run.sh root@"$HOST":/workspace/

echo "=== RUN ==="
$SSHC 'bash /workspace/pod_run.sh' 2>&1 | tee "$OUT/pod_run.log"

echo "=== DOWNLOAD ==="
scp $SSHOPT -P "$PORT" root@"$HOST":/workspace/out/pools.tgz "$OUT/" || echo "no pools.tgz"
scp $SSHOPT -P "$PORT" root@"$HOST":/workspace/out/logs.tgz  "$OUT/" || echo "no logs.tgz"
ls -la "$OUT"
echo "=== SESSION3 DONE ==="
