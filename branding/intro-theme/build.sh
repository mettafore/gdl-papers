#!/usr/bin/env bash
# Renders the intro theme. usage: ./build.sh [classic|vivid] [slow]
# "slow" stretches the picture 1.5x (24.75 s) and uses the composed theme in score.py.
# Frames go to $WORK (default: a temp dir).
set -euo pipefail
cd "$(dirname "$0")"
LOOK="${1:-classic}"
if [ "$LOOK" = vivid ]; then export PAGE=theme-vivid.html; NAME=intro-theme-vivid; MUSIC=--vivid
else export PAGE=theme.html; NAME=intro-theme; MUSIC=; fi
WORK="${WORK:-$(mktemp -d)}"; FPS=60; mkdir -p out
S=1; Q=""; if [ "${2:-}" = slow ]; then S=1.5; Q="slow=1.5&"; NAME="$NAME-breathe"; fi
N=$(python3 -c "print(round(16.5*$S*$FPS))"); A=$((N/3)); B=$((2*N/3)); C=$(python3 -c "print(int(775*$S))"); CUT=$(python3 -c "print(10.5*$S)")
node render.mjs "$WORK/frames" $FPS 0 $A "$Q" & node render.mjs "$WORK/frames" $FPS $A $B "$Q" &
node render.mjs "$WORK/frames" $FPS $B $N "$Q" & node render.mjs "$WORK/frames_clean" $FPS $C $N "${Q}title=0" & wait
if [ "$S" = 1 ]; then python3 music.py "out/$NAME.wav" $MUSIC; else python3 score.py "out/$NAME.wav"; fi
mkdir -p "$WORK/clean"
for f in "$WORK"/frames/*.png; do ln -sf "$f" "$WORK/clean/$(basename "$f")"; done
for f in "$WORK"/frames_clean/*.png; do ln -sf "$f" "$WORK/clean/$(basename "$f")"; done
X264="-c:v libx264 -preset slow -crf 19 -maxrate 14M -bufsize 28M"
enc() {  # film grain also dithers the dark gradients against banding
  ffmpeg -loglevel error -y -framerate $FPS -i "$1/f%05d.png" -i "out/$NAME.wav" \
    -vf "noise=alls=3:allf=t,format=yuv420p" $X264 \
    -af "volume=-2dB" -c:a aac -b:a 256k -movflags +faststart -shortest "$2"
}
enc "$WORK/frames" "out/$NAME.mp4"
enc "$WORK/clean" "out/$NAME-clean.mp4"
ffmpeg -loglevel error -y -ss $CUT -i "out/$NAME.mp4" -vf "fade=in:st=0:d=0.5" -af "afade=in:st=0:d=0.6" \
  $X264 -c:a aac -b:a 256k -movflags +faststart "out/$NAME-sting.mp4"
ls -la out
