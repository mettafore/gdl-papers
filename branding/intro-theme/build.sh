#!/usr/bin/env bash
# Renders the intro theme. Frames go to $WORK (default: a temp dir).
set -euo pipefail
cd "$(dirname "$0")"
WORK="${WORK:-$(mktemp -d)}"; FPS=60; mkdir -p out
node render.mjs "$WORK/frames" $FPS 0 330 & node render.mjs "$WORK/frames" $FPS 330 660 &
node render.mjs "$WORK/frames" $FPS 660 990 & node render.mjs "$WORK/frames_clean" $FPS 775 990 "title=0" & wait
python3 music.py out/intro-theme.wav
mkdir -p "$WORK/clean"
for f in "$WORK"/frames/*.png; do ln -sf "$f" "$WORK/clean/$(basename "$f")"; done
for f in "$WORK"/frames_clean/*.png; do ln -sf "$f" "$WORK/clean/$(basename "$f")"; done
enc() {  # film grain also dithers the dark gradients against banding
  ffmpeg -loglevel error -y -framerate $FPS -i "$1/f%05d.png" -i out/intro-theme.wav \
    -vf "noise=alls=3:allf=t,format=yuv420p" -c:v libx264 -preset slow -crf 19 -maxrate 14M -bufsize 28M \
    -af "volume=-2dB" -c:a aac -b:a 256k -movflags +faststart -shortest "$2"
}
enc "$WORK/frames" out/intro-theme.mp4
enc "$WORK/clean" out/intro-theme-clean.mp4
ffmpeg -loglevel error -y -ss 10.5 -i out/intro-theme.mp4 -vf "fade=in:st=0:d=0.5" -af "afade=in:st=0:d=0.6" \
  -c:v libx264 -preset slow -crf 19 -maxrate 14M -bufsize 28M -c:a aac -b:a 256k -movflags +faststart out/intro-theme-sting.mp4
ls -la out
