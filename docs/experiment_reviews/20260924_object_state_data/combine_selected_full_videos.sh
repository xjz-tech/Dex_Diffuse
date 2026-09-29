#!/usr/bin/env bash
set -euo pipefail

base=/home/carus/Program/Dexterous_Manipulation/Dex_diffuse/docs/experiment_reviews/20260924_object_state_data
root="$base/corrected_direct_vs_reference/all_full_episodes"
output=/home/carus/Downloads
ffmpeg=/home/carus/miniforge3/envs/dp/lib/python3.10/site-packages/imageio_ffmpeg/binaries/ffmpeg-linux-x86_64-v7.0.2

if (( $# < 1 )); then
  echo 'usage: combine_selected_full_videos.sh episode [episode ...]' >&2
  exit 2
fi
mkdir -p "$output"
list="$output/Object_state_data_full_four_episode_concat.txt"
: > "$list"
for episode in "$@"; do
  printf -v label '%02d' "$episode"
  source="$root/episode_${label}/video_full/episode_${label}_real_direct_guide2exec2_exec1_full.mp4"
  target="$output/Object_state_episode_${label}_full_four_panel.mp4"
  test -s "$source"
  if [[ ! -s "$target" || "$source" -nt "$target" ]]; then
    "$ffmpeg" -hide_banner -loglevel error -y -i "$source" \
      -c:v libx264 -preset veryfast -crf 20 -pix_fmt yuv420p -r 30 \
      -movflags +faststart -an "$target"
  fi
  printf "file '%s'\n" "$target" >> "$list"
  echo "ready: $target"
done
combined="$output/Object_state_data_four_distinct_episodes_full.mp4"
"$ffmpeg" -hide_banner -loglevel error -y -f concat -safe 0 -i "$list" \
  -c copy -movflags +faststart "$combined"
echo "ready: $combined"
