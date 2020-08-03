#!/bin/sh
DEVICE=$1
MODEL=$2
ALPHA=$3

for FOLD in 1 2 3 4 5 6 7 8 9 10; do
  echo "Train $MODEL ${FOLD} fold"
  python main.py --mode="train" --device="$DEVICE" --model="$MODEL" --fold=${FOLD} --loss "L1" --alpha=$ALPHA

  echo "Test $MODEL $FOLD fold"

  CHECKPOINT="/home/dongkyu/Workspace/MICCAI/checkpoints/$MODEL/$FOLD"
  DATE=$(ls $CHECKPOINT)
  echo "${DATE[0]}"

  python main.py --mode="test" --device="$DEVICE" --model="$MODEL" --fold=$FOLD --alpha=$ALPHA --test_date=$DATE
done
