#!/bin/sh
DEVICE=$1
MODEL=$2
ALPHA=$3

#for FOLD in 1 2 3 4 5 6 8 9 10; do
for FOLD in 7 1 2 3 4 5 6 8 9; do
  echo "Train $MODEL ${FOLD} fold"
  python main.py --mode="train" --device="$DEVICE" --model="$MODEL" --alpha=$ALPHA --fold=${FOLD}

  CHECKPOINT="/home/dongkyu/Workspace/CT_Denoising/cnn_oct_test/checkpoints/$MODEL/$FOLD"
  DATE=$(ls $CHECKPOINT)
  echo "Test $MODEL $FOLD fold (${DATE[0]})"
  python main.py --mode="test" --device="$DEVICE" --model="$MODEL" --alpha=$ALPHA --fold=$FOLD --test_date=$DATE
done
