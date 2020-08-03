#!/bin/sh
DEVICE=$1
MODEL=$2
BLOCKS=$3
ALPHA=$4

for FOLD in 1 2 3 4 5 6 7 8 9 10; do
    echo "Test $MODEL $FOLD fold"

    CHECKPOINT="/home/dongkyu/Workspace/MICCAI/checkpoints/$MODEL/$FOLD"
    DATE=$(ls $CHECKPOINT)
    echo "${DATE[0]}"

    python main.py --mode="test" --device="$DEVICE" --model="$MODEL" --fold=$FOLD --blocks=$BLOCKS --alpha=$ALPHA --test_date=$DATE
done

