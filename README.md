# LDCT-Octave
Low-Dose CT Denoising using Octave Convolution

# Training
Train with U-Net
```
CUDA_VISIBLE_DEVICES=0 python train.py --model=unet --test_fold=10 --scheduler
```

Train with REDCNN
```
CUDA_VISIBLE_DEVICES=0 python train_ct.py --model=redcnn --test_fold=10 --scheduler
```

Train with Octave convolution
```
CUDA_VISIBLE_DEVICES=0 python train_ct.py --model=cnn_oct --test_fold=10 --scheduler --alpha=0.75
```

# Test
For test, you need to specify some arguments.
```
--test_date: the directory where your experiment located.
--test_epoch: the epoch of your test
--desc     : description of your experiment. your predictions (image, nitfi, plots will be saved the folder given description)
```

Test with redcnn
```
python test_ct.py --model=redcnn --test_date=$TEST_DATE --test_epoch=best --subject=L506 --desc="redcnn"
```

Test with Octave convolution
```
python test_ct.py --model=cnn_oct --alpha=$ALPHA --test_date=$TEST_DATE --test_epoch=best --subject=L506 --desc="cnn_oct"
```