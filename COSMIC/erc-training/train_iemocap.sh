# #

#
for SEED in  101 102 103 104 105 106 107 108 109 ; do
    python train_iemocap.py --active-listener --seed $SEED --tensorboard --no-self-attn-emotions
    # python train_dailydialog.py --active-listener --class-weight --residual --seed $SEED
done

for SEED in  101 102 103 104 105 106 107 108 109 ; do
    python train_iemocap.py --active-listener --seed $SEED --tensorboard 
    # python train_dailydialog.py --active-listener --class-weight --residual --seed $SEED
done