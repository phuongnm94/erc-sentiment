#
for SEED in 100 101 102 103 104  105 106 107 108 109 ; do
    # python train_iemocap.py --active-listener --seed $SEED
    python train_emorynlp.py --active-listener --class-weight --residual --seed $SEED --dropout 0.25
done
