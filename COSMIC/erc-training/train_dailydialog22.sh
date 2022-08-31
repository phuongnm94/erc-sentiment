#
for SEED in  106 ; do
    # python train_iemocap.py --active-listener --seed $SEED
    python train_dailydialog.py --active-listener --class-weight --residual --seed $SEED --dropout 0.3
done

