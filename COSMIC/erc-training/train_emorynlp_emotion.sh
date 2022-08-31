#
for SEED in 100 101 102 103 104 ; do
    # python train_iemocap.py --active-listener --seed $SEED
    python train_emorynlp.py --active-listener --class-weight --residual --seed $SEED 
done
