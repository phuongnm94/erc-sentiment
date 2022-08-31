#
for SEED in 100 106  ; do
    python train_meld.py --active-listener --attention simple --dropout 0.25 --rec-dropout 0.3 --lr 0.0001 --mode1 2 --classify emotion --mu 0 --l2 0.00003 --epochs 60 --seed $SEED
done
#
for SEED in 100 106  ; do
    python train_meld.py --active-listener --attention simple --dropout 0.3 --rec-dropout 0.3 --lr 0.0001 --mode1 2 --classify emotion --mu 0 --l2 0.00003 --epochs 60 --seed $SEED
done
