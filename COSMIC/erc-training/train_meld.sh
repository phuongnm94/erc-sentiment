#
for SEED in 100 101 102 103 104 105 106 107 108 109 ; do
    python train_meld.py --active-listener --attention simple --dropout 0.5 --rec-dropout 0.3 --lr 0.0001 --mode1 2 --classify emotion --mu 0 --l2 0.00003 --epochs 60 --seed $SEED --no-self-attn-emotions
done
