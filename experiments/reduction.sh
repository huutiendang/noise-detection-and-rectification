# Variables
for DATA in 'snippets' 'imdb'
do
    for SEED in 16
    do
        for METHOD in 'dot' 'cos'
        do
            for TYPE_DATA in 'random_noise' 'structured_noise' 'concentrated_noise'
            do
                for PERCENT in 5 10 20
                do
                    FOLDER=seed-$SEED-$DATA
                    #Train model with noise full data
                    python3 reduction.py \
                        --model bert \
                        --data $DATA \
                        --feature-method $METHOD \
                        --save-path data/$DATA \
                        --df-train data/$DATA/$METHOD-rectified-$TYPE_DATA-$PERCENT%.csv \
                        --seed $SEED;
                done
            done
        done
    done
done
