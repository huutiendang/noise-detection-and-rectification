# Variables
REMOVE=removed
for DATA in 'imdb' 'snippets'
do
    for SEED in 16 32 64 128
    do
        for METHOD in 'dot' 'cos'
        do
            for TYPE_DATA in 'random_noise' 'structured_noise' 'concentrated_noise'
            do
                for PERCENT in 5 10 20
                do
                    FOLDER=seed-$SEED-$DATA
                    #Train model with noise full data
                    python3 test.py \
                        --model bert \
                        --data $DATA \
                        --df-test data/$DATA/test.csv \
                        --dir-checkpoint checkpoints_rectify/$DATA/$FOLDER/$METHOD-$TYPE_DATA-$PERCENT \
                        --seed $SEED;
                done
            done
        done
    done
done
