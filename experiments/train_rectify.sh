# Variables
for DATA in 'imdb' 'snippets'
do
    for SEED in 16 32 64 128
    do
        for METHOD in 'cos' 'dot'
        do
            for TYPE_DATA in 'random_noise' 'structured_noise' 'concentrated_noise'
            do
                for PERCENT in 5 10 20
                do
                    FOLDER=seed-$SEED-$DATA
                    #Train model with noise full data
                    python3 train.py \
                        --model bert \
                        --data $DATA \
                        --df-train data/$DATA/$METHOD-rectified-$TYPE_DATA-$PERCENT%.csv \
                        --df-val data/$DATA/val.csv \
                        --dir-checkpoint checkpoints_rectify/$DATA/$FOLDER/$METHOD-$TYPE_DATA-$PERCENT \
                        --seed $SEED;
                done
            done
        done
    done
done
