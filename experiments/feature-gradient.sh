# Variables
for DATA in 'imdb' 'snippets'
do
    for SEED in 16 32 64 128
    do
        for TYPE_DATA in 'random_noise' 'structured_noise' 'concentrated_noise'
        do
            for PERCENT in 5 10 20
            do
                FOLDER=seed-$SEED-$DATA
                 python feature-based.py \
                     --data $DATA \
                     --model bert \
                     --dir-checkpoint checkpoints/$DATA/$FOLDER/$TYPE_DATA-$PERCENT \
                     --df-train data/$DATA/$TYPE_DATA-$PERCENT%.csv \
                     --df-clean data/$DATA/val.csv \
                     --seed $SEED;
                python gradient-based.py \
                    --data $DATA \
                    --model bert \
                    --dir-checkpoint checkpoints/$DATA/$FOLDER/$TYPE_DATA-$PERCENT \
                    --df-train data/$DATA/$TYPE_DATA-$PERCENT%.csv \
                    --df-clean data/$DATA/val.csv \
                    --seed $SEED; 
            done
        done
    done
done
