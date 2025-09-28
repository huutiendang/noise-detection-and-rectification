for DATA in 'snippets' 'imdb'
do
    for SEED in 16 32 64 128
    do
        for TYPE_DATA in 'random_noise' 'structured_noise' 'concentrated_noise'
        do
            for PERCENT in 5 10 20
            do
                FOLDER=seed-$SEED-$DATA
                #Train model with noise full data
                python train.py \
                    --model bert \
                    --data $DATA \
                    --df-train data/$DATA/$TYPE_DATA-$PERCENT%.csv \
                    --df-val data/$DATA/val.csv \
                    --dir-checkpoint checkpoints/$DATA/$FOLDER/$TYPE_DATA-$PERCENT \
                    --seed $SEED \
                    --save-each-epoch;
            done
        done
    done
done
