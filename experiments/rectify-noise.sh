# Variables
MODEL=bert
for DATA in 'snippets' 'imdb'
do
    for SEED in 16 32 64 128
    do
        for TYPE_DATA in 'random_noise' 'structured_noise' 'concentrated_noise'
        do
            for PERCENT in 5 10 20
            do
                FOLDER=seed-$SEED-$DATA
                # Run feature-based
                python3 rectify-noise.py \
                    --model $MODEL \
                    --data $DATA \
                    --n-sample 1000 \
                    --k 100 \
                    --dir-checkpoint checkpoints/$DATA/$FOLDER/$TYPE_DATA-$PERCENT \
                    --save-path data/$DATA \
                    --df-noise data/$DATA/$TYPE_DATA-$PERCENT%.csv \
                    --df-clean data/$DATA/val.csv \
                    --seed $SEED;
            done
        done
    done
done
