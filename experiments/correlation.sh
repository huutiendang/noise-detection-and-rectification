# Variables
MODEL=bert
for DATA in 'snippets'
do
    for SEED in 16
    do
        for TYPE_DATA in 'random_noise'
        do
            for PERCENT in 20
            do
                FOLDER=seed-$SEED-$DATA
                # Run feature-based
                python3 correlation.py \
                    --model $MODEL \
                    --data $DATA \
                    --n-sample 1000 \
                    --k 100 \
                    --dir-checkpoint checkpoints/$DATA/$FOLDER/$TYPE_DATA-$PERCENT \
                    --df-noise data/$DATA/$TYPE_DATA-$PERCENT%.csv \
                    --df-clean data/$DATA/val.csv \
                    --seed $SEED;
            done
        done
    done
done
