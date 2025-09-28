# Variables
MODEL=bert
for DATA in 'snippets' 'imdb'
do
    for SEED in 16 32 64 128
    do
        for TYPE_DATA in 'random_noise' 'structured_noise' 'concentrated_noise'
        do
            for PERCENT in 20 10 5
            do
                FOLDER=seed-$SEED-$DATA
                # Run feature-based
                python3 stats.py \
                    --data $DATA \
                    --n-sample 1000 \
                    --dir-checkpoint checkpoints/$DATA/$FOLDER/$TYPE_DATA-$PERCENT \
                    --seed $SEED;
            done
        done
    done
done
