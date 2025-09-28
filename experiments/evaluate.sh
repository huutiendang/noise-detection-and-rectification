# Variables
MODEL=bert
for DATA in 'snippets' 'imdb'
do
    for SEED in 16
    do
        for TYPE_DATA in 'random_noise' 'structured_noise' 'concentrated_noise'
        do
            for PERCENT in 20 10 5
            do
                FOLDER=seed-$SEED-$DATA
                # Run feature-based
                # python3 confident-based.py \
                #     --model $MODEL \
                #     --data $DATA \
                #     --dir-checkpoint checkpoints/$DATA/$FOLDER/$TYPE_DATA-$PERCENT \
                #     --df-noise data/$DATA/$TYPE_DATA-$PERCENT%.csv \
                #     --seed $SEED;
                # for GRAD in 'TracIn' 'GD' 'GC' 'IF'
                # do
                #     for N_SAMPLE in 500 1000 1500
                #     do
                #     python3 gradient-base.py \
                #         --model $MODEL \
                #         --data $DATA \
                #         --dir-checkpoint checkpoints/$DATA/$FOLDER/$TYPE_DATA-$PERCENT \
                #         --df-noise data/$DATA/$TYPE_DATA-$PERCENT%.csv \
                #         --df-clean data/$DATA/val.csv \
                #         --n_sample $N_SAMPLE \
                #         --gradient-method $GRAD \
                #         --seed $SEED;
                #     done
                # done
                for SIM in 'cos' 'dot'
                do
                    for N_SAMPLE in 100 200 300 400 500 600 700 800 900 1000
                    do
                        for K in 100
                        do
                        python3 similarity-based.py \
                            --model $MODEL \
                            --data $DATA \
                            --dir-checkpoint checkpoints/$DATA/$FOLDER/$TYPE_DATA-$PERCENT \
                            --df-noise data/$DATA/$TYPE_DATA-$PERCENT%.csv \
                            --df-clean data/$DATA/val.csv \
                            --similarity $SIM \
                            --k $K\
                            --n_sample $N_SAMPLE \
                            --seed $SEED; 
                        done
                    done
                done
            done
        done
    done
done
