#!/bin/sh
timestamp(){
    date +"%Y-%m-%d %H:%M:%S"
}

run_all=false
n_steps=10001
echo "Testing model DKT on different sets of hyper-parameters......"

if $run_all; then
    echo "testing all the possible combinations of the parametsrs"
    for granularity in 'single' 'multi'; do
        for granularity_out in 'single' 'multi'; do
            for batch_size in 16 32 64; do
                for n_hidden_units in 100 200 300; do
                    for keep_prob in 0.5 0.7 0.9; do
                        for initial_learning_rate in 0.0008 0.001 0.01 0.1; do
                            for final_learning_rate in 0.0001 0.00001 0.000001; do
                                echo $(timestamp) "python main.py --batch-size $batch_size --train-steps $n_steps --num-hiddenunits $n_hidden_units --droupout-keep $keep_prob --learningrate-init $initial_learning_rate --learningrate-final $final_learning_rate --input-embedding 'one_hot' --input-grain $granularity" >> progress.out
                                python main.py --batch-size $batch_size --train-steps $n_steps --num-hiddenunits $n_hidden_units --droupout-keep $keep_prob --learningrate-init $initial_learning_rate --learningrate-final $final_learning_rate --input-embedding 'one_hot' --input-grain $granularity --output-grain $granularity_out >> print.out
                                echo "$(timestamp) ...end" >> progress.out
                            done
                        done
                    done
                done
            done
        done
    done
    for granularity in 'single' 'multi'; do
        for granularity_out in 'single' 'multi'; do
            for embedding_size in 50 100 200; do
                for batch_size in 16 32 64; do
                    for n_hidden_units in 100 200 300; do
                        for keep_prob in 0.5 0.7 0.9; do
                            for initial_learning_rate in 0.0008 0.001 0.01 0.1; do
                                for final_learning_rate in 0.0001 0.00001 0.000001; do
                                    echo $(timestamp) "python main.py --batch-size $batch_size --train-steps $n_steps --num-hiddenunits $n_hidden_units --embedding-size $embedding_size --droupout-keep $keep_prob --learningrate-init $initial_learning_rate --learningrate-final $final_learning_rate --input-embedding 'random' --input-grain $granularity" >> progress.out
                                    python main.py --batch-size $batch_size --train-steps $n_steps --num-hiddenunits $n_hidden_units --embedding-size $embedding_size --droupout-keep $keep_prob --learningrate-init $initial_learning_rate --learningrate-final $final_learning_rate --input-embedding 'random' --input-grain $granularity --output-grain $granularity_out >> print.out
                                    echo "$(timestamp) ...end" >> progress.out
                                done
                            done
                        done
                    done
                done
            done
        done
    done
else
    echo "testing the parameters' influences one by one"
    for granularity in 'single' 'multi'; do
        for granularity_out in 'single' 'multi'; do
            for batch_size in 16 32 64; do
                echo $(timestamp) "python main.py --input-embedding 'one_hot' --train-steps $n_steps --batch-size $batch_size" >> progress.out
                python main.py --input-embedding 'one_hot' --train-steps $n_steps --batch-size $batch_size >> print.out
                echo "$(timestamp) ...end" >> progress.out
            done
            for n_hidden_units in 100 200 300; do
                echo $(timestamp) "python main.py --input-embedding 'one_hot' --train-steps $n_steps --num-hiddenunits $n_hidden_units" >> progress.out
                python main.py --input-embedding 'one_hot' --train-steps $n_steps --num-hiddenunits $n_hidden_units >> print.out
                echo "$(timestamp) ...end" >> progress.out
            done
            for keep_prob in 0.5 0.7 0.9; do
                echo $(timestamp) "python main.py --input-embedding 'one_hot' --train-steps $n_steps --droupout-keep $keep_prob" >> progress.out
                python main.py --input-embedding 'one_hot' --train-steps $n_steps --droupout-keep $keep_prob >> print.out
                echo "$(timestamp) ...end" >> progress.out
            done
            for initial_learning_rate in 0.0008 0.001 0.01 0.1; do
                echo $(timestamp) "python main.py --input-embedding 'one_hot' --train-steps $n_steps --learningrate-init $initial_learning_rate" >> progress.out
                python main.py --input-embedding 'one_hot' --train-steps $n_steps --learningrate-init $initial_learning_rate >> print.out
                echo "$(timestamp) ...end" >> progress.out
            done
            for final_learning_rate in 0.0001 0.00001 0.000001; do
                echo $(timestamp) "python main.py --input-embedding 'one_hot' --train-steps $n_steps --learningrate-final $final_learning_rate" >> progress.out
                python main.py --input-embedding 'one_hot' --train-steps $n_steps --learningrate-final $final_learning_rate >> print.out
                echo "$(timestamp) ...end" >> progress.out
            done                
        done
    done
    for embedding_size in 50 100 200; do
        for granularity in 'single' 'multi'; do
            for granularity_out in 'single' 'multi'; do
                for batch_size in 16 32 64; do
                    echo $(timestamp) "python main.py --input-embedding 'random' --embedding-size $embedding_size --train-steps $n_steps --batch-size $batch_size" >> progress.out
                    python main.py --input-embedding 'random' --embedding-size $embedding_size --train-steps $n_steps --batch-size $batch_size >> print.out
                    echo "$(timestamp) ...end" >> progress.out
                done
                for n_hidden_units in 100 200 300; do
                    echo $(timestamp) "python main.py --input-embedding 'random' --embedding-size $embedding_size --train-steps $n_steps --num-hiddenunits $n_hidden_units" >> progress.out
                    python main.py --input-embedding 'random' --embedding-size $embedding_size --train-steps $n_steps --num-hiddenunits $n_hidden_units >> print.out
                    echo "$(timestamp) ...end" >> progress.out
                done
                for keep_prob in 0.5 0.7 0.9; do
                    echo $(timestamp) "python main.py --input-embedding 'random' --embedding-size $embedding_size --train-steps $n_steps --droupout-keep $keep_prob" >> progress.out
                    python main.py --input-embedding 'random' --embedding-size $embedding_size --train-steps $n_steps --droupout-keep $keep_prob >> print.out
                    echo "$(timestamp) ...end" >> progress.out
                done
                for initial_learning_rate in 0.0008 0.001 0.01 0.1; do
                    echo $(timestamp) "python main.py --input-embedding 'random' --embedding-size $embedding_size --train-steps $n_steps --learningrate-init $initial_learning_rate" >> progress.out
                    python main.py --input-embedding 'random' --embedding-size $embedding_size --train-steps $n_steps --learningrate-init $initial_learning_rate >> print.out
                    echo "$(timestamp) ...end" >> progress.out
                done
                for final_learning_rate in 0.0001 0.00001 0.000001; do
                    echo $(timestamp) "python main.py --input-embedding 'random' --embedding-size $embedding_size --train-steps $n_steps --learningrate-final $final_learning_rate" >> progress.out
                    python main.py --input-embedding 'random' --embedding-size $embedding_size --train-steps $n_steps --learningrate-final $final_learning_rate >> print.out
                    echo "$(timestamp) ...end" >> progress.out
                done                
            done
        done
    done
fi