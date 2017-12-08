#!/bin/sh
timestamp(){
    date +"%Y-%m-%d %H:%M:%S"
}

echo $(timestamp) "python main.py --input-embedding 'one_hot' --train-steps $n_steps --learningrate-init $initial_learning_rate" >> progress.out
python main.py --input-embedding one_hot --input-grain single --output-grain single >> print.out
echo "$(timestamp) ...end" >> progress.out
echo $(timestamp) "python main.py --input-embedding 'one_hot' --train-steps $n_steps --learningrate-init $initial_learning_rate" >> progress.out
python main.py --input-embedding one_hot --input-grain multi --output-grain single >> print.out
echo "$(timestamp) ...end" >> progress.out
echo $(timestamp) "python main.py --input-embedding 'one_hot' --train-steps $n_steps --learningrate-init $initial_learning_rate" >> progress.out
python main.py --input-embedding one_hot --input-grain multi --output-grain multi >> print.out
echo "$(timestamp) ...end" >> progress.out