#!/bin/sh
timestamp(){
    date +"%Y-%m-%d %H:%M:%S"
}

n_steps=20001

echo $(timestamp) "python main.py --input-embedding 'one_hot' --train-steps $n_steps --input-grain single --output-grain single" >> progress.out
python main.py --input-embedding one_hot --train-steps $n_steps --input-grain single --output-grain single >> print.out
echo "$(timestamp) ...end" >> progress.out
echo $(timestamp) "python main.py --input-embedding 'one_hot' --train-steps $n_steps --input-grain single --output-grain multi --output-loss-mode max" >> progress.out
python main.py --input-embedding one_hot --train-steps $n_steps --input-grain single --output-grain multi --output-loss-mode max >> print.out
echo "$(timestamp) ...end" >> progress.out
echo $(timestamp) "python main.py --input-embedding 'one_hot' --train-steps $n_steps --input-grain single --output-grain multi --output-loss-mode mean" >> progress.out
python main.py --input-embedding one_hot --train-steps $n_steps --input-grain single --output-grain multi --output-loss-mode mean >> print.out
echo "$(timestamp) ...end" >> progress.out
echo $(timestamp) "python main.py --input-embedding 'one_hot' --train-steps $n_steps --input-grain multi --output-grain single" >> progress.out
python main.py --input-embedding one_hot --train-steps $n_steps --input-grain multi --output-grain single >> print.out
echo "$(timestamp) ...end" >> progress.out
echo $(timestamp) "python main.py --input-embedding 'one_hot' --train-steps $n_steps  --input-grain multi --output-grain multi --output-loss-mode max" >> progress.out
python main.py --input-embedding one_hot --train-steps $n_steps --input-grain multi --output-grain multi --output-loss-mode max >> print.out
echo "$(timestamp) ...end" >> progress.out
echo $(timestamp) "python main.py --input-embedding 'one_hot' --train-steps $n_steps  --input-grain multi --output-grain multi --output-loss-mode mean" >> progress.out
python main.py --input-embedding one_hot --train-steps $n_steps --input-grain multi --output-grain multi --output-loss-mode mean >> print.out
echo "$(timestamp) ...end" >> progress.out