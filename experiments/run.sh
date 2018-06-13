#!/bin/sh

if [[ "$1" == "train" ]]; then
    DATA_DIR=../data/plag_train_data
    TASKS_DIR=$DATA_DIR/tasks
elif [[ $1 == "dialog_train" ]]; then
    DATA_DIR=../data/text_alignment
    TASKS_DIR=$DATA_DIR/tasks/manually-paraphrased
    #TASKS_DIR=$DATA_DIR/tasks/manually_paraphrased2
    #TASKS_DIR=$DATA_DIR/tasks/generated_copypast_meta
    #TASKS_DIR=$DATA_DIR/tasks/generated_paraphrased_meta
    #TASKS_DIR=$DATA_DIR/tasks
elif [[ $1 == "pan_train" ]]; then
    DATA_DIR=../data/pan13-text-alignment-training-corpus-2013-01-21
    #TASKS_DIR=$DATA_DIR/tasks
    #TASKS_DIR=$DATA_DIR/tasks/03-random-obfuscation
    #TASKS_DIR=$DATA_DIR/tasks/04-translation-obfuscation
    TASKS_DIR=$DATA_DIR/tasks/05-summary-obfuscation
elif [[ $1 == "pan_test" ]]; then
    DATA_DIR=../data/pan13-text-alignment-test-corpus2-2013-01-21
    #DATA_DIR=../data/pan13-text-alignment-test-corpus1-2013-03-08
    #TASKS_DIR=$DATA_DIR/tasks/01-no-plagiarism
    #TASKS_DIR=$DATA_DIR/tasks/02-no-obfuscation
    #TASKS_DIR=$DATA_DIR/tasks/03-random-obfuscation
    #TASKS_DIR=$DATA_DIR/tasks/04-translation-obfuscation
    TASKS_DIR=$DATA_DIR/tasks/05-summary-obfuscation
    #TASKS_DIR=$DATA_DIR/05-summary-obfuscation
else
    DATA_DIR=../data/plag_test_data
    TASKS_DIR=$DATA_DIR/tasks
fi

rm -rf predictions/
mkdir predictions/
python3 text_alignment_solution.py $TASKS_DIR/pairs $DATA_DIR/src/ $DATA_DIR/susp/ predictions/ $2
python3 text_alignment_measures_new.py $3 -p $TASKS_DIR/ -d predictions/
