##! /bin/bash

# Note: Don't forget to edit the hyper-parameters for part d.

# Train on the names dataset
python src/run.py finetune wiki.txt \
--writing_params_path model.params \
--finetune_corpus_path birth_places_train.tsv
# Evaluate on the dev set, writing out predictions
python src/run.py evaluate wiki.txt \
--reading_params_path model.params \
--eval_corpus_path birth_dev.tsv \
--outputs_path nopretrain.dev.predictions
# Evaluate on the test set, writing out predictions
python src/run.py evaluate wiki.txt \
--reading_params_path model.params \
--eval_corpus_path birth_test_inputs.tsv \
--outputs_path nopretrain.test.predictions
