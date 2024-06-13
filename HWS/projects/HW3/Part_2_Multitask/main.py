import logging

import nlp
import torch
import nltk
import numpy as np
from datasets import load_dataset, load_metric
from torch.utils.data.dataloader import DataLoader
from tqdm.auto import tqdm
from tqdm import tqdm as tqdm1

import transformers
from accelerate import Accelerator
from filelock import FileLock
from transformers import set_seed
from transformers.file_utils import is_offline_mode
from arguments import parse_args
from multitask_model import MultitaskModel
from preprocess import *
from multitask_data_collator import MultitaskTrainer, NLPDataCollator, DataLoaderWithTaskname
from checkpoint_model import save_model
from pathlib import Path
# from transformers import logging
# logging.set_verbosity_error()


logger = logging.getLogger(__name__)


def main():
    args = parse_args()

    dataset_dict = {
                "rte": load_dataset('glue', name="rte"),
                "stsb": load_dataset('glue', name="stsb"),
                "commonsense_qa": load_dataset('commonsense_qa'),
            }

    # Display examples from each dataset
    for task_name, dataset in dataset_dict.items():
        print(task_name)
        print(dataset_dict[task_name]["train"][0])
        print()

    
    # Here you define your Multi-task model
    # Complete it in multitask_model.py
    model_name = args.model_name_or_path
    multitask_model = MultitaskModel.create(
        model_name=model_name,
        model_type_dict={
            "rte": transformers.AutoModelForSequenceClassification,
            "stsb": transformers.AutoModelForSequenceClassification,
            "commonsense_qa": transformers.AutoModelForMultipleChoice,
        },
        model_config_dict={
            "rte": transformers.AutoConfig.from_pretrained(model_name, num_labels=2),
            "stsb": transformers.AutoConfig.from_pretrained(model_name, num_labels=1),
            "commonsense_qa": transformers.AutoConfig.from_pretrained(model_name),
        },
    )

    # To confirm that all three task-models use the same encoder, we can check the data pointers of the respective encoders.
    # In this case, we'll check that the word embeddings in each model all point to the same memory location.
    if model_name.startswith("roberta-"):
        print("Data pointers:")
        print(multitask_model.encoder.embeddings.word_embeddings.weight.data_ptr())
        print(multitask_model.taskmodels_dict["stsb"].roberta.embeddings.word_embeddings.weight.data_ptr())
        print(multitask_model.taskmodels_dict["rte"].roberta.embeddings.word_embeddings.weight.data_ptr())
        print(multitask_model.taskmodels_dict["commonsense_qa"].roberta.embeddings.word_embeddings.weight.data_ptr())
   
    # Complete these functions in preprocess.py
    convert_func_dict = {
        "stsb": convert_to_stsb_features,
        "rte": convert_to_rte_features,
        "commonsense_qa": convert_to_commonsense_qa_features,
    }

    columns_dict = {
        "stsb": ['input_ids', 'attention_mask', 'labels'],
        "rte": ['input_ids', 'attention_mask', 'labels'],
        "commonsense_qa": ['input_ids', 'attention_mask', 'labels'],
    }

    
    # We can use dataset.map method available in the datasets library to apply the preprocessing functions over 
    # our entire datasets. The datasets library handles the mapping efficiently and caches the features.
    # Select columns and set to pytorch tensors
    features_dict = {}
    for task_name, dataset in dataset_dict.items():
        features_dict[task_name] = {}
        for phase, phase_dataset in dataset.items():
            features_dict[task_name][phase] = phase_dataset.map(
                convert_func_dict[task_name],
                batched=True,
                load_from_cache_file=True,
            )
            features_dict[task_name][phase].set_format(
                type="torch", 
                columns=columns_dict[task_name],
            )
            print(task_name, phase, "num samples:", len(phase_dataset), len(features_dict[task_name][phase]))
    

    train_dataset = {
        task_name: dataset["train"] for task_name, dataset in features_dict.items()
    }

    # Complete the NLPDataCollator class in multitask_data_collator.py
    trainer = MultitaskTrainer(
        model=multitask_model,
        args=transformers.TrainingArguments(
            output_dir=args.output_dir,
            overwrite_output_dir=True,
            learning_rate=args.learning_rate,
            do_train=True,
            num_train_epochs=args.num_train_epochs,
            # Adjust batch size if this doesn't fit on the GPU
            per_device_train_batch_size=args.train_batch_size,  
            save_steps=3000
        ),
        data_collator=NLPDataCollator(),
        train_dataset=train_dataset,
    )
    trainer.train()

    preds_dict = {}
    for task_name in ["rte", "stsb", "commonsense_qa"]:
        
        # Create the corresponding eval dataloaders and evaluate using the builtin method 'prediction loop'
        ################
        # Your code here
        ################
        validation_dataset = {
            task_name: dataset["validation"] for task_name, dataset in features_dict.items()
        }
        eval_dataloader = DataLoaderWithTaskname(task_name, trainer.get_eval_dataloader(validation_dataset))
        preds_dict[task_name] = trainer.prediction_loop(
            eval_dataloader, 
            description=f"Validation: {task_name}",
        )
    
    # Calculate the predictions and labels to calculate the metrics
    #####################
    # Your code here
    #####################
    
    # Evalute RTE
    preds_rte = preds_dict["rte"]["predictions"]
    labels_rte = preds_dict["rte"]["labels"]
    rte_metric = load_metric('glue', "rte")
    rte = rte_metric.compute(
        predictions=preds_rte,
        references=labels_rte,
    )
    print("rte_metrics:\n", rte)

    # Evalute STS-B
    preds_stsb = preds_dict["stsb"]["predictions"]
    labels_stsb = preds_dict["stsb"]["labels"]
    stsb_metric=load_metric('glue', "stsb")
    stsb=stsb_metric.compute(
        predictions=preds_stsb,
        references=labels_stsb,
    )
    print("stsb metrics:\n", stsb)

    # Evalute Commonsense QA, (accuracy)
    preds_qa = preds_dict["commonsense_qa"]["predictions"]
    labels_qa = preds_dict["commonsense_qa"]["labels"]
    qa = np.mean(
        preds_qa
        == labels_qa
    )
    print("qa metrics:\n", qa)
    
    ## save model for later use
    save_model("roberta-base", multitask_model)

if __name__ == "__main__":
    main()
