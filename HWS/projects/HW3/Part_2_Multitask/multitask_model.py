import torch
import torch.nn as nn
import transformers


class MultitaskModel(transformers.PreTrainedModel):
    def __init__(self, encoder, taskmodels_dict):
        """
        Setting MultitaskModel up as a PretrainedModel allows us
        to take better advantage of Trainer features
        """
        super().__init__(transformers.PretrainedConfig())

        #self.taskmodels_dict = None
        ##########
        # Your code here if needed
        self.encoder = encoder
        self.taskmodels_dict = nn.ModuleDict(taskmodels_dict)

        ##########

    @classmethod
    def create(cls, model_name, model_type_dict, model_config_dict):
        """
        This creates a MultitaskModel using the model class and config objects
        from single-task models. 

        We do this by creating each single-task model, and having them share
        the same encoder transformer.
        """
        taskmodels_dict = {}
        ##########
        # Your Code here
        encoder_ = None
        for task, model_type in model_type_dict.items():
            model_config = model_config_dict[task]

            task_model = model_type.from_pretrained(model_name, config=model_config)

            if encoder_ is None:
                encoder_ = task_model.roberta
            else:
                task_model.roberta = encoder_

            taskmodels_dict[task] = task_model

        multitask_model = cls(encoder_, taskmodels_dict)

        return multitask_model

        ##########

    def forward(self, task_name, **kwargs):
        #print("self.taskmodels_dict[task_name]   ", self.taskmodels_dict[task_name])
        return self.taskmodels_dict[task_name](**kwargs)
