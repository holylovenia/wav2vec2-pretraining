from dataclasses import dataclass, field
from typing import Optional
from transformers import TrainingArguments

@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to utilize.
    """
    model_name_or_path: Optional[str] = field(
        default="CAiRE/wav2vec2-large-xlsr-53-cantonese", metadata={"help": "The path of the HuggingFace model."}
    )
    mask_time_prob: float = field(
        default=0.065,
        metadata={
            "help": "Probability of each feature vector along the time axis to be chosen as the start of the vector"
            "span to be masked. Approximately ``mask_time_prob * sequence_length // mask_time_length`` feature"
            "vectors will be masked along the time axis."
        },
    )
    mask_time_length: float = field(
        default=10,
        metadata={
            "help": "Length of vector span along the time axis."
        },
    )
    mask_time_min_masks: int = field(
        default=2,
        metadata={"help": "The minimum number of masks of length mask_feature_length generated along the time axis, each time step, irrespectively of mask_feature_prob. Only relevant if ”mask_time_prob*len(time_axis)/mask_time_length < mask_time_min_masks”."},
    )
    mask_feature_prob: float = field(
        default=0.004,
        metadata={
            "help": "Probability of each feature vector along the feature axis to be chosen as the start of the vector"
            "span to be masked. Approximately ``mask_feature_prob * sequence_length // mask_feature_length`` feature bins will be masked along the time axis."
        },
    )
    mask_feature_length: float = field(
        default=10,
        metadata={
            "help": "Length of vector span along the feature axis."
        },
    )
    mask_feature_min_masks: int = field(
        default=2,
        metadata={"help": "The minimum number of masks of length mask_feature_length generated along the feature axis, each time step, irrespectively of mask_feature_prob. Only relevant if ”mask_feature_prob*len(feature_axis)/mask_feature_length < mask_feature_min_masks”."},
    )

@dataclass
class DataArguments:
    """
    Arguments pertaining to the data loading and preprocessing pipeline.
    """
    train_manifest_path: Optional[str] = field(
        default="data/common_voice_zh-HK/preprocessed_validated_train.csv", metadata={"help": "The path of the training dataset to use."}
    )
    valid_manifest_path: Optional[str] = field(
        default="data/common_voice_zh-HK/preprocessed_validated_dev.csv", metadata={"help": "The path of the validation dataset to use."}
    )
    test_manifest_path: Optional[str] = field(
        default="data/common_voice_zh-HK/preprocessed_validated_test.csv", metadata={"help": "The path of the testing dataset to use."}
    )
    preprocessing_num_workers: Optional[int] = field(
        default=16,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    preprocessing_only: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether to only run preprocessing."},
    )
    audio_column_name: Optional[str] = field(
        default="audio_path",
        metadata={"help": "The name of the dataset column containing the audio path. Defaults to 'audio_path'"},
    )
    text_column_name: Optional[str] = field(
        default="text_path",
        metadata={"help": "The name of the dataset column containing the text data. Defaults to 'text_path'"},
    )
    cache_dir_name: Optional[str] = field(
        default="cache",
        metadata={"help": "Name of cache directory"},
    )
    cache_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path of cache directory. Overrides cache_dir_name settings."}
    )

@dataclass
class TrainingArguments(TrainingArguments):
    """
    Arguments pertraining to the training pipeline.
    """
    output_dir: Optional[str] = field(
        default="./save",
        metadata={"help": "Output directory"},
    )
    output_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path of output directory. Overrides output_dir settings."}
    )
    eval_accumulation_steps: Optional[int] = field(
        default=1,
        metadata={"help": "Evaluation accumulation steps"}
    )

@dataclass
class AdditionalTrainingArguments:
    """
    Additional training arguments
    """
    early_stopping_patience: Optional[int] = field(
        default=5,
        metadata={"help": "Early stopping patience."},
    )