""" Script for training a Roberta Masked-Language Model for DeepSMILES

Usage [BPE tokenizer (DeepSMILES)]:
    python train_roberta_mlm.py --dataset_path=<DATASET_PATH> --output_dir=<OUTPUT_DIR> --run_name=<RUN_NAME> --tokenizer_type=bpe
"""
import os
from absl import app
from absl import flags

import transformers
from tokenizers.implementations import ByteLevelBPETokenizer
from transformers import EarlyStoppingCallback

import torch
from torch.utils.data import random_split

import wandb
from transformers import RobertaConfig
from transformers import RobertaTokenizerFast
from transformers import RobertaForMaskedLM

from transformers import DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments

FLAGS = flags.FLAGS

# RobertaConfig params
flags.DEFINE_integer(name="vocab_size", default=767, help="")   #
flags.DEFINE_integer(name="max_position_embeddings", default=804,
                     help="")  # This needs to be longer than max_tokenizer_len. max_len is currently 514 in seyonec/SMILES_tokenized_PubChem_shard00_160k
flags.DEFINE_integer(name="num_attention_heads", default=12, help="")
flags.DEFINE_integer(name="num_hidden_layers", default=6, help="")
flags.DEFINE_integer(name="type_vocab_size", default=1, help="")
flags.DEFINE_bool(name="fp16", default=True, help="Mixed precision.")

# Tokenizer params
flags.DEFINE_enum(name="tokenizer_type", default="smiles", enum_values=["smiles", "bpe", "SMILES", "BPE"], help="")
flags.DEFINE_string(name="tokenizer_path", default="", help="")
flags.DEFINE_integer(name="BPE_min_frequency", default=2, help="")
flags.DEFINE_string(name="output_tokenizer_dir", default="tokenizer_dir", help="")
flags.DEFINE_integer(name="max_tokenizer_len", default=802, help="")
flags.DEFINE_integer(name="tokenizer_block_size", default=802, help="")

# Dataset params
flags.DEFINE_string(name="dataset_path", default=None, help="")
flags.DEFINE_string(name="output_dir", default="default_dir", help="")
flags.DEFINE_string(name="run_name", default="default_run", help="")

# MLM params
flags.DEFINE_float(name="mlm_probability", default=0.15, lower_bound=0.0, upper_bound=1.0, help="")

# Train params
flags.DEFINE_float(name="frac_train", default=0.95, help="")
flags.DEFINE_integer(name="eval_steps", default=500, help="")
flags.DEFINE_integer(name="logging_steps", default=100, help="")
flags.DEFINE_boolean(name="overwrite_output_dir", default=True, help="")
flags.DEFINE_integer(name="num_train_epochs", default=10, help="")
flags.DEFINE_integer(name="per_device_train_batch_size", default=4, help="")
flags.DEFINE_integer(name="save_steps", default=1000, help="")
flags.DEFINE_integer(name="save_total_limit", default=2 , help="")

flags.mark_flag_as_required("dataset_path")

from datasets import Dataset


def load_text_dataset(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        # Split into lines, stripping any extra whitespace
        lines = [line.strip() for line in f if line.strip()]
    # Create a dataset with a single column "text"
    dataset = Dataset.from_dict({"text": lines})
    return dataset


def tokenize_function(examples):
    #tokenizer = RobertaTokenizerFast.from_pretrained(tokenizer_path = FLAGS.tokenizer_path, max_len=FLAGS.max_tokenizer_len)

    #tokenizer = AutoTokenizer.from_pretrained("seyonec/SMILES_tokenized_PubChem_shard00_160k")
    #    import os
    #os.environ["HF_HUB_OFFLINE"] = "1"

    #tokenizer = RobertaTokenizerFast.from_pretrained("roberta-base")

    tokenizer = RobertaTokenizerFast.from_pretrained(FLAGS.output_tokenizer_dir)
    return tokenizer(
        examples["text"],
        truncation=True,
        max_length=FLAGS.tokenizer_block_size,
        padding="max_length",  # or you could omit this if you want dynamic padding later,
        #special_tokens = ["<s>", "</s>", "<pad>", "<mask>", "<unk>"]
    )


# Assuming your dataset is loaded with a "text" column:

def main(argv):
    torch.manual_seed(0)

    wandb.login()

    is_gpu = torch.cuda.is_available()

    config = RobertaConfig(
        vocab_size=FLAGS.vocab_size,
        max_position_embeddings=FLAGS.max_position_embeddings,
        num_attention_heads=FLAGS.num_attention_heads,
        num_hidden_layers=FLAGS.num_hidden_layers,
        type_vocab_size=FLAGS.type_vocab_size,
    )

    if FLAGS.tokenizer_path:
        tokenizer_path = FLAGS.tokenizer_path
    elif FLAGS.tokenizer_type.upper() == "BPE":
        tokenizer_path = FLAGS.output_tokenizer_dir
        if not os.path.isdir(tokenizer_path):
            os.makedirs(tokenizer_path)

        bpe_tokenizer = ByteLevelBPETokenizer()
        #tokenizer.mask_token = "<mask>"
        bpe_tokenizer.train(files=FLAGS.dataset_path, vocab_size=FLAGS.vocab_size,
                            min_frequency=FLAGS.BPE_min_frequency,
                            special_tokens=["<s>", "</s>", "<pad>", "<mask>", "<unk>"])
        bpe_tokenizer.save_model(tokenizer_path)
        #tokenizer_json = tokenizer.model.to_str()
        #with open(os.path.join(tokenizer_path, "tokenizer.json"), "w", encoding="utf-8") as f:
        #    f.write(tokenizer_json)

        from transformers import RobertaTokenizerFast
        #tokenizer_path = "/home/sunkot/frag-ml-project/tokenizer_dir"
        #tokenizer.save_pretrained(tokenizer_path)

        tokenizer = RobertaTokenizerFast(
            tokenizer_object=bpe_tokenizer,
            unk_token="<unk>",
            pad_token="<pad>",
            cls_token="<s>",
            sep_token="</s>",
            mask_token="<mask>",
        )
        tokenizer.save_pretrained(tokenizer_path)
        #from transformers import PreTrainedTokenizerFast
        #tokenizer = PreTrainedTokenizerFast.from_pretrained(tokenizer_path)


    else:
        print("Please provide a tokenizer path if using the SMILES tokenizer")

    model = RobertaForMaskedLM(config=config)
    print(f"Model size: {model.num_parameters()} parameters.")

    #dataset = RawTextDataset(tokenizer=tokenizer, file_path=FLAGS.dataset_path, block_size=FLAGS.tokenizer_block_size)
    dataset = load_text_dataset(FLAGS.dataset_path)
    dataset = dataset.map(tokenize_function, batched=True, remove_columns=["text"])
    train_size = max(int(FLAGS.frac_train * len(dataset)), 1)
    eval_size = len(dataset) - train_size
    print(f"Train size: {train_size}")
    print(f"Eval size: {eval_size}")

    train_dataset, eval_dataset = random_split(dataset, [train_size, eval_size])

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=True, mlm_probability=FLAGS.mlm_probability
    )

    training_args = TrainingArguments(
        evaluation_strategy="steps",
        eval_steps=FLAGS.eval_steps,
        load_best_model_at_end=True,
        logging_steps=FLAGS.logging_steps,
        output_dir=os.path.join(FLAGS.output_dir, FLAGS.run_name),
        overwrite_output_dir=FLAGS.overwrite_output_dir,
        num_train_epochs=FLAGS.num_train_epochs,
        per_device_train_batch_size=FLAGS.per_device_train_batch_size,
        save_steps=FLAGS.save_steps,
        save_total_limit=FLAGS.save_total_limit,
        fp16=is_gpu and FLAGS.fp16,  # fp16 only works on CUDA devices
        report_to="wandb",
        run_name=FLAGS.run_name,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    )

    trainer.train()
    trainer.save_model(os.path.join(FLAGS.output_dir, FLAGS.run_name, "final"))


if __name__ == '__main__':
    app.run(main)

