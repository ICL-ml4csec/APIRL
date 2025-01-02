import json
import re
import heapq
from transformers import Trainer, TrainingArguments, DataCollatorForLanguageModeling, LineByLineTextDataset, \
    RobertaTokenizer, RobertaConfig, RobertaModel, RobertaTokenizerFast, RobertaForMaskedLM, AlbertConfig, AlbertTokenizer, AlbertForMaskedLM, AlbertTokenizerFast
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers import Tokenizer, pre_tokenizers, ByteLevelBPETokenizer, SentencePieceBPETokenizer
from tokenizers.implementations import ByteLevelBPETokenizer
from tokenizers.processors import RobertaProcessing
from tokenizers.models import  BPE
from tokenizers.normalizers import Lowercase, NFKC, Sequence
import pickle
import pandas as pd
from transformers import pipeline
from os.path import isfile, join
from pathlib import Path

def train_tokenizer(paths):
    # Train the tokeniser

    tokenizer = SentencePieceBPETokenizer()
    tokenizer.add_special_tokens(["<s>", "<pad>", "</s>", "<unk>", "<mask>"])
    tokenizer.normalizer = Sequence([NFKC(), Lowercase()])
    tokenizer.train(files=paths, vocab_size=52_000, min_frequency=2, special_tokens=[
        "<s>",
        "<pad>",
        "</s>",
        "<unk>",
        "<mask>",
    ])
    return tokenizer

def compile_dataset(data_dir, save_file):
    paths = [str(x) for x in Path(data_dir).glob("*.txt")]
    for path in paths:
        with open(path, 'r') as f:
            for line in f.readlines():
                with open(save_file, 'a') as api_f:
                    api_f.write(line)
    return paths


def train_RoBERTa(model_dir, data_file, vocab_size=52_000, max_position_embeddings=512,
                 num_attention_heads=12,
                 num_hidden_layers=6,
                 type_vocab_size=1,
                 hidden_size=768):
    # Train RoBERTa
    config = RobertaConfig(
        vocab_size=vocab_size,
        max_position_embeddings=max_position_embeddings,
        num_attention_heads=num_attention_heads,
        num_hidden_layers=num_hidden_layers,
        type_vocab_size=type_vocab_size,
        hidden_size=hidden_size
        )
    tokenizer = RobertaTokenizerFast.from_pretrained(model_dir, max_len=750)
    model = RobertaForMaskedLM(config=config)

    dataset = LineByLineTextDataset(
        tokenizer=tokenizer,
        file_path=data_file,
        block_size=128,
    )

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=True, mlm_probability=0.15
    )
    training_args = TrainingArguments(
        output_dir="./logs",
        overwrite_output_dir=True,
        num_train_epochs=10,
        logging_strategy='steps',
        logging_steps=1,
        per_gpu_train_batch_size=64,
        save_steps=1000,
        save_total_limit=2,
        prediction_loss_only=True,
        no_cuda=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=dataset,
    )
    trainer.train()
    trainer.save_model(model_dir)

if __name__ == '__main__':

    dataset_file = "./api_dataset.txt"
    model_dir = "./api_transformer"
    paths = compile_dataset("", dataset_file)
    train_RoBERTa(model_dir, dataset_file)


