import os
import sys
from datasets import load_dataset
from functools import partial
import torch
from transformers import (
    AutoTokenizer,
    AutoConfig,
    BlipForQuestionAnswering,
    Trainer,
    TrainingArguments,
)
import fire
from torchvision.io import ImageReadMode, read_image
from data.transformation import ImageTransform
from data.collactor import Blip1Collactor


def main(
    model_name: str = "Salesforce/blip-vqa-capfilt-large",  # Salesforce/blip-vqa-base, Salesforce/blip-vqa-capfilt-large
    dataset_path: str = "./dataset/preprocessed_train.csv",
    image_size: int = 384,
    val_set_size: float = 0.2,
    output_dir: str = "./blip1_output",
    num_epochs: int = 10,
    learning_rate: float = 2e-5,
    global_batch_size: int = 256,
    per_device_batch_size: int = 8,
    save_total_limit: int = 10,
    eval_steps: int = 200,
    device_map: str = "auto",
    wandb_run_name: str = "test",
    use_wandb: bool = True,
    wandb_project: str = "dacon-vqa-blip1",
    optim: str = "adamw_torch",
    lr_scheduler_type: str = "cosine",
    fp16: bool = True,
    bf16: bool = False,
    gradient_checkpointing: bool = False,
    resume_from_checkpoint: str = None,
    warmup_steps: int = 1000,
    seed: int = 42,
    **kwargs
):
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
    gradient_accumulation_steps = global_batch_size // per_device_batch_size

    if ddp:
        device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}
        gradient_accumulation_steps = gradient_accumulation_steps // world_size

    if len(wandb_project) > 0:
        os.environ["WANDB_PROJECT"] = wandb_project

    config = AutoConfig.from_pretrained(model_name)
    model = BlipForQuestionAnswering.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if bf16 else torch.float32,
        device_map=device_map,
        config=config,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    blip_collactor = Blip1Collactor()

    if not ddp and torch.cuda.device_count() > 1:
        # keeps Trainer from trying its own DataParallelism when more than 1 gpu is available
        model.is_parallelizable = True
        model.model_parallel = True

    data = load_dataset("csv", data_files=dataset_path)

    def tokenize_captions(examples):
        captions = [caption for caption in examples["question"]]
        answers = [answer for answer in examples["answer"]]
        text_inputs = tokenizer(
            captions, max_length=32, padding="max_length", truncation=True
        )
        labels = tokenizer(
            answers, max_length=32, padding="max_length", truncation=True
        )
        examples["input_ids"] = text_inputs.input_ids
        examples["attention_mask"] = text_inputs.attention_mask
        examples["labels"] = labels.input_ids
        return examples

    column_names = data["train"].column_names
    train_image_transformation = ImageTransform(img_size=image_size)
    test_image_transformation = ImageTransform(img_size=image_size, is_train=False)

    def transform_images(transformation, examples):
        images = [
            read_image(image_file, mode=ImageReadMode.RGB)
            for image_file in examples["image_path"]
        ]
        examples["pixel_values"] = [transformation(image) for image in images]
        return examples

    if val_set_size > 0.0:
        data = (
            data["train"]
            .train_test_split(test_size=val_set_size, shuffle=True, seed=seed)
            .map(
                function=tokenize_captions,
                batched=True,
                load_from_cache_file=False,
                remove_columns=[col for col in column_names if col != "image_path"],
                num_proc=2,
                desc="Running tokenizer on train and validation dataset",
            )
        )
        data["train"].set_transform(
            partial(transform_images, train_image_transformation)
        )
        data["test"].set_transform(partial(transform_images, test_image_transformation))
    else:
        data = data.shuffle(seed=seed).map(
            function=tokenize_captions,
            batched=True,
            remove_columns=[col for col in column_names if col != "image_path"],
            load_from_cache_file=False,
            num_proc=2,
            desc="Running tokenizer on train dataset",
        )
        data.set_transform(partial(transform_images, train_image_transformation))

    training_args = TrainingArguments(
        per_device_train_batch_size=per_device_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        gradient_checkpointing=gradient_checkpointing,
        warmup_steps=warmup_steps,
        num_train_epochs=num_epochs,
        learning_rate=learning_rate,
        fp16=fp16,
        bf16=bf16,
        logging_steps=10,
        optim=optim,
        lr_scheduler_type=lr_scheduler_type,
        evaluation_strategy="epoch" if val_set_size > 0 else "no",
        save_strategy="epoch",
        eval_steps=eval_steps if val_set_size > 0 else None,
        save_steps=eval_steps,
        output_dir=output_dir,
        save_total_limit=save_total_limit,
        load_best_model_at_end=True if val_set_size > 0 else False,
        report_to="wandb" if use_wandb else None,
        run_name=wandb_run_name if use_wandb else None,
        remove_unused_columns=False,
        seed=seed,
        **kwargs
    )

    trainer = Trainer(
        model=model,
        train_dataset=data["train"],
        eval_dataset=data["test"] if val_set_size > 0.0 else None,
        args=training_args,
        data_collator=blip_collactor,
    )

    if torch.__version__ >= "2" and sys.platform != "win32":
        model = torch.compile(model)

    trainer.train()

    model.save_pretrained(output_dir)


if __name__ == "__main__":
    fire.Fire(main)
