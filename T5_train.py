import logging
import random
import numpy as np
import torch
from T5_data import T5DataModule
from transformers import HfArgumentParser, set_seed, AutoTokenizer, T5ForConditionalGeneration, Seq2SeqTrainer, Seq2SeqTrainingArguments, DataCollatorForSeq2Seq, EarlyStoppingCallback, DataCollatorWithPadding
from utils.arguments_gen_reader import ModelArguments, DataTrainingArguments, OurTrainingArguments
from utils.metric import compute_metrics
from T5_trainer import T5Trainer
import os

logger = logging.getLogger("dpr")
logger.setLevel(logging.INFO)
fmt = logging.Formatter('%(asctime)s: [ %(message)s ]','%m/%d/%Y %I:%M:%S %p')
console = logging.StreamHandler()
console.setFormatter(fmt)
logger.addHandler(console)

seed = 104
random.seed(seed) # python random seed 고정
np.random.seed(seed) # numpy random seed 고정
torch.manual_seed(seed) # torch random seed 고정
torch.cuda.manual_seed_all(seed)

def main():
    logger.info('*** T5 Training ***')
    parser = HfArgumentParser(
        (ModelArguments, DataTrainingArguments, OurTrainingArguments)
    )
    _, data_args, _ = parser.parse_args_into_dataclasses()
    
    # Load Arguments 
    training_args = Seq2SeqTrainingArguments(
    output_dir="./resources/checkpoint/generation/T5", num_train_epochs = 10,
    do_predict=False, do_eval=True, max_grad_norm=1.0,
    save_strategy='steps', evaluation_strategy="steps",
    save_steps=2000, eval_steps=2000, remove_unused_columns=False,
    optim="adamw_hf", prediction_loss_only=False, eval_accumulation_steps=2,
    per_device_train_batch_size=16, per_device_eval_batch_size=4,
    warmup_ratio=0.1, learning_rate=5e-5,save_total_limit = 3, load_best_model_at_end = True,
    dataloader_num_workers=os.cpu_count()//2, metric_for_best_model='eval_exact_match', greater_is_better=True,
    )
    #logger.info(f"Our training arguments: {training_args}")
    
    # 모델을 초기화하기 전에 난수를 고정
    set_seed(training_args.seed)
    tokenizer = AutoTokenizer.from_pretrained("google/mt5-base")
    model = T5ForConditionalGeneration.from_pretrained("google/mt5-base")
    
    # Load Dataset
    dm = T5DataModule(data_args, training_args, tokenizer) 
    train_dataset, valid_dataset = dm.get_processing_data()

    # logger.info(f"Train dataset size: {len(train_dataset)}")
    # logger.info(f"Eval dataset size: {len(valid_dataset)}")
    # logger.info(f"First item in train dataset: {train_dataset[0]}")
    # 배치 단위의 데이터 전처리, 주어진 샘플들을 하나의 배치로 묶는 역할
    #data_collator = DataCollatorForSeq2Seq(tokenizer, model=model, return_tensors="pt")
    data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8 if training_args.fp16 else None)
    # logger.info(model)
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"모델의 전체 파라미터 수 : {total_params}")
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"모델의 학습 가능한 파라미터 수 : {trainable_params}")

    # Trainer 초기화
    trainer = T5Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        post_process_function=dm._post_process_function,
        callbacks = [EarlyStoppingCallback(early_stopping_patience=5)]
    )
    # 모델 학습 및 평가 진행
    torch.cuda.empty_cache()
    #trainer.train()
    metric = trainer.evaluate()
    #print(f"최종 eval_loss : {metric['eval_loss']}, exact match : {metric['eval_exact_match']}, f1 : {metric['eval_f1']}")

if __name__ == "__main__":
    main()