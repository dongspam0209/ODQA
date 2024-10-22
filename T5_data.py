from datasets import load_from_disk
from datasets import load_metric
from transformers import EvalPrediction
import numpy as np

class T5DataModule:
    def __init__(self, data_args, training_args, tokenizer):
        self.data_args = data_args
        self.training_args = training_args
        self.tokenizer = tokenizer
        self.datasets = load_from_disk('./resources/data/data_kosquadv1_train_dataset')
        #self.datasets = load_from_disk('./resources/data/train_dataset')
        if training_args.do_train:
            self.column_names = self.datasets["train"].column_names
        else:
            self.column_names = self.datasets["validation"].column_names
        self.metric = load_metric("squad")
    
    def _prepare_features(self, examples):
    # truncation과 padding(length가 짧을때만)을 통해 toknization을 진행하며, stride를 이용하여 overflow를 유지합니다.
    # 각 example들은 이전의 context와 조금씩 겹치게됩니다.
        inputs = [f'question: {q}  context: {c}' for q, c in zip(examples['question'], examples['context'])]
        targets = [f'{a["text"][0]}' for a in examples['answers']]
        '''tokenized_examples = self.tokenizer(
            text=examples['question'],
            text_pair=examples['context'],
            text_target=labels,
            padding="max_length",
            truncation="only_second",    # context만 truncate하기
            max_length=384,
            stride=128, # 이전 chunk와 overlap되는 부분을 두어 긴 문서를 처리할 때 유용. 모델이 더 나은 임베딩을 하도록 도와 QA에 유용.
            return_overflowing_tokens=True,
        )'''

        tokenized_examples = self.tokenizer(
            text=inputs,
            text_target=targets,
            padding="max_length",
            truncation=True,
            max_length=512,
            stride=128, # 이전 chunk와 overlap되는 부분을 두어 긴 문서를 처리할 때 유용. 모델이 더 나은 임베딩을 하도록 도와 QA에 유용.
            return_overflowing_tokens=True,
            return_tensors='pt',
        )
        # Setup the tokenizer for targets
        tokenized_examples["labels"][tokenized_examples["labels"] == self.tokenizer.pad_token_id] = -100
        
        # labels 확장
        sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")
        tokenized_examples["labels"] = [tokenized_examples['labels'][i] for i in sample_mapping]

        return tokenized_examples
    
    def get_processing_data(self):
        train_dataset = self.datasets['train']
        train_dataset = train_dataset.shuffle(seed=104)
        train_dataset = train_dataset.map(
            self._prepare_features,
            batched=True,
            num_proc=self.training_args.dataloader_num_workers,
            remove_columns=train_dataset.column_names,
            load_from_cache_file=not self.data_args.overwrite_cache)

        val_dataset = self.datasets['validation']
        val_dataset = val_dataset.map(
            self._prepare_features,
            batched=True,
            num_proc=self.training_args.dataloader_num_workers,
            remove_columns=val_dataset.column_names,
            load_from_cache_file=not self.data_args.overwrite_cache)
        return train_dataset, val_dataset
    
    # T5
    # def postprocess_text(self, preds, labels):
    #     preds = [pred.strip() for pred in preds]
    #     labels = [label.strip() for label in labels]

    #     preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in preds]
    #     labels = ["\n".join(nltk.sent_tokenize(label)) for label in labels]

    #     return preds, labels
    
    def _post_process_function(self, features, predictions, training_args):
        # BART 모델의 출력 처리
        if isinstance(predictions, tuple):
            predictions = predictions[0]
        if isinstance(features, tuple):
            features = features[0]

        if predictions.ndim == 3:
            predictions = np.argmax(predictions, axis=-1)
        # decoding -> token_ids to text
        refs = features['labels']
        
        if isinstance(refs, list):
            refs = np.array(refs)
        refs[refs==-100] = self.tokenizer.pad_token_id

        preds = self.tokenizer.batch_decode(predictions, skip_special_tokens=True)
        refs = self.tokenizer.batch_decode(refs, skip_special_tokens=True)

        print('preds : ', preds[:5])
        print('refs : ', refs[:5])

        #후처리된 예측 ==> {"id"(예제ID), "prediction_text"(예측답변텍스트)} 딕셔너리 리스트
        #do_predict인 경우 ==> formatted_predictions (inference해야함)
        #do_eval인 경우 ==>  예측, 정답 함께 반환 (f1, em결과 확인용)
        if training_args.do_predict:
            return preds
        elif training_args.do_eval:
            return EvalPrediction(predictions=preds, label_ids=refs)
    