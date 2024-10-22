# coding=utf-8
# Copyright 2020 The HuggingFace Team All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Question-Answering task와 관련된 'Trainer'의 subclass 코드 입니다.
"""
import torch
from transformers import Trainer, EvalPrediction

class T5Trainer(Trainer):
    def __init__(self, *args, post_process_function=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.post_process_function = post_process_function

    # 학습 중 또는 학습 후 squad 매트릭으로 평가
    def evaluate(self, ignore_keys=None):
        # eval metric을 return. eval loss와 추가 metric
        eval_dataloader = self.get_eval_dataloader(self.eval_dataset)

        if eval_dataloader.batch_size==None:
            eval_dataloader.batch_size = self.args.per_device_train_batch_size
        # eval loss 쓰려면
        # valid_dataset의 prediction들 뽑기 -> predictions, labels, metrics 각각을 logit 형태로 받음

        output = self.evaluation_loop(
            eval_dataloader,
            description="Evaluation",
            # metric이 없으면 예측값을 모으는 이유가 없으므로 아래의 코드를 따르게 됩니다.
            # self.args.prediction_loss_only
            # metrics가 없으면 loss만 내보내고 있으면 metric까지 함께 내보내는?
            prediction_loss_only=False if self.args.do_eval else True,
            ignore_keys=ignore_keys,
        )

        if self.post_process_function is not None and self.compute_metrics is not None:
            eval_preds = self.post_process_function(
                self.eval_dataset, output.predictions, self.args
            )
            # do_eval일 경우 metric 계산
            if isinstance(eval_preds, EvalPrediction):
                metrics = self.compute_metrics(eval_preds)
            else:   # do_predict일 경우 그대로 값만 내보냄
                return eval_preds
        else:
            metrics = output.metrics
            del metrics['eval_model_preparation_time']
        print('')
        print(f"eval loss : {output.metrics['eval_loss']}")
        print(f"output EM : {output.metrics['eval_exact_match']}, metrics : {metrics}")
        print('------------------------')
        return metrics
    
    # predict -> infernece. 그냥 generate해서 내뱉기.
    # prediction_loop로 가능할까
    def predict(self, inputs):
        # question과 retrieval이 prompt 형태로 합쳐져서 token화 된 후에 와야 함.
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model.generate(**inputs, max_new_tokens=30, num_beams=5)
        pred = self.tokenizer.decode(outputs, skip_special_tokens=True)
        return pred
    