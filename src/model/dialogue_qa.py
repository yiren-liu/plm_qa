import json

import numpy as np
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, DefaultDataCollator, TrainingArguments, Trainer, pipeline
import torch
import pandas as pd

from datasets import Dataset, load_dataset

from collections import defaultdict
from os import listdir
from os.path import isfile, join

from utils import LivesafeDataset

# Hyperparameters
max_per_slot = 2 # Max number of candidate responses to extract (per iteration)
max_span_length = 20 # Max length of one answer span
min_score = 0 # Min score for an answer span

class QA_Model:
    def __init__(self, model):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(model)
        self.model = AutoModelForQuestionAnswering.from_pretrained(model).to(self.device)
    
    def answer(self, text, questions, slot_temp):
        answerss = {}
        for i in range(len(questions)):
            question = questions[i]
            slot = slot_temp[i]

            inputs = self.tokenizer.encode_plus(
                question, text, add_special_tokens=True, return_tensors="pt",
                truncation= True, max_length = 512, padding='max_length'
                )
            inputs = inputs.to(self.device)

            input_ids = inputs["input_ids"].tolist()[0]

            text_tokens = self.tokenizer.convert_ids_to_tokens(input_ids)
            answer_start_scores, answer_end_scores = self.model(**inputs, return_dict=False)

            answer_start = torch.argmax(answer_start_scores)  # Get the most likely beginning of answer with the argmax of the score
            answer_end = torch.argmax(answer_end_scores) + 1  # Get the most likely end of answer with the argmax of the score
            score = torch.max(answer_start_scores) + torch.max(answer_end_scores)
            
            if answer_start < 1: 
                answer = [] # cannot start with CLS
            elif answer_end - answer_start + 1 > max_span_length or answer_end - answer_start + 1 < 1:
                answer = [] # cannot be longer than hyperparam, or empty
            elif score < min_score:
                answer = [] # cannot be < hyperparam
            else:
                answer = [self.tokenizer.convert_tokens_to_string(self.tokenizer.convert_ids_to_tokens(input_ids[answer_start:answer_end]))]
#                 print(f"Question: {question}")
#                 print(f"Answer: {answer}")
#                 print(f"Score: {score}") 
                
            answerss[slot] = answer
        return answerss

    def finetune(self, examples):
        # {
        # 'answers': {'answer_start': [515], 'text': ['Saint Bernadette Soubirous']},
        # 'context': 'Architecturally, the school has a Catholic character. Atop the Main Building\'s gold dome is a golden statue of the Virgin Mary. Immediately in front of the Main Building and facing it, is a copper statue of Christ with arms upraised with the legend "Venite Ad Me Omnes". Next to the Main Building is the Basilica of the Sacred Heart. Immediately behind the basilica is the Grotto, a Marian place of prayer and reflection. It is a replica of the grotto at Lourdes, France where the Virgin Mary reputedly appeared to Saint Bernadette Soubirous in 1858. At the end of the main drive (and in a direct line that connects through 3 statues and the Gold Dome), is a simple, modern stone statue of Mary.',
        # 'id': '5733be284776f41900661182',
        # 'question': 'To whom did the Virgin Mary allegedly appear in 1858 in Lourdes France?',
        # 'title': 'University_of_Notre_Dame'
        # }
        train_split = 0.8
        train_examples = examples[:int(len(examples)*train_split)]
        eval_examples = examples[int(len(examples)*train_split):]

        dataset_train = Dataset.from_pandas(pd.DataFrame(train_examples))
        dataset_eval = Dataset.from_pandas(pd.DataFrame(eval_examples))

        # dump the dataset to disk
        dataset_train.to_json("results/gt/train.json", lines=False)
        dataset_eval.to_json("results/gt/eval.json", lines=False)


        dataset_train = dataset_train.map(self.preprocess_function, batched=True, remove_columns=dataset_train.column_names)
        dataset_eval = dataset_eval.map(self.preprocess_function, batched=True, remove_columns=dataset_eval.column_names)

        data_collator = DefaultDataCollator()

        training_args = TrainingArguments(
            output_dir="./results",
            evaluation_strategy="epoch",
            learning_rate=2e-5,
            per_device_train_batch_size=4,
            per_device_eval_batch_size=4,
            num_train_epochs=20,
            weight_decay=0.01,
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=dataset_train,
            eval_dataset=dataset_eval,
            tokenizer=self.tokenizer,
            data_collator=data_collator,
        )

        trainer.train()


    
        # evaluate the model
        nlp = pipeline('question-answering', model=self.model, tokenizer=self.tokenizer, device=0)
        # QA_input = {
        #     'question': 'Why is model conversion important?',
        #     'context': 'The option to convert models between FARM and transformers gives freedom to the user and let people easily switch between frameworks.'
        # }
        eval_inputs = [{'question': example['question'], 'context': example['context']} for example in eval_examples]
        res = nlp(eval_inputs)
        
        # print(res) # [{'score': 0.09541689604520798, 'start': 522, 'end': 527, 'answer': 'Admin'}, ...]
        preds = []
        for idx, d in enumerate(eval_examples):
            preds.append({
                "id":d['id'],
                "context": d["context"],
                "answers":
                    {
                    "answer_start": [res[idx]['start']],
                    "text":[res[idx]['answer']]
                }
            })
        

        # dump the predictions to disk
        with open("results/pred/predictions.json", "w") as f:
            json.dump(preds, f)
        # raise NotImplementedError


    def preprocess_function(self, examples):
        # print(examples)
        # examples = [examples]

        tokenizer, max_length = self.tokenizer, 512
        questions = [q.strip() for q in examples["question"]]
        inputs = tokenizer(
            questions,
            examples["context"],
            max_length=max_length,
            truncation="only_second",
            return_offsets_mapping=True,
            padding="max_length",
        )

        offset_mapping = inputs.pop("offset_mapping")
        answers = examples["answers"]
        start_positions = []
        end_positions = []

        for i, offset in enumerate(offset_mapping):
            answer = answers[i]
            start_char = answer["answer_start"][0]
            end_char = answer["answer_start"][0] + len(answer["text"][0])
            sequence_ids = inputs.sequence_ids(i)

            # Find the start and end of the context
            idx = 0
            while sequence_ids[idx] != 1:
                idx += 1
            context_start = idx
            while sequence_ids[idx] == 1:
                idx += 1
            context_end = idx - 1

            # If the answer is not fully inside the context, label it (0, 0)
            if offset[context_start][0] > end_char or offset[context_end][1] < start_char:
                start_positions.append(0)
                end_positions.append(0)
            else:
                # Otherwise it's the start and end token positions
                idx = context_start
                while idx <= context_end and offset[idx][0] <= start_char:
                    idx += 1
                start_positions.append(idx - 1)

                idx = context_end
                while idx >= context_start and offset[idx][1] >= end_char:
                    idx -= 1
                end_positions.append(idx + 1)

        inputs["start_positions"] = start_positions
        inputs["end_positions"] = end_positions
        return inputs

class DialogueStateTracking:
    def __init__(self, model):
        self.qa = QA_Model(model)
        self.slot_temp = [] # slot template
        self.slot_questions = [] # questions corresponding to slot template
        
    # dialogue: list of strings (each utterance in a new string)
    # returns: answers: list of set of tuples. for each utterance, the slots predicted for the dialogue history
    def predict_slots(self, dialogue):
        answers = [] # length of dialogue
        for i in range(len(dialogue)):
            section = " ".join(dialogue[:i+1])
            answers_i_dict = self.qa.answer(section, self.slot_questions, self.slot_temp)
            answers.append(answers_i_dict)

        # Carry through all predicted slots
        for i in range(len(answers) - 1):
            prev_ans = answers[i]
            ans = answers[i + 1]
            for slot in self.slot_temp:
                for p_a in prev_ans[slot]:
                    if p_a not in ans[slot]:
                        ans[slot].append(p_a)
            answers[i + 1] = ans
        return answers

    def finetune(self, examples):
        # finetune bert QA model
        self.qa.finetune(examples)

        # raise NotImplementedError

