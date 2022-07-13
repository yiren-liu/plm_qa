import pprint 
import json
pp = pprint.PrettyPrinter(indent=4)

from os import listdir
from os.path import isfile, join

from tqdm import tqdm

from utils import offset2lineNum
from dialogue_qa import DialogueStateTracking

SKIP_LOAD_DATASET = False

model = "deepset/roberta-base-squad2"
dst = DialogueStateTracking(model)

# HarassmentAbuse + TheftLostItem
domains = ["HarassmentAbuse", "TheftLostItem"]
slot_temp_dict = {
    "HarassmentAbuse": [
        "Target-ARG", "Place-Arg", "Attacker-Arg", "End_Time-Arg"
        ],
    "TheftLostItem": [
        "Target-ARG", "Place-Arg", "Attacker-Arg", "End_Time-Arg", 
        "Target_Object-Arg", "Start_Time-Arg"
        ],
}
slot_questions_dict = {
    "HarassmentAbuse": [
        "Who is the victim?" ,
        "Where did the incident take place?",
        "Who is the attacker?",
        "When did this happen?",
        ],
    "TheftLostItem": [
        "Who is the victim?" ,
        "Where did the theft take place?",
        "Who is the attacker?",
        "When did you last see the stolen object?",
        "What object was stolen?",
        "When did you notice the object was missing?",
    ],
}

if not SKIP_LOAD_DATASET:
    examples = []
    qid = 0
    print("Loading training examples...")
    for d in domains:
        print("evaluating %s instances..."%d)
        dst.slot_temp = slot_temp_dict[d]
        dst.slot_questions = slot_questions_dict[d]
        # val_data_path = "./data/subset_1/%s/"%d
        # val_gold_data_path = "./data/val_gold/"
        # val_gold_dict_path = "./data/val_gold_dict/"

        # output_path = "./outputs/all_annotated/%s/"%d

        train_data_path = "./data/all_annotated/%s/"%d

        fnames = [f for f in listdir(train_data_path) if isfile(join(train_data_path, f))]
        eventIds = [fname.split('_')[-1].split('.')[0] for fname in fnames]
        accs = []
        f1s = []
        for eventId in tqdm(eventIds):
            # examples: []{
            # 'answers': {'answer_start': [515], 'text': ['Saint Bernadette Soubirous']},
            # 'context': 'Architecturally, the school has a Catholic character. Atop the Main Building\'s gold dome is a golden statue of the Virgin Mary. Immediately in front of the Main Building and facing it, is a copper statue of Christ with arms upraised with the legend "Venite Ad Me Omnes". Next to the Main Building is the Basilica of the Sacred Heart. Immediately behind the basilica is the Grotto, a Marian place of prayer and reflection. It is a replica of the grotto at Lourdes, France where the Virgin Mary reputedly appeared to Saint Bernadette Soubirous in 1858. At the end of the main drive (and in a direct line that connects through 3 statues and the Gold Dome), is a simple, modern stone statue of Mary.',
            # 'question': 'To whom did the Virgin Mary allegedly appear in 1858 in Lourdes France?',
            # }
            with open(train_data_path + 'event_%s.txt'%eventId, 'r', encoding='utf8') as f:
                dialogue = f.readlines()
                with open(train_data_path + 'event_%s.ann'%eventId, 'r', encoding='utf8') as f:
                    lines = f.readlines()
                    for line in lines:
                        if line.startswith('E'):
                            for t in line.split('\t')[-1].split(' '):
                                arg_name, eid = t.split(':')
                                if arg_name  in dst.slot_temp:
                                    for l in lines:
                                        if l.startswith(eid):
                                            l = l.strip()
                                            _, span, text = l.split('\t')
                                            span_start = int(span.split(' ')[1])

                                            line_num = offset2lineNum(train_data_path + 'event_%s.txt'%eventId, span_start)
                                            examples.append({
                                                'id': qid,
                                                'context': '\n'.join(dialogue[:line_num+1]),
                                                'question': dst.slot_questions[dst.slot_temp.index(arg_name)],
                                                'answers': {
                                                    'answer_start': [span_start],
                                                    'text': [text]
                                                }
                                            })
                                            qid += 1

                                            # pp.pprint(examples)
                                            # print(len(examples[0]['context'].split()))
                                            # assert False
                            continue

    # pp.pprint(examples)
    # print(len(examples))

    json.dump(examples, open('data/all_annotated/all_annotated.json', 'w', encoding='utf8'))


# start training here
examples = json.load(open('data/all_annotated/all_annotated.json', 'r', encoding='utf8'))
dst.finetune(examples)


