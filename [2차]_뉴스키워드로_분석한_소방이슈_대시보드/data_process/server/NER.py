
import json
from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import pipeline

model_path = "/home/intern2/model/models/training_modu-ner_jinmang2-kpfbert_2022-11-16_00-00-00"

class NerFactory:
    
    def __init__(self):
        # Load a trained sentiments model
        classifier = AutoModelForTokenClassification.from_pretrained(model_path)
        tokenizer = AutoTokenizer.from_pretrained(model_path, model_max_length=512)
        self.model = pipeline('ner', model=classifier, tokenizer=tokenizer, device=0)
        with open('/home/intern2/model/models/utils/label_config.json', 'r') as f: conf = json.load(f)
        self.label2tag = conf['named_entity_recognition']['label2tag']
        self.tag2code = conf['named_entity_recognition']['tag2code']
        self.stopwords = conf['named_entity_recognition']['stopwords']

    def get_entities(self, sentence):
        values = self.model(sentence)
        
        dict_of_ner_word = []
        word, code, prev_code = "", "", ""
        for value in values:
            token = value['word'].strip().replace("##", "")
            tag = self.label2tag[value['entity']]
            if tag.startswith('B-'):
                code = self.tag2code[tag]
                if prev_code != code and len(word) > 1 and word not in self.stopwords:
                    result = {"code": prev_code, "word": word}
                    if result not in dict_of_ner_word: dict_of_ner_word.append(result)
                word = sentence[value['start']: value['end']]
                idx = value['start']
                prev_code = code
            elif tag == "I-"+code:
                word = sentence[idx: value['end']]
            else:
                if len(word) > 1 and word not in self.stopwords:
                    result = {"code": code, "word": word}
                    if result not in dict_of_ner_word: dict_of_ner_word.append(result)
                word, code, prev_code = "", "", ""
    
        return dict_of_ner_word