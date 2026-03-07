import re  
from gliner import GLiNER

ner_model = GLiNER.from_pretrained("urchade/gliner_medium-v2")


master_text='''John Smith from OpenAI emailed maria.garcia@gmail.com on May 5th, 2023. 
He later met Dr. Robert Brown at MIT in California. 
Apple announced a new product while Jordan scored 30 points in the game. 
Contact Rahul Verma before Friday. 
Rahul contacted Dr. Robert Brown yesterday on his phone number 8728374654.
The system processes data efficiently.'''

class Anonymizer_advanced: #handles both email and phone masking in a single loop.
    def __init__(self):
        self.patterns={
            "EMAIL":r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b',
            "PHONE":r'\b[0-9]{10}\b'
        }
        self.labels={
            "EMAIL":"[EMAIL]",
            "PHONE":"[PHONE]"
        }
        
    def mask(self,text):
        for label,pattern in self.patterns.items():
            def replace(match,label=label):
                return self.labels[label] 
            text=re.sub(pattern,replace,text) 
        return text 

labels=["PERSON","ORGANIZATION","LOCATION"]
entities=ner_model.predict_entities(master_text,labels)
print(entities)

class NER_masker:
    def __init__(self):
        self.labels={
            "PERSON":"[PER]",
            "ORGANIZATION":"[ORG]",
            "LOCATION":"[LOC]"
        }
    
    def mask(self,text):
        entity=ner_model.predict_entities(text,self.labels)
        spans=[]
        for e in entity:
            spans.append((e['text'],e['label']))

        spans.reverse()
        for tup in spans:
            word,label=tup
            text=text.replace(word,self.labels[label])
        return text 
    
Anonymizer=Anonymizer_advanced()
master_text=Anonymizer.mask(master_text)
ner_masker=NER_masker()
master_text=ner_masker.mask(master_text)
print(master_text)
