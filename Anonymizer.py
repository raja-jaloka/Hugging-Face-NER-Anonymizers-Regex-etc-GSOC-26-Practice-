from transformers import pipeline
import re

ner_pipe=pipeline("ner")

master_text='''John Smith from OpenAI emailed maria.garcia@gmail.com on May 5th, 2023. 
He later met Dr. Robert Brown at MIT in California. 
Apple announced a new product while Jordan scored 30 points in the game. 
Contact Rahul Verma before Friday. 
Rahul contacted Dr. Robert Brown yesterday on his phone number 8728374654.
The system processes data efficiently.'''

class Anonymizer:
    def __init__(self):
        self.patterns={
            "EMAIL": r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b',
            "PHONE": r'\b[0-9]{10}\b'
        }
        self.labels={
            "EMAIL": "[EMAIL]",
            "PHONE": "[PHONE]"
        }
        

    def mask(self,text):
        masked_emails=[]
        masked_phones=[]


        def replace_email(match):
            masked_emails.append(match.group())
            return self.labels["EMAIL"]
        
        def replace_phone(match):
            masked_phones.append(match.group())
            return self.labels["PHONE"]
        

        for label,pattern in self.patterns.items():
            if label=="EMAIL":
                text=re.sub(pattern,replace_email,text)
            elif label=="PHONE":
                text=re.sub(pattern,replace_phone,text)

        return text,masked_emails,masked_phones
    
anonymizer=Anonymizer()
#master_text,masked_emails,masked_phones=anonymizer.mask(master_text)
#print(masked_emails)
#print(masked_phones)
#print(master_text)

class Anonymizer_advanced: #handles both email and phone masking in a single loop.
    def __init__(self):
        self.patterns={
            "EMAIL":r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b',
            "PHONE":r'\b[0-9]{10}\b'
        }
        self.labels={
            "EMAIL":"__EMAIL__",
            "PHONE":"__PHONE__"
        }
        
    def mask(self,text):
        for label,pattern in self.patterns.items():
            '''def replace(match):
                return self.labels[label] #we can't use this because label access only the value at the end of the loop i.e "PHONE"
            text=re.sub(pattern,replace,text)''' #However this would still work but it's not always safe to rely on
            def replace(match,label=label):
                return self.labels[label] #we can use this because we are passing the label as a default argument to the function and thus it will access the correct label value for each pattern
            text=re.sub(pattern,replace,text) #gives replace(match)i.e the object that matches the pattern type.
        return text 
    
anonymizer_adv=Anonymizer_advanced()
master_text=anonymizer_adv.mask(master_text)
print(master_text)
print("\n") 

#The customised masking part is now done. 
#Now we mask persons, organizations and locations using NER. 

for e in ner_pipe(master_text):
    print(e)

class Ner_Masker:
    def __init__(self):
        self.labels={
            "I-PER":"[PERSON]",
            "I-ORG":"[ORGANIZATION]",
            "I-LOC":"[LOCATION]"
        }

    def mask(self,text):
        masked_entities=[]
        span=[]
        for e in ner_pipe(text):
            if(e['entity']=="I-PER"):
                span.append((e['start'],e['end'],self.labels[e['entity']]))
            elif(e['entity']=="I-ORG"):
                span.append((e['start'],e['end'],self.labels[e['entity']]))
            elif(e['entity']=="I-LOC"):
                span.append((e['start'],e['end'],self.labels[e['entity']]))
        merged=[]
        i=0
        while(i<len(span)):
            start,end,label=span[i]
            while(i+1<len(span) and (span[i+1][0]==end or span[i+1][0]==end+1) and span[i+1][2]==label):
                end=span[i+1][1]
                i+=1
            merged.append((start,end,label))
            i+=1

        merged.reverse()#to prevent index shifting while replacing
        
        for tup in merged:
            start,end,label=tup 
            text=text[:start]+label+text[end:]
        
        text=text.replace("__EMAIL__","[EMAIL]")
        text=text.replace("__PHONE__","[PHONE]")
        
        return text

#Ner_Anonymizer=Ner_Masker()
#master_text=Ner_Anonymizer.mask(master_text)
#print(master_text) 
#One problem noticed here is that it considers "EMAIL" to be an I-ORG. Therefore the tokeniser is not able to identify masked identities and 
#so we will use a different model for NER which is more accurate and is trained on a larger dataset. Which will be discussed later in a different file.
protected_spans=[]
Ner_Anonymizer=Ner_Masker()
master_text=Ner_Anonymizer.mask(master_text)
print(master_text) 

import sklearn.metrics as metrics 
#we treat the data to be evaluated as a binary classification this evaluation tells 
#whether the model correctly masks the correct entity regardless of the type of label of the mask.
y_true=[1,0,1,0,1,0,0,0,0,0,0,0,0,1,0,1,0,1,1,0,0,0,0,0,1,0,0,0,0,0,0,0,1,0,0,1,0,0,1,0,0,0,0,0,1,0,0,0,0]
y_pred=[1 if word in ["[PERSON]","[ORGANIZATION]","[LOCATION]"] else 0 for word in master_text.split(" ")]
print("Precision_score:",metrics.precision_score(y_true,y_pred,average="binary"))
print("Recall_score:",metrics.recall_score(y_true,y_pred,average="binary"))
print("F1_score:",metrics.f1_score(y_true,y_pred))

#The precision score is a little vague because the dataset is very small. 
#Initial manual evaluation showed high precision in masking structured entities, while recall for semantic entities was moderate, indicating scope for improving detection sensitivity through configurable thresholds.
#We cured Yet another problem which was masking of name and surname as different entities. 



#FINAL CONCLUSION: the present model of ner detects words in a very confusing manner which might/might not be true.
#Thus we're Using GLiNER for better detection check out Ganony.py
