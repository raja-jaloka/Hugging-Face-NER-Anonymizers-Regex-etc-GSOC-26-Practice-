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

class Anonymizer_advanced:
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
            '''def replace(match):
                return self.labels[label] #we can't use this because label access only the value at the end of the loop i.e "PHONE"
            text=re.sub(pattern,replace,text)''' #However this would still work but it's not always safe to rely on
            def replace(match,label=label):
                return self.labels[label] #we can use this because we are passing the label as a default argument to the function and thus it will access the correct label value for each pattern
            text=re.sub(pattern,replace,text)
        return text 
    
anonymizer_adv=Anonymizer_advanced()
master_text=anonymizer_adv.mask(master_text)
print(master_text)