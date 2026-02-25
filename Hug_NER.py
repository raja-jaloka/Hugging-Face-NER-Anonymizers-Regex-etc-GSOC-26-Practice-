from transformers import pipeline
import sys

print("[1/5] Loading NER classifier...", file=sys.stderr) #acts as a checkpoint, and is printed to standard error
sys.stderr.flush() #flush is used to output the text immediately without buffering 
classifier=pipeline("ner") #ner stands for named Entity Recognition 
print("[2/5] Classifier loaded!", file=sys.stderr)
sys.stderr.flush()
#NER as the name suggests is used to identify the named entities in a text.
#  It can be used to identify the names of people, organizations, locations, etc. in a text.

text="Hugging Face Inc. is a company in which Yuvraj works, based in New York City. Its headquarters are in DUMBO, therefore very close to the Manhattan Bridge."
print("[3/5] Running NER on text...", file=sys.stderr)
sys.stderr.flush()

'''for e in classifier(text): #classifier(text) will return a list of dictionaries, where each dictionary contains the entity, its label, and its score.
   print(e)'''  #here we are printing the output of the classifier which is a list of dictionaries.
    #Each dictionary contains the entity, its label, and its score.

#tokenizers in the classifier tokenize the input text into smaller chunks based on the modal's vocabulary.largest subword units
#  The tokenizers are used to convert the input text into a format that can be processed by the model.
#  The tokenizers are also used to convert the output of the model back into human-readable text.

#Now we Plan to mask a few entities in the text and then
 # use the fill-mask pipeline to predict the masked entities.
#print(classifier(text)) #here we are printing the output of the classifier which is a list of dictionaries.

'''for e in classifier(text):
    if e['entity']=='I-PER':
        text=text.replace(e['word'],'[MASK]')'''
        #This is wrong way because it alters the text and the indices change, 
        #moreover words are split into tokens which aren't the same as the original words.

#solution to this is 
'''spans=[]
for e in classifier(text):
    if(e['entity']=='I-PER'):
        spans.append((e['start'],e['end'])) #stored as a tuple of start and end indices 

spans.reverse() #reversing is very important because left to right replacement again alters the indices.

for start,end in spans: #Note start,end unpacks the indices of in the spans in left-right order
    text=text[:start]+'[MASK]'+text[end:]''' #Note end isnt inclusive for the token 

#here the word is replaced by mask as many times as there are subwords. 

# print(text)

#But that isn't effective there's too much redundancy in the text and 
# the complexity increases as the number of subwords increases.
#moreover it's decieving as it might mean there are too many people in the text which isn't the case. 

#SO The Solution to this is...
'''spans=[]
for e in classifier(text):
    if(e['entity']=='I-PER'):
        spans.append((e['start'],e['end']))

# spans.reverse()  # I don't require reversing span here because my text operation isn't based on spans 

merged=[]
i=0

print("[4/5] Merging entity spans...", file=sys.stderr)
sys.stderr.flush()'''

'''while i<len(spans): #DOn't use for loop beczuse internal i change will not be updated in the for for loop 
    start,end=spans[i]

    while i+1<len(spans) and spans[i+1][0]==end: #to check if the next token is a continuation fo the current
        end=spans[i+1][1]
        i+=1
    merged.append((start,end))
    i+=1'''

'''merged.reverse()''' #reversing is very important because left to right replacement again alters the indices.

'''for tup in merged:
    start,end=tup
    text=text[:start]+'[MASK]'+text[end:]

print("[5/5] Done!", file=sys.stderr)
sys.stderr.flush()
print(text)'''
 #Thus we have successfully masked the entities in the text without altering the indices 
 #and without creating redundancy in the text.
 #  We can now use the fill-mask pipeline to predict the masked entities.



#Now we notice that the model outputs probability scores 
#that means we can filter the output based on a threshold score to get more accurate predictions.

master_text='''John Smith from OpenAI emailed maria.garcia@gmail.com on May 5th, 2023. 
He later met Dr. Robert Brown at MIT in California. 
Apple announced a new product while Jordan scored 30 points in the game. 
Contact Rahul Verma before Friday. 
The system processes data efficiently.'''

threshold0=0.5

for e in classifier(master_text):
   print(e) #here we are printing the output of the classifier which is a list of dictionaries, where each dictionary contains the entity, its label, and its score.
 #Notice when we print the above output we notice that0-9 the email is not fully classified as an entity 
 #because it's split into multiple tokens, moreover it might still not flag them as entities because of incorrect 
 #tokenization of the email address. Therefore to resolve it we have to manually flag the emails using regex.

import re 
email_pattern=r"\b[A-Za-z0-9.]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"
print(re.findall(email_pattern, master_text)) #this will return a list of all email addresses found in the master_text

#let's assign these addresses to a list 
emails=re.findall(email_pattern, master_text)

#Now we manually flag these emails as ['EMAIL']
'''words_master_text=master_text.split(" ") #split the master text into words

for i in range(len(words_master_text)):
   if(words_master_text[i] in emails):
      words_master_text[i]='[EMAIL]'

print(words_master_text) 

text_final=' '.join(words_master_text) #join the list of words back into a single string gapped with a whitespace
print(text_final)''' #this will print the final text with the email addresses replaced by [EMAIL]

#This way we have successfully flagged the email addresses in the text without relying on the NER model's tokenization

#But this is still not the most efficient way to do it because splitting the text can still contain some unwanted
#characters like punctuation. 

final_text=re.sub(email_pattern,'[EMAIL]',master_text) #this will replace all email addresses of the pattern 
#email_pattern with [EMAIL] in the master_text and return the final text with email addresses replaced by [EMAIL]
print(final_text) 

#We Repeat the same process for phone Numbers as well 
phone_pattern=r"\b[0-9]{10}\b" #This pattern will match any 10 digit number
phony_text='''Contact Rahul Verma at 1234567890 or 9876543210.'''
print(re.sub(phone_pattern,'[PHONE]',phony_text))

