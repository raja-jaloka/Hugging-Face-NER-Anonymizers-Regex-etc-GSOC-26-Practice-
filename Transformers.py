from transformers import pipeline

print("creating pipeline…")
pipe = pipeline("sentiment-analysis")
print("pipeline ready")

result = pipe("I love you 3000") 
result1 = pipe("I will kill ")
result2 = pipe("I am not sure how I feel about you")
#the output is a list of dictionaries, 
# where each dictionary contains the label and the score of the sentiment analysis.

print(result)
print(result1)
print(result2)