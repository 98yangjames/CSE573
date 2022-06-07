
#import the following libraries
import pandas as pd
import nlp

df = pd.read_csv("netflix_titles.csv")

#Convert text to all lowercase
df['new_description'] = df['description'].apply(lambda x: " ".join(x.lower() for x in x.split()))
df['new_description'].head()

#remove the punctuation
df['new_description'] = df['new_description'].str.replace('[^\w\s]','')
df['new_description'].head()


from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()

for word in df['new_description']:
    print(word)