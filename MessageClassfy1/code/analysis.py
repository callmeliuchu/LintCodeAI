import csv
import numpy as np
import pandas as pd
import re
from nltk.corpus import stopwords
inputfile = '../data/train.csv'




def getData():
	csv_reader = csv.reader(open(inputfile,encoding='utf-8'))
	data = np.array(list(csv_reader)[1:])
	return data


def convert(label):
	return 0 if label == 'ham' else 1


def getLabelsAndContens():
	data = getData()
	labels = [convert(arr[0]) for arr in data]
	contents = [arr[1] for arr in data]
	return labels,contents



labels,contents = getLabelsAndContens()
# print(labels)
s = contents[0]



def review_to_wordlist(review,remove_stopwords=False):
	review_text = re.sub("[^a-zA-Z]"," ",review)
	words = review_text.lower().split()
	if remove_stopwords:
		words = [word for word in words if word not in stopwords.words('english')]
	return words

print(review_to_wordlist(s))