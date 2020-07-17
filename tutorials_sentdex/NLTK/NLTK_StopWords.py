# -*- coding: utf-8 -*-
"""
Created on Wed Jul 15 06:24:27 2020

@author: Levitannin

This program is a tutoiral in using NLTK to identify words and sentences.
The purposes of this program is to practice with NLTK before moving into
dynamic projects in the future.

This is Part 2 -- focused on tokenization and identifying stop words.
"""

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

#nltk.download() #   Well that's cool.  This is downloadable items for the nltk library.
#-----------------------------------------------------------------------------
#   Tokenizing can be broken down into word tokenizers to break down words
#       or into sentence tokenizers to break down sentences.
#   Corpora == body of text
#   Lexicon == dictionary; words and their means

example_text = "Hello Mr. Stranger, how are you doing today?  The weather is great and python is awesome.  Let's meet later for tea.  We can discuss why the sky is pinkish blue, and how that tells us not to eat cardboard."

stop_words = set(stopwords.words("english"))
print(stop_words)

words = word_tokenize(example_text)
filtered_sentence = [w for w in words if not w in stop_words]
        
print(filtered_sentence)
