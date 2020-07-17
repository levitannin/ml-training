# -*- coding: utf-8 -*-
"""
Created on Wed Jul 15 06:24:27 2020

@author: Levitannin

This program is a tutoiral in using NLTK to identify words and sentences.
The purposes of this program is to practice with NLTK before moving into
dynamic projects in the future.

This is Part 3 -- focused on tokenization and stemming words.
"""

from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer

#nltk.download() #   Well that's cool.  This is downloadable items for the nltk library.
#-----------------------------------------------------------------------------
#   Tokenizing can be broken down into word tokenizers to break down words
#       or into sentence tokenizers to break down sentences.
#   Corpora == body of text
#   Lexicon == dictionary; words and their means

example_text = "Hello Mr. Stranger, how are you doing today?  The weather is great and python is awesome.  Let's meet later for tea.  We can discuss why the sky is pinkish blue, and how that tells us not to eat cardboard."

words = word_tokenize(example_text)
#   Taking the root or stem of a word; removing 'ing', 'ed', etc.
#   We do this to help with processing.  Two words that mean the same thing but
#       are in different tenses is inefficent for processing.  This reduces that.
ps = PorterStemmer()

for w in words:
    print(ps.stem(w))
