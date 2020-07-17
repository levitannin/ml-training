# -*- coding: utf-8 -*-
"""
Created on Wed Jul 15 06:24:27 2020

@author: Levitannin

This program is a tutoiral in using NLTK to identify words and sentences.
The purposes of this program is to practice with NLTK before moving into
dynamic projects in the future.

This is Part 1 -- focused on tokenization.
"""

from nltk.tokenize import sent_tokenize, word_tokenize

#nltk.download() #   Well that's cool.  This is downloadable items for the nltk library.
#-----------------------------------------------------------------------------
#   Tokenizing can be broken down into word tokenizers to break down words
#       or into sentence tokenizers to break down sentences.
#   Corpora == body of text
#   Lexicon == dictionary; words and their means

example_text = "Hello Mr. Stranger, how are you doing today?  The weather is great and python is awesome.  Let's meet later for tea.  We can discuss why the sky is pinkish blue, and how that tells us not to eat cardboard."

print(sent_tokenize(example_text))
print(word_tokenize(example_text))  #   This will treat punctuation as a 'word' by default.

#   Below will print each word outside of the list.
for i in word_tokenize(example_text):
    print(i)
