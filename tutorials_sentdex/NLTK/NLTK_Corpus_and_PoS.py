# -*- coding: utf-8 -*-
"""
Created on Wed Jul 15 06:24:27 2020

@author: Levitannin

This program is a tutoiral in using NLTK to identify words and sentences.
The purposes of this program is to practice with NLTK before moving into
dynamic projects in the future.

This is Part 4 -- focused on identifying tool used for text corpus embedded
  in NLTK.  This part does not follow the tutorial and was self practice.
  Also works on identifying parts of speach (PoS), which does follow the tutorial.
"""

from nltk.tokenize import word_tokenize, PunktSentenceTokenizer
from nltk.corpus import gutenberg, state_union
from nltk import pos_tag

#   Speech Tagging -- Identifying the different parts of speech.
#-----------------------------------------------------------------------------
#   Identify what items the imported corpus may have.
print("\nText available from Gutenberg: \n")
print(gutenberg.fileids())
print("\nText available from State of the Union: \n")
print(state_union.fileids())

#   Useful tools we can use on the imported text.

#   The following will create a table to identify:
#       Average word length
#       Average sentence length
#       Frequency of a vocab word appearing
#   For each text in the Gutenberg corpus
print("Ave Word Len \t Ave Sent Len \t Vocab Occurance \t Title ")
for fileid in gutenberg.fileids():
    num_chars = len(gutenberg.raw(fileid))
    num_words = len(gutenberg.words(fileid))
    num_sents = len(gutenberg.sents(fileid))
    num_vocab = len(set(w.lower() for w in gutenberg.words(fileid)))
    
    print(round(num_chars / num_words), "\t\t", round(num_words / num_sents),
          "\t\t", round(num_words / num_vocab), "\t\t", fileid)

paradise_sent = gutenberg.sents("milton-paradise.txt")
print(len(paradise_sent))

#   Print a sentence in the corpus after breaking into sents(ences)
print(paradise_sent[1313])

#   Identify the longest sentence in the chosen corpus.
longest_sent = max(len(s) for s in paradise_sent)
print(s for s in paradise_sent if len (s) == longest_sent)

#   Can find other text sources at: https://www.nltk.org/book/ch02.html
#   Raw gives the content of the file without any linguistic processing.
train_text = state_union.raw("1953-Eisenhower.txt")
sample_text = state_union.raw("1959-Eisenhower.txt")
#   If you want to re-train the Punkt for your purposes
custom_sent_tokenizer = PunktSentenceTokenizer(train_text)
token = custom_sent_tokenizer.tokenize(sample_text)

'''
POS tag list:

    CC      coordinating conjunction
    CD      cardinal digit
    DT      determiner
    EX      existential there (like: "there is" ... think of it like "there exists")
    FW      foreign word
    IN      preposition/subordinating conjunction
    JJ      adjective 'big'
    JJR     adjective, comparative 'bigger'
    JJS     adjective, superlative 'biggest'
    LS      list marker 1)
    MD      modal could, will
    NN      noun, singular 'desk'
    NNS     noun plural 'desks'
    NNP     proper noun, singular 'Harrison'
    NNPS    proper noun, plural 'Americans'
    PDT     predeterminer 'all the kids'
    POS     possessive ending parent's
    PRP     personal pronoun I, he, she
    PRP$    possessive pronoun my, his, hers
    RB      adverb very, silently,
    RBR     adverb, comparative better
    RBS     adverb, superlative best
    RP      particle give up
    TO      to go 'to' the store.
    UH      interjection errrrrrrrm
    VB      verb, base form take
    VBD     verb, past tense took
    VBG     verb, gerund/present participle taking
    VBN     verb, past participle taken
    VBP     verb, sing. present, non-3d take
    VBZ     verb, 3rd person sing. present takes
    WDT     wh-determiner which
    WP      wh-pronoun who, what
    WP$     possessive wh-pronoun whose
    WRB     wh-abverb where, when

'''
try:
    for i in token:#    You can specify here if you want to start at a certain level of the chunk.  IE [5:]
        words = word_tokenize(i)
        tag = pos_tag(words)
        print(tag)
        
except Exception as e:
    print(str(e))
