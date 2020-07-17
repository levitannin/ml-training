# -*- coding: utf-8 -*-
"""
Created on Wed Jul 15 06:24:27 2020

@author: Levitannin

This program is a tutoiral in using NLTK to identify words and sentences.
The purposes of this program is to practice with NLTK before moving into
dynamic projects in the future.

This is Part 6 -- focused on chinking; removing from the text what is not wanted.
"""

from nltk.tokenize import word_tokenize, PunktSentenceTokenizer
from nltk.corpus import state_union
from nltk import pos_tag, RegexpParser

#   Speech Tagging -- Identifying the different parts of speech.
#-----------------------------------------------------------------------------
#   Identify what items the imported corpus may have.

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
    for i in token[6:]:#    You can specify here if you want to start at a certain level of the chunk.  IE [5:]
        words = word_tokenize(i)
        tag = pos_tag(words)
        
        #   To find all versions of a POS use regular expressions to identify it.
        #   Example here are Adverbs (RB, RBR, RBS)
        #   . == any character other than new line
        #   ? == 0 or 1
        #   * == 0 or MORE
        #   | == or
        chunkGram = r""" Chunk: {<.*>+} 
                                }<VB.?|IN|DT>+{"""
        chunkParser = RegexpParser(chunkGram)
        chunked = chunkParser.parse(tag)
        #   Will generate a pop-up box with the chucks drawn out in a chart.
        chunked.draw()
        
except Exception as e:
    print(str(e))
