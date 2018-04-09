#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 19 22:50:09 2018

@author: changlongjiang
"""

import re
import os
def clean_up(string):
    """
    get rid of some useless '\' and make sure all the text is lower case
    """
    string = re.sub(r"\\", "", string)
    string = re.sub(r"\'", "", string)
    string = re.sub(r"\"", "", string)
    return string.strip().lower()

def get_word_embedding(DIR,SUB_DIR):
    
    d = {}
    f = open(os.path.join(DIR, SUB_DIR))
    for line in f:
        v = line.split()
        word = v[0]
        vec = np.asarray(v[1:], dtype='float32')
        d[word] = vec
    f.close()
    return d