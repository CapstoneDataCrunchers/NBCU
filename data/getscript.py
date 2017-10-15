#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 14 06:52:02 2017

@author: GraceG
"""
#read script from a .srt file
import urllib.request, urllib.parse, urllib.error
url = input('url>')#input the url of script

fh = urllib.request.urlopen(url)
srt = fh.read().decode()

#create a .txt file and write the srt into file
import os
fname = input('fname>')#input the filename you want to create
    
script=open(fname,'w')
script.write(srt)
script.close()
