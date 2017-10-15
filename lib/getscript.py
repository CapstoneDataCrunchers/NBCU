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
