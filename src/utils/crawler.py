'''
This crawler is used to crawl through a given transcript txt file and extract all the repeating three word phrases.
This was used during the initial stages of the project to then generate spectrograms in PRAAT for each of the phrases and compare the effects of emotion on the spectrograms.
As the project progressed, this was not used anymore since the chosen model was not based on spectrograms.
'''

import re
# file = open('./semaine-database-download/Sessions/16/alignedTranscript_03.4Poppy.txt')
file = open('alignedTranscript_10.3Prudence.txt', encoding='utf8')

words = []
index = 0

# Read lines from transcript
for line in file:
    # Remove punctuation from line
    newstr = re.sub(r'[^\w\s]','',line)
    newstr = newstr.lower()
    newstr = newstr.split()
    newSlice = newstr[3::]
    # Line has been obtained and appended to list
    words.extend(newSlice)

matchedPhrases = []
size = len(words)
# Loop through words and add repeating three-word phrases to list
for i in range(size - 5):
    compar1 = words[i:i+3]
    for j in range(i, size - 3):
        compar2 = words[j+1:j+4]
        if compar1 == compar2:
            matchStr = ' '.join(map(str, compar1))
            matchedPhrases.append(matchStr)

for phrase in matchedPhrases:
    print(phrase)
    