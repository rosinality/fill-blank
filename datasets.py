import os
from textblob import TextBlob
import tqdm
import functools
from multiprocessing import Lock
import pickle
import re
from nltk.corpus import stopwords

from passage.collections import Dictionary, FrequencyMap, UnknownMap

#en_nlp = spacy.load('en')
dictionary = Dictionary(add_marker=False)
filter = re.compile(r'[^\w\.\?!, ]')
stop = set(stopwords.words('english'))
freq = FrequencyMap()

def preprocess_imdb(directory, filename):
    with open(filename + '.data', 'w') as f, \
        open(filename + '.meta', 'w') as c:
        for category in ['pos', 'neg', 'unsup']:
            dir_path, files = get_files(directory, category)
            reader = functools.partial(read_imdb, dir_path)
            for file, rating, meta, review in tqdm.tqdm(
                map(reader, files), desc=category):
                offset = f.tell()
                c.write('{} {} {} {} {}\n'.format(offset, meta[0], meta[1],
                                                rating, file))
                f.write('\n'.join(' '.join(str(num) for num in line) \
                                        for line in review) + '\n')

    with open(filename + '.dict', 'wb') as f:
        pickle.dump(freq, f)

def encode_data(filename):
    with open(filename + '.dict', 'rb') as f:
        freq = pickle.load(f)

    freq.cut_item(80)
    unk = UnknownMap(freq.map)

    with open(filename + '.data', 'r') as str_file, \
        open(filename + '.code', 'w') as code_file, \
        open(filename + '.meta', 'r') as meta_file, \
        open(filename + '.code.meta', 'w') as code_meta:
        for meta_line in tqdm.tqdm(meta_file):
            meta = [int(v) for v in meta_line.split()[:-2]]
            offset = code_file.tell()
            sen_count = 0
            sen_len = []
            str_file.seek(meta[0])
            for i in range(meta[1]):
                val = str_file.readline().split()
                if len(val) < 1:
                    continue
                token = unk.to_seq(val, '<UNK>')
                code_file.write(' '.join(str(i) for i in token) + '\n')
                sen_count += 1
                sen_len.append(len(token))
            if sen_count < 1:
                continue
            sen_max = max(sen_len)
            code_meta.write('{} {} {}\n'.format(offset, sen_count, sen_max))

    with open(filename + '.code.dict', 'wb') as f:
        pickle.dump(unk, f)

def get_files(directory, category='pos'):
    expanded_path = os.path.expanduser(directory)
    sub = os.path.join(expanded_path, category)

    return sub, os.listdir(sub)

def read_imdb(directory, file):
    rating = int(os.path.splitext(file)[0].split('_')[1])
    with open(os.path.join(directory, file)) as f:
        content = f.read()
    encoded = imdb_parse('\n'.join(content.split('<br />')))
    return file, rating, make_meta(encoded), encoded

def make_meta(lines):
    max_len = max(map(len, lines))
    length = len(lines)
    return length, max_len

def imdb_parse(content):
    preproc = content.lower()
    result = []
    for sent in TextBlob(preproc).sentences:
        words = [word for word in sent.words if word not in stop]
        for word in words:
            freq.count(word)
        result.append(words)
    return result
    #return words
        #dictionary.add(words)
        #result.append(dictionary.to_seq(words, add_marker=False))

    #return result