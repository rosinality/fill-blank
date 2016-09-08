def write_list(filename, iterable):
    with open(filename, 'w') as f:
        f.writelines(str(l) + '\n' for l in iterable)

def read_list(filename, func=None):
    with open(filename) as f:
        if func is not None:
            return [func(line) for line in f]

        else:
            return f.readlines()

def write_map(filename, map, sep=' '):
    with open(filename, 'w') as f:
        for k, v in map.items():
            f.write(sep.join((str(k), str(v))))

def batch_range(max_num, batch_size):
    batch_num, residue = divmod(max_num, batch_size)

    for i in range(batch_num):
        yield batch_size * i, batch_size * (i + 1)

    if residue != 0:
        yield batch_size * (i + 1), max_num

class Map:
    def to_bow(self, iterable):
        result = []
        unique = {}
        
        for word in iterable:
            if word in unique:
                result[unique[word]][1] += 1

            else:
                try: # skip unknown symbol
                    result.append([self.map[word], 1])

                except:
                    continue

                unique[word] = len(result) - 1

        return result 

    def to_seq(self, iterable, unk_symbol=None):
        result = []
        if unk_symbol:
            unk_id = self.map[unk_symbol]
            add_unk = True
        else:
            add_unk = False
        
        for word in iterable:
            try:
                key = self.map[word]

            except:
                if add_unk:
                    key = unk_id

                else:
                    continue

            result.append(key)

        return result

class KeyMap(Map):
    def __init__(self, start=0):
        self.map = {}
        self.id = start

    def add(self, key):
        self.__getitem__(key)

    def __getitem__(self, key):
        try:
            return self.map[key]

        except:
            key_id = self.id
            self.map[key] = self.id
            self.id += 1

            return key_id

class KeyFrequency:
    __slots__ = ('key', 'frequency')

    def __init__(self, key, frequency):
        self.key = key
        self.frequency = frequency

class Dictionary(Map):
    def __init__(self, start=1, add_marker=['<GO>', '<EOS>'],
                auto_count=False):
        self.map = {}
        self.frequency = {}
        self.id = start
        self.auto_count = auto_count
        self.marker = False
        if add_marker:
            self.add(add_marker)
            self.marker = True
            self.start_mark = add_marker[0]
            self.end_mark = add_marker[1]
            self.start_id = start
            self.end_id = start + 1

    def add(self, keys):
        if isinstance(keys, str):
            self.__getitem__(keys)

        else:
            try:
                for key in keys:
                    self.__getitem__(key)

            except:
                self.__getitem__(keys)

    def to_seq(self, iterable, unk_symbol=None, add_marker=True):
        result = super().to_seq(iterable, unk_symbol)

        if self.marker and add_marker:
            result.insert(0, self.start_id)
            result.append(self.end_id)
        
        return result

    def __getitem__(self, key):
        try:
            if self.auto_count:
                try:
                    self.frequency[key] += 1

                except:
                    self.frequency[key] = 1

            return self.map[key]

        except:
            key_id = self.id
            self.map[key] = self.id
            self.id += 1

            if self.auto_count:
                self.frequency[key] = 1

            return key_id

    def count(self, key):
        try:
            self.frequency[key] += 1

        except:
            self.frequency[key] = 1

    def frequency(self):
        return self.frequency.values()

    def over_threshold(self, threshold):
        no = 0
        for k, v in self.frequency.items():
            if v > threshold:
                no += 1
        return no

    def compactify(self):
        for i, (k, v) in enumerate(self.map.items()):
            self.map[k] = i

    def cut_word(self, threshold):
        candidate = {}
        for k, v in self.frequency.items():
            if v > threshold:
                candidate[k] = self.map[k]

        if self.marker:
            candidate[self.start_mark] = self.map[self.start_mark]
            candidate[self.end_mark] = self.map[self.end_mark]
        
        self.map = candidate

class FrequencyMap(Map):
    def __init__(self):
        self.map = {}

    def copy(self):
        new = FrequencyMap()
        new.map = self.map.copy()

        return new

    def count(self, key):
        try:
            self.map[key] += 1

        except:
            self.map[key] = 1

    def frequency(self):
        return self.map.values()

    def over_threshold(self, threshold):
        no = 0
        for k, v in self.map.items():
            if v > threshold:
                no += 1

        return no

    def cut_item(self, threshold):
        candidate = {}
        for k, v in self.map.items():
            if v > threshold:
                candidate[k] = v

        self.map = candidate

class UnknownMap(Map):
    def __init__(self, iterable, keys=[], unknown_symbol='<UNK>', start=1):
        self.map = {}
        self.length = start
        
        for key in keys:
            self.map[key] = self.length
            self.length += 1
        
        for key in iterable:
            self.map[key] = self.length
            self.length += 1

        self.unknown_symbol = unknown_symbol
        self.unknown = self.length
        self.map[self.unknown_symbol] = self.unknown
        self.length += 1

    def __getitem__(self, key):
        try:
            return self.map[key]

        except:
            return self.unknown