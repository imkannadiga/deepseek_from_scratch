import re

PATTERN = re.compile(r""" ?[a-zA-Z]+| ?[0-9]+| ?[^\sa-zA-Z0-9]+|\s+""")

class BPETokenizer:
    def __init__(self):
        self.merges = []
        self.vocab = {}
        self.id_to_symbol = {}


    def _pretokenize(self, text):
        return PATTERN.findall(text)

    def _get_word_freqs(self, words):
        freqs = dict()

        for word in words:
            tokens = tuple(word)
            count = freqs.get(tokens, 0) + 1
            freqs[tokens] = count

        return freqs

    def _get_pair_counts(self, word_freqs):
        pair_counts = {}
        for word,freq in word_freqs.items():
            for i in range(len(word)-1):
                pair = (word[i], word[i+1])
                pair_counts[pair] = pair_counts.get(pair, 0) + freq 

        return pair_counts

    def _merge_pair(self, pair, word_freqs):
        new_word_freqs = {}
        for word, freq in word_freqs.items():
            new_word = []
            i = 0
            while i < len(word):
                if i < len(word) - 1 and (word[i], word[i + 1]) == pair:
                    new_word.append(word[i] + word[i + 1]) 
                    i += 2 
                else:
                    new_word.append(word[i])
                    i += 1
            new_word_freqs[tuple(new_word)] = freq
        return new_word_freqs

    def _collect_symbols(self, word_freqs):
        symbols = set()
        for word in word_freqs:
            symbols.update(word) 
        return symbols

    def _build_vocab(self, symbols):
        self.vocab = {}
        self.id_to_symbol = {}
        for idx, symbol in enumerate(sorted(symbols)):
            self.vocab[symbol] = idx
            self.id_to_symbol[idx] = symbol

    def train(self, corpus_text, vocab_size, min_occurrences=1):
        words = self._pretokenize(corpus_text)
        word_freqs = self._get_word_freqs(words)

        all_symbols = self._collect_symbols(word_freqs) 

        while True:
            pair_counts = self._get_pair_counts(word_freqs)
            if not pair_counts:
                break

            best_pair = max(pair_counts, key=pair_counts.get)
            if pair_counts[best_pair] < min_occurrences:
                break
            if len(all_symbols) >= vocab_size:
                break

            merged_symbol = best_pair[0] + best_pair[1]
            all_symbols.add(merged_symbol) 

            self.merges.append(best_pair)
            word_freqs = self._merge_pair(best_pair, word_freqs)

        self._build_vocab(all_symbols) 

    def _apply_merge_to_word(self, pair, symbols):
        new_symbols = []
        i = 0
        while i < len(symbols):
            if i < len(symbols) - 1 and (symbols[i], symbols[i + 1]) == pair:
                new_symbols.append(symbols[i] + symbols[i + 1])
                i += 2
            else:
                new_symbols.append(symbols[i])
                i += 1
        return new_symbols
    
    def encode(self, text):
        words = self._pretokenize(text)

        all_ids = []
        for word in words:
            symbols = list(word) 

            for pair in self.merges:
                symbols = self._apply_merge_to_word(pair, symbols)

            for symbol in symbols:
                all_ids.append(self.vocab[symbol])

        return all_ids

    def decode(self, ids):
        symbols = [self.id_to_symbol[i] for i in ids]
        return ''.join(symbols)