from typing import List
from stop_words import get_stop_words
from ufal.udpipe import Model, Pipeline, ProcessingError

from config import conf

# Preload Czech stopwords for quick token filtering.
stopwords: set[str] = set(get_stop_words('czech'))
# Load the UDPipe model once at import time to avoid repeated disk reads.
tag_model: Model = Model.load(conf.tagger_name)
# Tokenize + tag + parse into CoNLL-U for structured POS/dependency extraction.
pipeline = Pipeline(
        tag_model,
        'tokenize',
        Pipeline.DEFAULT,  # tagger
        Pipeline.DEFAULT,  # parser
        'conllu'
    )

def text_to_ngrams(text: str) -> List[str]:
    error = ProcessingError()

    conllu_text = pipeline.process(text, error)
    if error.occurred(): raise RuntimeError(error.message)

    sentences = []
    cnt_sentence = []
    for line in conllu_text.split('\n'):
        if not line:  # inbetween conllu sentences/segments; add already processed sentence, skip
            if cnt_sentence: sentences.append(cnt_sentence)
            cnt_sentence = []
            continue

        # Parse one CoNLL-U token line into a structured dict.
        parts = line.split('\t')
        if len(parts) != 10 or \
                '-' in parts[0] or '.' in parts[0] or \
                line.startswith('#'):  # generated format comments
            continue

        token = {
            'id': int(parts[0]),
            'form': parts[1],  # exact original text word
            'lemma': parts[2],  # declension-free text form
            'upos': parts[3],  # POS tag
            'head': int(parts[6]) if parts[6].isdigit() else None,
            # depencency tree position (základní sklad. dvojice atd)
            'deprel': parts[7],
            # token and its head's relationship (if any); head and deprel make up input's dependency tree
        }
        cnt_sentence.append(token)

    if cnt_sentence:
        sentences.append(cnt_sentence)

    ngrams = set()
    for sentence in sentences:
        # Ignore punctuation and symbols so spans contain real words only.
        tokens = [t for t in sentence if t['upos'] not in {'PUNCT', 'SYM'}]
        forms = [t['form'] for t in tokens]
        pos_tags = [t['upos'] for t in tokens]

        for n in range(1, conf.max_ngram + 1):
            # sliding window across tokens to form candidate spans
            for i in range(len(tokens) - n + 1):
                span_tokens = tokens[i:i + n]
                span_forms = forms[i:i + n]
                span_upos = pos_tags[i:i + n]

                first = span_tokens[0]
                last = span_tokens[-1]
                first_word = first['form'].lower()
                last_word = last['form'].lower()

                # Skill phrases are usually noun-like; skip spans without nouns/proper nouns/adjectives
                if (not any(tag in {'NOUN'} for tag in span_upos) or
                        not any(tag in {'ADJ','PROPN','VERB'} for tag in span_upos)):
                    continue

                # Filter spans with conjunctions, stopwords, or trailing verbs/adverbs.
                if first['upos'] in {'CCONJ', 'SCONJ', 'NUM','DET'} or \
                        last['upos'] in {'CCONJ', 'SCONJ', 'ADJ', 'ADV', 'VERB', 'NUM','AUX','ADV'} or \
                        first_word in stopwords or \
                        last_word in stopwords:
                    continue

                ngram_text = ' '.join(span_forms).strip()
                if len(ngram_text) < 2: continue  # mostly shortcuts or parsing remainders
                ngrams.add(ngram_text.lower())

    return list(ngrams)


if __name__ == "__main__":
    print(pipeline.process("jakéhokoliv přinejmenším"))