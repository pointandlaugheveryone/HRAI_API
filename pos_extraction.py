from typing import List
from stop_words import get_stop_words
from ufal.udpipe import Model, Pipeline, ProcessingError

from config import conf


stopwords: set[str] = set(get_stop_words('czech'))
tag_model: Model = Model.load(conf.tagger_name)


def text_to_ngrams(text: str) -> List[str]:
    """
    create deduplicated n-grams for matching
    """
    error = ProcessingError()

    pipeline = Pipeline(
        tag_model,
        'tokenize',
        Pipeline.DEFAULT,  # tagger
        Pipeline.DEFAULT,  # parser
        'conllu'
    )

    conllu_text = pipeline.process(text, error)
    if error.occurred(): raise RuntimeError(error.message)

    sentences = []
    cnt_sentence = []
    for line in conllu_text.split('\n'):
        if not line: # inbetween conllu sentences/segments; add already processed sentence, skip
            if cnt_sentence: sentences.append(cnt_sentence)
            cnt_sentence = []
            continue

        # conllu format parsing
        parts = line.split('\t')
        if len(parts) != 10 or \
            '-' in parts[0] or '.' in parts[0] or \
            line.startswith('#'): # generated format comments
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
        tokens = [t for t in sentence if t['upos'] not in {'PUNCT', 'SYM'}]  # exclude non-word tokens
        forms = [t['form'] for t in tokens]
        pos_tags = [t['upos'] for t in tokens]

        for n in range(1, conf.max_ngram + 1):
            # iterate over sentence tokens to create ngrams of decreasing size
            for i in range(len(tokens) - n + 1):
                span_tokens = tokens[i:i + n]
                span_forms = forms[i:i + n]
                span_upos = pos_tags[i:i + n]

                # skill words are mostly nouns; skip ngrams without any
                if not any(tag in {'NOUN', 'ADJ'}
                           for tag in span_upos):
                    continue

                first = span_tokens[0]
                last = span_tokens[-1]
                first_word = first['form'].lower()
                last_word = last['form'].lower()

                # drop spans starting/ending as uncommon POS to reduce ngram count
                if first['upos'] in {'CCONJ', 'SCONJ'} or \
                        last['upos'] in {'CCONJ', 'SCONJ', 'ADJ'} or \
                        first_word in stopwords or \
                        last_word in stopwords:
                    continue

                ngram_text = ' '.join(span_forms).strip()
                if len(ngram_text) < 2: continue  # mostly shortcuts or parsing remainders
                ngrams.add(ngram_text.lower())

    return list(ngrams)