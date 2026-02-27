from models import Skill
from lookups import load_data, query
from settings import config

import os, re
from functools import lru_cache
from typing import Any, Dict, List, Sequence, Union

from stop_words import get_stop_words
from ufal.udpipe import Model, Pipeline


stopwords_cz = set(get_stop_words('czech'))
pdt_path = os.path.join(config.base_dir, 'data', 'czech-pdt-ud-2.5.udpipe')


def conllu_to_tokenized(conll_text: str) -> List[List[Dict[str, Any]]]:  # {sentence:tokens}
    sentences: List[List[Dict[str, Any]]] = []
    sentence = []

    for line in conll_text.split('\n'):  # new sentence
        if not line.strip():
            if sentence:
                sentences.append(sentence)
                sentence = []
            continue

        # conll format checks
        parts = line.split('\t')
        if len(parts) != 10: continue
        if '-' in parts[0] or '.' in parts[0]: continue
        if line.startswith('#'): continue

        token = {
            'id': int(parts[0]),
            'form': parts[1],   # exact original text word
            'lemma': parts[2],  # declension-free text form
            'upos': parts[3],   # POS tag
            'head': int(parts[6])
            if parts[6].isdigit()
            else None,
            'deprel': parts[7]  # token and its head's (nadřazené slovo) relationship (if any is found)
                                # depencency tree position (základní sklad. dvojice atd)
        }
        sentence.append(token)

    if sentence: sentences.append(sentence)
    return sentences


@lru_cache(maxsize=1)
def load_udpipe() -> Pipeline:
    model = Model.load(pdt_path)
    return Pipeline(
        model,
        'tokenize',
        Pipeline.DEFAULT,  # tagging
        Pipeline.DEFAULT,  # parsing
        'conllu'
    )


def create_noun_ngrams(text: str, max_ngram: int = 3) -> List[str]:
    if not text:
        return []
    conllu_text = load_udpipe() .process(
        text.replace('-', ' ')
    )
    sentences = conllu_to_tokenized(conllu_text)

    # this is horrifyingly unoptimal but saves more time compared to encoding all the ngrams
    ngrams = set()
    for sentence in sentences:
        tokens = [t for t in sentence if t['upos'] not in {'PUNCT', 'SYM'}] # exclude non-word tokens
        forms = [t['form'] for t in tokens]
        pos_tags = [t['upos'] for t in tokens]

        for n in range(1, max_ngram + 1):
            # iterate over sentence tokens to create ngrams of decreasing size
            for i in range(len(tokens) - n + 1):
                span_tokens = tokens[i:i + n]
                span_forms = forms[i:i + n]
                span_upos = pos_tags[i:i + n]

                # since skill words are mostly nouns, skip ngrams without any
                if not any(tag in {'NOUN'} for tag in span_upos): continue

                first = span_tokens[0]
                last = span_tokens[-1]
                first_word = first['form'].lower()
                last_word = last['form'].lower()

                # drop spans starting/ending with conjunctions, stopwords, or dates
                if first['upos'] in {'CCONJ', 'SCONJ'} or \
                    last['upos'] in {'CCONJ', 'SCONJ'} or \
                    first_word in stopwords_cz or \
                    last_word in stopwords_cz or \
                    re.fullmatch(r'\d{1,2}|\d{4}', first_word):
                    continue

                ngram_text = ' '.join(span_forms).strip()
                if len(ngram_text) < 2: continue # shortcuts
                ngrams.add(ngram_text.lower())

    return sorted(ngrams)


def _normalize_text_input(raw: Union[str, Sequence[str]]) -> str:
    if isinstance(raw, (list, tuple, set)):
        return '\n'.join(str(item) for item in raw)
    return raw or ''


def match_skill_texts(text: Union[str, Sequence[str]], encoder, top_n: int, result_n: int = 30) -> List[Skill]:
    indexes = load_data()
    normalized_text = _normalize_text_input(text)
    texts = create_noun_ngrams(normalized_text)
    if not texts:
        return []
    encoded = encoder.encode(
        texts,
        normalize_embeddings=config.normalize_embeddings,
        convert_to_numpy=True,
    )

    results: List[Skill] = []
    faiss_results = query(indexes['skill']['index'],
                          indexes['skill']['metadata'],
                          encoded, top_n)

    for text, row in zip(texts, faiss_results):
        if not row: continue
        best_meta, score = row[0]
        results.append(
            Skill(
                id=best_meta.get('id'),
                esco_uri=best_meta.get('esco_uri', ''),
                label=best_meta.get('preferred_label', ''),
                relation_type='',
                score=score,
                source_text=text,
            )
        )

    res_sorted = sorted(results, key=lambda m: m.score, reverse=True)
    unique: List[Skill] = []

    # deduplicate (ngram substrings that are ngrams themselves but have lower score)
    for ngram in res_sorted:
        if any(ngram.source_text in existing.source_text and ngram.score < existing.score
               or existing.source_text in ngram.source_text
               for existing in unique
               ):
            continue
        unique.append(ngram)

    unique.sort(key=lambda r: r.score, reverse=True)
    return unique[:result_n]
