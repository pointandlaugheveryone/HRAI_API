import os
import re
from functools import lru_cache
from typing import List
from stop_words import get_stop_words
from ufal.udpipe import Model, Pipeline, ProcessingError
import spacy

from datamodels import MatchedSkill
from lookups import load_data, query
from settings import config

stopwords_cz = set(get_stop_words("czech"))
_base_dir = os.path.dirname(os.path.abspath(__file__))
pdt_path = os.path.join(_base_dir, "models", "czech-pdt-ud-2.5.udpipe")
spacy_path = os.path.join(_base_dir, "models", "spacy")


def input_to_conllu(text: str, pipeline: Pipeline) -> str:
    error = ProcessingError()
    processed = pipeline.process(text, error)
    if error.occurred():
        raise RuntimeError(error.message)
    return processed


def conllu_to_tokenized(conll_text: str):  # {sentence:tokens}
    sentences = []
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
            'form': parts[1],  # exact original text word
            'lemma': parts[2],  # declension-free text form
            'upos': parts[3],  # POS tag
            'head': int(parts[6]) if parts[6].isdigit() else None,
            # depencency tree position (základní sklad. dvojice atd)
            'deprel': parts[7],
            # token and its head's relationship (if any); head and deprel make up input's dependency tree
        }
        sentence.append(token)

    if sentence:
        sentences.append(sentence)

    return sentences


@lru_cache(maxsize=1)
def udpipe_pipeline() -> Pipeline:
    model = Model.load(pdt_path)
    return Pipeline(
        model,
        'tokenize',
        Pipeline.DEFAULT,  # tagging
        Pipeline.DEFAULT,  # parsing
        'conllu'
    )


def extract_noun_ngrams(text, max_ngram=3) -> List[str]:
    pipeline = udpipe_pipeline()
    text = text.replace('-', ' ')
    conllu_text = input_to_conllu(text, pipeline)
    sentences = conllu_to_tokenized(conllu_text)

    ngrams = set()
    for sentence in sentences:
        tokens = [t for t in sentence if t['upos'] not in {'PUNCT', 'SYM'}]
        forms = [t['form'] for t in tokens]
        pos_tags = [t['upos'] for t in tokens]

        for n in range(1, max_ngram + 1):
            for i in range(len(tokens) - n + 1):
                span_tokens = tokens[i:i + n]
                span_forms = forms[i:i + n]
                span_upos = pos_tags[i:i + n]

                if not any(tag in {'NOUN', 'PROPN'} for tag in span_upos):
                    continue

                first_token = span_tokens[0]
                last_token = span_tokens[-1]
                first_word = first_token["form"].lower()
                last_word = last_token["form"].lower()

                if first_token["upos"] in {"CCONJ", "SCONJ"} or \
                        last_token["upos"] in {"CCONJ", "SCONJ"} or \
                        first_word in stopwords_cz or \
                        last_word in stopwords_cz or \
                        re.fullmatch(r"\d{1,2}|\d{4}", first_word):
                    continue

                ngram_text = ' '.join(span_forms).strip()
                if len(ngram_text) < 2:
                    continue

                ngrams.add(ngram_text.lower())

    return sorted(ngrams)


def extract_spacy_skills(text) -> List[str]:
    nlp = spacy.load(spacy_path)
    doc = nlp(text)
    return sorted({ent.text.strip() for ent in doc.ents
                   if ent.label_.lower() == 'skill'})


def deduplicate(results):
    # brings down inaccuracy a lot
    # the time it takes to deduplicate is also lower than encoding hundreds of ngrams
    res_sorted = sorted(results, key=lambda m: m.score, reverse=True)
    unique = []
    for ngram in res_sorted:
        if any(
                ngram.source_text in existing.source_text
                or existing.source_text in ngram.source_text
                for existing in unique
        ):
            continue
        unique.append(ngram)
    return unique


def match_skill_texts(
        texts,
        encoder,
        source,
        top_n,
        result_n = 30) -> List[MatchedSkill]:
    if not texts: return []
    indexes = load_data()

    encoded = encoder.encode(
        texts,
        normalize_embeddings=config.normalize_embeddings,
        convert_to_numpy=True,
    )

    results = []
    faiss_results = query(indexes['skill']['index'],
                          indexes['skill']['meta'],
                          encoded, top_n)
    for text, row in zip(texts, faiss_results):
        if not row: continue
        best_meta, score = row[0]
        results.append(
            MatchedSkill(
                id=best_meta.get('id'),
                esco_uri=best_meta.get('esco_uri', ''),
                label=best_meta.get('preferred_label', ''),
                score=score,
                source_text=text,
                source=source,
            )
        )

    results = deduplicate(results)
    results.sort(key=lambda r: r.score, reverse=True)
    return results[:result_n]


def extract_skill_ents(text: str, encoder, use_spacy: bool, top_n: int) -> List[MatchedSkill]:
    if use_spacy:
        skill_texts = extract_spacy_skills(text)
        source = 'spacy'
    else:
        skill_texts = extract_noun_ngrams(text)
        source = 'ngram'
    return match_skill_texts(skill_texts, encoder, top_n=top_n, source=source)
