"""Mask personal data before it reaches the model or logs; flag answers that leak it."""
import unicodedata
from functools import lru_cache

from presidio_analyzer import AnalyzerEngine, Pattern, PatternRecognizer
from presidio_analyzer.nlp_engine import NlpEngineProvider
from presidio_anonymizer import AnonymizerEngine

import logging
logging.getLogger("presidio-analyzer").setLevel(logging.ERROR)

ENTITIES = ["PERSON", "EMAIL_ADDRESS", "PHONE_NUMBER", "CREDIT_CARD", "IBAN_CODE", "UK_NHS", "UK_POSTCODE"]
MIN_SCORE = 0.5
ALLOW = {"covea", "uinsure", "policy expert"}


@lru_cache(maxsize=1)
@lru_cache(maxsize=1)
def engines():
    nlp = NlpEngineProvider(nlp_configuration={
        "nlp_engine_name": "spacy",
        "models": [{"lang_code": "en", "model_name": "en_core_web_sm"}],
    }).create_engine()
    analyzer = AnalyzerEngine(nlp_engine=nlp, supported_languages=["en"])
    analyzer.registry.add_recognizer(PatternRecognizer(
        supported_entity="UK_POSTCODE",
        patterns=[Pattern("uk_postcode", r"\b[A-Z]{1,2}\d[A-Z\d]?\s?\d[A-Z]{2}\b", 0.6)]))
    analyzer.registry.add_recognizer(PatternRecognizer(
        supported_entity="PHONE_NUMBER",
        patterns=[Pattern("uk_phone", r"(?<!\d)(?:\+44\s?|0)\d(?:\s?\d){8,9}(?!\d)", 0.6)]))
    return analyzer, AnonymizerEngine()


def plain(text):
    return unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode().lower()


def find(text):
    analyzer, _ = engines()
    return [r for r in analyzer.analyze(text=text, language="en", entities=ENTITIES)
            if r.score >= MIN_SCORE and not any(a in plain(text[r.start:r.end]) for a in ALLOW)]


def mask(text):
    _, anonymizer = engines()
    return anonymizer.anonymize(text=text, analyzer_results=find(text)).text


def leaks(answer_texts, source_texts):
    source = plain(" ".join(source_texts))
    return [r.entity_type for t in answer_texts for r in find(t)
            if plain(t[r.start:r.end]) not in source]
