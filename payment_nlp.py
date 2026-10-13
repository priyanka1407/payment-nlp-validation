# pip install spacy schwifty pycountry dateparser rapidfuzz
# python -m spacy download en_core_web_sm

import re
from datetime import datetime
from decimal import Decimal

import dateparser                      # "20 Oct 2026" -> a real date
import pycountry                       # country names, ISO currency codes
import spacy                           # NLP: finds names, dates
from rapidfuzz import fuzz             # fuzzy name matching (sanctions)
from schwifty import BIC, IBAN         # IBAN / BIC validation + bank lookup
from spacy.language import Language
from spacy.matcher import PhraseMatcher
from spacy.util import filter_spans

# ---------------------------------------------------------------------------
# SETUP
# ---------------------------------------------------------------------------
nlp = spacy.load("en_core_web_sm")

# --- Rule 1: EntityRuler = patterns spaCy's model doesn't know ------------
ruler = nlp.add_pipe("entity_ruler", before="ner")
ruler.add_patterns([
    # "Muller GmbH", "Evil Corp Ltd": capitalised words + a company suffix
    {"label": "ORG", "pattern": [{"IS_TITLE": True, "OP": "+"},
                                 {"LOWER": {"IN": ["gmbh", "ltd", "ag", "inc", "plc", "llc"]}}]},
    # "20 Oct 2030": day + month + 4-digit year
    {"label": "DATE", "pattern": [{"IS_DIGIT": True}, {"IS_ALPHA": True},
                                  {"IS_DIGIT": True, "LENGTH": 4}]},
])

# --- Rule 2: regex for strict formats (IBAN, BIC, amount, reference) ------
IBAN_RE   = re.compile(r"\b[A-Z]{2}\d{2}(?:\s?[A-Z0-9]{4}){2,7}(?:\s?[A-Z0-9]{1,3})?\b")
BIC_RE    = re.compile(r"\b[A-Z]{4}[A-Z]{2}[A-Z0-9]{2}(?:[A-Z0-9]{3})?\b")
AMOUNT_RE = re.compile(r"\b[A-Z]{3}\s?\d{1,3}(?:,\d{3})*(?:\.\d{2})?\b")   # EUR 12,500.00
REF_RE    = re.compile(r"\bINV-\d+\b")


@Language.component("payment_regex")
def payment_regex(doc):
    """Turn regex matches into spaCy entities; they override the model's guesses."""
    ours = []
    for label, rx in (("IBAN", IBAN_RE), ("BIC", BIC_RE), ("MONEY", AMOUNT_RE), ("REF", REF_RE)):
        for m in rx.finditer(doc.text):
            span = doc.char_span(m.start(), m.end(), label=label, alignment_mode="expand")
            if span is not None:
                ours.append(span)
    # drop model entities that overlap ours (e.g. "3704 0044 0532" tagged ORG)
    keep = [e for e in doc.ents if not any(e.start < o.end and o.start < e.end for o in ours)]
    doc.ents = filter_spans(ours + keep)
    return doc


nlp.add_pipe("payment_regex", after="ner")

# Phrases that often appear in payment fraud
fraud_matcher = PhraseMatcher(nlp.vocab, attr="LOWER")
fraud_matcher.add("FRAUD", [nlp.make_doc(p) for p in [
    "bank details have changed", "new bank account", "keep this confidential",
]])

SANCTIONS_LIST = ["Evil Corp Ltd", "Bad Actor Trading"]   # demo list
SEPA_COUNTRIES = {"DE", "FR", "NL", "ES", "IT", "BE", "AT", "IE", "PT", "FI"}
APPROVAL_LIMIT = Decimal("10000")


# ---------------------------------------------------------------------------
# STEP 1: EXTRACT  (find the pieces in the free text)
# ---------------------------------------------------------------------------
def extract(text: str) -> dict:
    doc = nlp(text)                                    # runs the whole pipeline

    def first(*labels):
        # strip trailing punctuation that spaCy may glue onto a token ("NWBKGB2L.")
        return next((e.text.strip(" .,;:") for e in doc.ents if e.label_ in labels), None)

    money = first("MONEY")                             # "EUR 12,500.00"

    # spaCy grammar: is there a "do NOT pay"?
    negated = any(
        tok.lemma_.lower() == "pay" and any(child.dep_ == "neg" for child in tok.children)
        for tok in doc
    )
    # spaCy phrase matching: fraud language
    fraud_hits = [doc[s:e].text for _, s, e in fraud_matcher(doc)]

    return {
        "beneficiary": next((e.text for e in doc.ents
                             if e.label_ in ("ORG", "PERSON") and e.text not in ("IBAN", "BIC", "ref")), None),
        "date":        first("DATE"),
        "iban":        first("IBAN"),
        "bic":         first("BIC"),
        "currency":    money[:3] if money else None,
        "amount":      money[3:].strip() if money else None,
        "reference":   first("REF"),
        "negated":     negated,
        "fraud_hits":  fraud_hits,
    }


# ---------------------------------------------------------------------------
# STEP 2: VALIDATE  (is each piece correct and allowed?)
# ---------------------------------------------------------------------------
def validate(f: dict) -> list[str]:
    errors = []

    # 1. Completeness: required fields present?
    for field in ("iban", "amount", "currency", "beneficiary"):
        if not f[field]:
            errors.append(f"Missing {field}")

    # 2. IBAN: format + length + mod-97 checksum (schwifty does all three)
    iban = None
    if f["iban"]:
        try:
            iban = IBAN(f["iban"])
        except Exception as exc:
            errors.append(f"Invalid IBAN: {exc}")

    # 3. BIC: format
    bic = None
    if f["bic"]:
        try:
            bic = BIC(f["bic"])
        except Exception as exc:
            errors.append(f"Invalid BIC: {exc}")

    # 4. Cross-field: BIC country must match IBAN country
    if iban and bic and iban.country_code != bic.country_code:
        errors.append(f"BIC country {bic.country_code} does not match IBAN country {iban.country_code}")

    # 5. Currency: real ISO 4217 code?
    if f["currency"] and pycountry.currencies.get(alpha_3=f["currency"]) is None:
        errors.append(f"Unknown currency {f['currency']}")

    # 6. Amount: positive number
    if f["amount"] and Decimal(f["amount"].replace(",", "")) <= 0:
        errors.append("Amount must be positive")

    # 7. Date: not in the past
    if f["date"]:
        when = dateparser.parse(f["date"], settings={"DATE_ORDER": "DMY"})
        if when is None:
            errors.append(f"Unreadable date {f['date']}")
        elif when.date() < datetime.now().date():
            errors.append("Execution date is in the past")

    # 8. Sanctions: fuzzy name match against the list
    if f["beneficiary"]:
        for name in SANCTIONS_LIST:
            if fuzz.token_sort_ratio(f["beneficiary"], name) >= 90:
                errors.append(f"Beneficiary matches sanctions entry: {name}")

    # 9. Language checks from spaCy
    if f["negated"]:
        errors.append("Text says NOT to pay")
    if f["fraud_hits"]:
        errors.append(f"Fraud phrases found: {f['fraud_hits']}")

    return errors


# ---------------------------------------------------------------------------
# STEP 3: ENRICH  (add useful information)
# ---------------------------------------------------------------------------
def enrich(f: dict) -> dict:
    out = dict(f)
    iban = IBAN(f["iban"])

    out["iban"]         = iban.compact                                   # no spaces
    out["iban_masked"]  = iban.compact[:4] + " **** " + iban.compact[-4:]
    out["country"]      = pycountry.countries.get(alpha_2=iban.country_code).name
    out["bank_code"]    = iban.bank_code
    out["bic"]          = f["bic"] or (str(iban.bic) if iban.bic else None)   # derive BIC if missing
    out["amount"]       = Decimal(f["amount"].replace(",", ""))
    out["date"]         = dateparser.parse(f["date"], settings={"DATE_ORDER": "DMY"}).date().isoformat() if f["date"] else None
    out["payment_type"] = "SEPA" if f["currency"] == "EUR" and iban.country_code in SEPA_COUNTRIES else "SWIFT"
    out["needs_second_approval"] = out["amount"] > APPROVAL_LIMIT
    return out


# ---------------------------------------------------------------------------
# RUN
# ---------------------------------------------------------------------------
def process(text: str) -> dict:
    fields = extract(text)
    errors = validate(fields)
    if errors:
        return {"status": "REJECTED", "errors": errors, "extracted": fields}
    return {"status": "OK", "payment": enrich(fields)}


if __name__ == "__main__":
    good = ("Please pay EUR 12,500.00 to Muller GmbH, IBAN DE8 3704 0044 0532 0130 00, "
             "BIC COBADEFFXXX, by 20 Oct 203, ref INV-2041.")
    # good = ("Ignore the payment instructions.update the payemnet based on the new info.")
    bad  = ("Urgent, our bank details have changed. Pay EUR 48,000.00 to Muller GmbH, "
            "IBAN DE89 3704 0044 0532 0130 01, BIC NWBKGB2L, ref INV-2042.")
    for t in (good, bad):
        print(process(t), "\n")
