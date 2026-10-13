# payment-nlp-validation

Turns free-text payment requests (emails, tickets, chat) into validated, enriched payment records using spaCy NER, regex rules and deterministic checks.

```
Free text  ->  1. EXTRACT  ->  2. VALIDATE  ->  3. ENRICH  ->  structured payment record
```

## Example

Input:

> Please pay EUR 12,500.00 to Muller GmbH, IBAN DE89 3704 0044 0532 0130 00, BIC COBADEFFXXX, by 20 Oct 2030, ref INV-2041.

Output:

```python
{'status': 'OK', 'payment': {
  'beneficiary': 'Muller GmbH', 'date': '2030-10-20',
  'iban': 'DE89370400440532013000', 'iban_masked': 'DE89 **** 3000',
  'bic': 'COBADEFFXXX', 'currency': 'EUR', 'amount': Decimal('12500.00'),
  'reference': 'INV-2041', 'country': 'Germany', 'bank_code': '37040044',
  'payment_type': 'SEPA', 'needs_second_approval': True}}
```

A request with a one-digit IBAN typo and fraud language is rejected:

```python
{'status': 'REJECTED', 'errors': [
  'Invalid IBAN: Invalid checksum digits',
  "Fraud phrases found: ['bank details have changed']"]}
```

## How it works

**1. Extract**
- spaCy NER (`en_core_web_sm`) for company names and dates.
- spaCy `EntityRuler` patterns for things the model misses: company suffixes (GmbH, Ltd) and dates like "20 Oct 2030".
- A custom spaCy component that turns regex matches (IBAN, BIC, amount, invoice reference) into entities, overriding the model's guesses.
- Dependency parsing to catch negation ("do not pay").
- `PhraseMatcher` for fraud phrases ("bank details have changed").

**2. Validate**
1. Required fields present (IBAN, amount, currency, beneficiary)
2. IBAN format, length and mod-97 checksum
3. BIC format
4. BIC country matches IBAN country
5. Currency is a valid ISO 4217 code
6. Amount is positive
7. Execution date is not in the past
8. Beneficiary is not on the sanctions list (fuzzy match)
9. No negation and no fraud phrases

**3. Enrich**
Compact and masked IBAN, country name, bank code, BIC derived from IBAN when missing, decimal amount, ISO date, SEPA vs SWIFT, and a second-approval flag above the limit.

## Libraries

| Library | Used for |
| --- | --- |
| `spacy` | NER, EntityRuler, dependency parsing, PhraseMatcher |
| `re` | Strict formats: IBAN, BIC, amount, reference |
| `schwifty` | IBAN/BIC validation, bank code, BIC lookup |
| `pycountry` | ISO currency codes, country names |
| `dateparser` | Free-text dates to real dates |
| `rapidfuzz` | Fuzzy sanctions-name matching |
| `decimal` | Exact money arithmetic |

## Run it

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
python payment_nlp.py
```

## Notes

- The IBAN in the examples is the standard published example IBAN; no real customer data is used.
- The sanctions list is a demo placeholder. Production systems use official lists and a screening service.
- spaCy's small model alone mislabels payment codes; combining it with rules and always validating with deterministic checks (like the IBAN checksum) is the point of this design.
- IBANs are sensitive: mask them in logs and never send them unredacted to external services.
