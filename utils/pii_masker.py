"""
PII masking utilities for Singapore personal documents.

Masks NRIC/FIN, phone numbers, names, addresses, dates of birth,
and bank account numbers before text is sent to the LLM.
"""

import re
from datetime import datetime


# ---------------------------------------------------------------------------
# Compiled regex patterns
# ---------------------------------------------------------------------------

# Singapore NRIC / FIN: T/F/G/S/M + 7 digits + letter
NRIC_PATTERN = re.compile(
    r"\b[TtFfGgSsMm]\d{7}[A-Za-z]\b"
)

# Singapore phone numbers: +65 followed by 8 digits (mobile prefixes 6/8/9)
PHONE_PATTERN = re.compile(
    r"\+65\s?[689]\d{7}\b"
)

# Generic phone numbers (local 8-digit)
LOCAL_PHONE_PATTERN = re.compile(
    r"\b[68]\d{7}\b"
)

# Date of birth patterns (common formats in SG statements)
DOB_PATTERNS = [
    re.compile(r"\bDOB[:\s]+(\d{1,2}[-/]\d{1,2}[-/]\d{2,4})\b", re.IGNORECASE),
    re.compile(r"\bDate of Birth[:\s]+(\d{1,2}[-/]\d{1,2}[-/]\d{2,4})\b", re.IGNORECASE),
    re.compile(r"\b(\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{2,4})\b", re.IGNORECASE),
]

# Block + street address (common in Singapore)
ADDRESS_PATTERN = re.compile(
    r"\b\d{1,4}\s+[A-Za-z][A-Za-z\s]{3,40}\s+(?:Road|Street|Avenue|Lane| Boulevard|Close| Drive| Terrace| Way| Hill| Rise| Valley| Grove| Place| Walk| Terrace)\b",
    re.IGNORECASE,
)

# Bank account numbers (6+ digits, possibly with spaces/dashes)
BANK_ACCOUNT_PATTERN = re.compile(
    r"\b\d{3,4}[-\s]?\d{3,4}[-\s]?\d{3,4}\b"
)

# Credit card numbers (13-19 digits, formatted as XXXX-XXXX-XXXX-XXXX or similar)
CREDIT_CARD_PATTERN = re.compile(
    r"\b(?:\d{4}[ -\.]?){3,4}\d{1,4}\b"
)

# Postal codes (6-digit Singapore postal codes)
POSTAL_CODE_PATTERN = re.compile(
    r"\b\d{6}\b"
)

# Labels that typically precede a person's name
NAME_LABELLED_AS = [
    "issued to",
    "patient",
    "to:",
    "to ",
    "name:",
    "name ",
    "policy holder",
    "insured",
    "member name",
    "card holder",
    "employee",
    "beneficiary",
    "payer",
    "from:",
]

# Generic name patterns: Title Firstname Lastname
NAME_PATTERN = re.compile(
    r"\b(Mr|Mrs|Ms|Mdm|Dr|Prof)\.\s+[A-Z][a-z]+\s+[A-Z][a-zA-Z]+\b"
)


# ---------------------------------------------------------------------------
# Masking functions
# ---------------------------------------------------------------------------

def mask_nric(text: str) -> str:
    """Replace NRIC/FIN numbers with [NRIC]."""
    return NRIC_PATTERN.sub("[NRIC]", text)


def mask_phone(text: str) -> str:
    """Replace Singapore mobile and local phone numbers with [PHONE]."""
    text = PHONE_PATTERN.sub("[PHONE]", text)
    text = LOCAL_PHONE_PATTERN.sub("[PHONE]", text)
    return text


def mask_dob(text: str) -> str:
    """Replace dates of birth with [DOB]."""
    for pattern in DOB_PATTERNS:
        text = pattern.sub("[DOB]", text)
    return text


def mask_address(text: str) -> str:
    """Replace block + street addresses and postal codes with placeholders."""
    text = ADDRESS_PATTERN.sub("[ADDRESS]", text)
    text = POSTAL_CODE_PATTERN.sub("[POSTAL_CODE]", text)
    return text


def mask_bank(text: str) -> str:
    """Replace potential bank account numbers with [BANK_ACCOUNT]."""
    # Only mask if the number looks like a bank account (not generic invoice numbers)
    # Heuristic: bank accounts are often in groups of 3-4 digits
    matches = list(BANK_ACCOUNT_PATTERN.finditer(text))
    for match in matches:
        # Only mask if preceded/followed by banking-related context
        start, end = match.span()
        context = text[max(0, start - 30): min(len(text), end + 30)].lower()
        banking_keywords = ["account", "bank", "giro", "transfer", "pay", "payment", "credit", "debit"]
        matched_value = match.group()
        if any(kw in context for kw in banking_keywords):
            text = text.replace(matched_value, "[BANK_ACCOUNT]")
        else:
            # Replace long digit sequences that aren't obviously invoice/amount numbers
            cleaned = matched_value.replace(" ", "").replace("-", "")
            if len(cleaned) >= 9 and not any(c.isalpha() for c in matched_value):
                text = text.replace(matched_value, "[BANK_ACCOUNT]")
    return text


def mask_credit_card(text: str) -> str:
    """Replace potential credit card numbers with [CREDIT_CARD]."""
    def _is_credit_card(cleaned: str) -> bool:
        # Must be 13-19 digits and pass Luhn check
        if not cleaned.isdigit() or len(cleaned) < 13 or len(cleaned) > 19:
            return False
        # Luhn algorithm
        total = 0
        reverse_digits = cleaned[::-1]
        for i, digit in enumerate(reverse_digits):
            n = int(digit)
            if i % 2 == 1:
                n *= 2
                if n > 9:
                    n -= 9
            total += n
        return total % 10 == 0

    result = text
    for match in CREDIT_CARD_PATTERN.finditer(text):
        cleaned = match.group().replace(" ", "").replace("-", "").replace(".", "")
        if _is_credit_card(cleaned):
            result = result.replace(match.group(), "[CREDIT_CARD]")
    return result


def mask_names(text: str) -> str:
    """Replace labelled names (Patient:, Issued to:, etc.) with [NAME]."""
    result = text
    for label in NAME_LABELLED_AS:
        pattern = re.compile(re.escape(label) + r"\s+([A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+){0,2})", re.IGNORECASE)
        result = pattern.sub(label + " [NAME]", result)

    # Also mask "Mr. Name Name" / "Mrs. Name Name" patterns
    result = NAME_PATTERN.sub("[NAME]", result)
    return result


def mask_all(text: str) -> str:
    """
    Apply all PII masks in sequence.
    Order matters: names first (so labels don't get caught by other patterns).
    """
    text = mask_names(text)
    text = mask_nric(text)
    text = mask_phone(text)
    text = mask_dob(text)
    text = mask_address(text)
    text = mask_credit_card(text)
    text = mask_bank(text)
    return text
