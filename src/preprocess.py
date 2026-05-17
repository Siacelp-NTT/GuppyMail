"""
Task 1.2 — Email Preprocessing/Cleaning Pipeline

Cleans raw emails by removing HTML, signatures, thread history, legal disclaimers,
and noise. Also provides optional Vietnamese→English translation for Gmail input.

Usage:
    conda run -n email-summarizer python src/preprocess.py
"""

import re
import json
import os
import hashlib
from tqdm import tqdm


def remove_html_tags(text: str) -> str:
    """Remove HTML tags and decode common entities."""
    text = re.sub(r'<[^>]+>', ' ', text)
    entities = {
        '&nbsp;': ' ', '&amp;': '&', '&lt;': '<', '&gt;': '>',
        '&quot;': '"', '&#39;': "'", '&apos;': "'",
        '\n': '\n', '\t': '\t',
    }
    for entity, char in entities.items():
        text = text.replace(entity, char)
    return text


def remove_signatures(text: str) -> str:
    """Remove email signatures using common patterns."""
    patterns = [
        r'\n[-=_]{2,}\s*\n.*$',           # Separator lines
        r'\n--\s*\n[\s\S]*$',              # Standard signature separator
        r'\nBest regards?[\s,]*[\s\S]*$',  # Common sign-offs
        r'\nSincerely[\s,]*[\s\S]*$',
        r'\nRegards?[\s,]*[\s\S]*$',
        r'\nThanks?[\s,]*[\s\S]*$',
        r'\nThank you[\s,]*[\s\S]*$',
        r'\nSent from my [\s\S]*$',        # Mobile signatures
        r'\nGet Outlook for [\s\S]*$',
        r'\nSent from Outlook [\s\S]*$',
        r'\nSent from my iPhone[\s\S]*$',
        r'\nSent from my Android[\s\S]*$',
        r'\n-{2,}\s*\n.*?(?:\n|$)',        # Any dashed separator block
    ]
    for pattern in patterns:
        text = re.sub(pattern, '', text, flags=re.IGNORECASE | re.MULTILINE)
    return text


def remove_thread_history(text: str) -> str:
    """Remove forwarded/replied email thread history."""
    thread_markers = [
        r'\n-{3,} ?Original Message ?-{3,}',
        r'\n>{1,}.*$',                       # Quoted replies
        r'\nOn .*? wrote:.*?\n',             # "On Mon, Jan 1 wrote:"
        r'\nFrom: .*?\nSent: .*?\nTo: .*?\nSubject:.*?\n',
        r'\n-----Original Message-----',
        r'\n_{3,}.*?\n',                     # Underscore separators
        r'Forwarded by.*?\n',
        r'\nTo: .*?\nFrom: .*?\nSubject:.*?\n',
        r'\n\s*From: .*?\n\s*To: .*?\n\s*Subject:.*?\n',
    ]
    for marker in thread_markers:
        match = re.search(marker, text, re.IGNORECASE | re.DOTALL)
        if match:
            text = text[:match.start()]
            break
    return text


def remove_legal_disclaimers(text: str) -> str:
    """Remove legal disclaimers and confidentiality notices."""
    patterns = [
        r'(?:CONFIDENTIALITY|PRIVILEGED|DISCLAIMER|NOTICE).{0,800}$',
        r'This email and any attachments.{0,800}$',
        r'This message is intended only.{0,800}$',
        r'If you are not the intended recipient.{0,800}$',
        r'This communication is confidential.{0,800}$',
        r'The contents of this email.{0,800}$',
    ]
    for pattern in patterns:
        text = re.sub(pattern, '', text, flags=re.IGNORECASE | re.DOTALL)
    return text


def clean_whitespace(text: str) -> str:
    """Normalize whitespace."""
    text = re.sub(r'\n{3,}', '\n\n', text)   # Max 2 consecutive newlines
    text = re.sub(r' {2,}', ' ', text)       # Single spaces
    text = re.sub(r'\t+', ' ', text)        # Tabs to spaces
    text = text.strip()
    return text


def extract_email_parts(text: str) -> dict:
    """Extract subject, sender, date from email headers if present."""
    result = {'subject': '', 'from': '', 'date': '', 'body': text}

    subject_match = re.search(r'Subject:\s*(.+)', text, re.IGNORECASE)
    from_match = re.search(r'(?:From|Sender):\s*(.+)', text, re.IGNORECASE)
    date_match = re.search(r'(?:Date|Sent):\s*(.+)', text, re.IGNORECASE)

    if subject_match:
        result['subject'] = subject_match.group(1).strip()

    if from_match:
        result['from'] = from_match.group(1).strip()

    if date_match:
        result['date'] = date_match.group(1).strip()

    # Remove header block if present
    header_end = re.search(r'\n\n', text)
    if header_end and len(text[:header_end.start()]) < 500:
        # Only treat as header if it's reasonably short
        if re.search(r'^(From|To|Subject|Date|Message-ID):', text, re.MULTILINE | re.IGNORECASE):
            result['body'] = text[header_end.end():].strip()
    else:
        result['body'] = text.strip()

    return result


def preprocess_email(raw_text: str) -> dict:
    """
    Full preprocessing pipeline for a single email.
    Returns dict with cleaned_body, subject, from, date.
    """
    text = raw_text
    text = remove_html_tags(text)
    text = remove_thread_history(text)
    text = remove_signatures(text)
    text = remove_legal_disclaimers(text)
    text = clean_whitespace(text)

    parts = extract_email_parts(text)
    return parts


def translate_vietnamese_to_english(text: str) -> str:
    """
    Translate Vietnamese text to English.
    Used for Gmail integration when Vietnamese emails are detected.
    For Enron dataset (all English), this is skipped.
    """
    try:
        from deep_translator import GoogleTranslator
        translator = GoogleTranslator(source='vi', target='en')
        return translator.translate(text)
    except Exception as e:
        print(f"Translation failed: {e}")
        return text  # fallback


def detect_language(text: str) -> str:
    """
    Simple heuristic language detection.
    Returns 'vi' if Vietnamese characters found, else 'en'.
    """
    # Vietnamese-specific characters
    vietnamese_chars = set('àáảãạăằắẳẵặâầấẩẫậèéẻẽẹêềếểễệìíỉĩịòóỏõọôồốổỗộơờớởỡợùúủũụưừứửữựỳýỷỹỵđ')
    text_lower = text.lower()
    for char in text_lower:
        if char in vietnamese_chars:
            return 'vi'
    return 'en'


def dedupe_key(text: str) -> str:
    """Stable hash for exact cleaned-body deduplication."""
    normalized = re.sub(r'\s+', ' ', text).strip().lower()
    return hashlib.md5(normalized.encode('utf-8')).hexdigest()


def process_dataset(input_path: str, output_path: str, translate_vi: bool = False, dedupe: bool = True):
    """
    Process entire dataset from JSONL input to JSONL output.
    
    Args:
        input_path: Path to raw emails JSONL
        output_path: Path to save cleaned emails JSONL
        translate_vi: If True, auto-translate Vietnamese emails to English
        dedupe: If True, skip duplicate cleaned bodies after preprocessing
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    cleaned_count = 0
    skipped_count = 0
    duplicate_count = 0
    translated_count = 0
    results = []
    seen_cleaned = set()

    with open(input_path, 'r') as f:
        lines = f.readlines()

    for line in tqdm(lines, desc="Processing emails"):
        email = json.loads(line)
        raw_text = email.get('text', email.get('message', email.get('body', '')))

        if not raw_text or len(raw_text) < 200:
            skipped_count += 1
            continue

        # Preprocess
        parts = preprocess_email(raw_text)
        cleaned = parts['body']

        if len(cleaned) < 100:
            skipped_count += 1
            continue

        # Optional: translate Vietnamese emails
        lang = detect_language(cleaned)
        if translate_vi and lang == 'vi':
            cleaned = translate_vietnamese_to_english(cleaned)
            translated_count += 1

        if dedupe:
            key = dedupe_key(cleaned)
            if key in seen_cleaned:
                duplicate_count += 1
                continue
            seen_cleaned.add(key)

        record = {
            'original_text': raw_text,
            'cleaned_body': cleaned,
            'subject': parts.get('subject', ''),
            'from': parts.get('from', ''),
            'date': parts.get('date', ''),
            'original_length': len(raw_text),
            'cleaned_length': len(cleaned),
            'language': lang,
        }
        results.append(record)
        cleaned_count += 1

    with open(output_path, 'w') as f:
        for record in results:
            f.write(json.dumps(record) + '\n')

    print(f"\nProcessing complete:")
    print(f"  Cleaned:   {cleaned_count}")
    print(f"  Skipped:   {skipped_count}")
    if dedupe:
        print(f"  Duplicates:{duplicate_count}")
    if translate_vi:
        print(f"  Translated: {translated_count}")
    print(f"  Saved to:  {output_path}")

    # Save a few before/after examples
    examples_path = output_path.replace('.jsonl', '_examples.txt')
    with open(examples_path, 'w') as f:
        f.write("=" * 60 + "\n")
        f.write("PREPROCESSING EXAMPLES (first 3)\n")
        f.write("=" * 60 + "\n\n")
        for i, r in enumerate(results[:3], 1):
            f.write(f"--- Example {i} ---\n")
            f.write(f"ORIGINAL ({r['original_length']} chars):\n")
            f.write(r['original_text'][:500] + "...\n\n")
            f.write(f"CLEANED ({r['cleaned_length']} chars):\n")
            f.write(r['cleaned_body'][:500] + "...\n\n")
            f.write("\n")
    print(f"  Examples:  {examples_path}")


if __name__ == '__main__':
    os.chdir('/mnt/d/documents/year-2/lab/project/email-summarizer')
    process_dataset(
        input_path='data/raw/enron_sample.jsonl',
        output_path='data/processed/cleaned_emails.jsonl',
        translate_vi=False  # Enron is all English
    )
