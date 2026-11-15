"""Text recognition of the finished page, with layout and simple text styles.

Tesseract finds the words with their positions, grouped into blocks, paragraphs
and lines. On top of that this module works out:

- layout: blank lines from the vertical gaps between lines, indentation from
  the left margin, and tabs for wide gaps inside a line (columns, signatures);
- bold words, from the thickness of their strokes compared to the page's usual
  text (Tesseract's LSTM engine doesn't report font styles);
- larger text such as headings, from line height;
- uncertain words (handwriting, smudges), from Tesseract's confidence.
"""

import csv
import functools
import io
import locale
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field

import cv2
import numpy as np

PREFERRED_LANGUAGES = ('deu', 'eng')   # used when installed, after the system's language
# Tesseract's codes for the languages of common locales, with their names
LANGUAGES = {
    'cs': ('ces', 'Czech'), 'da': ('dan', 'Danish'), 'de': ('deu', 'German'), 'el': ('ell', 'Greek'),
    'en': ('eng', 'English'), 'es': ('spa', 'Spanish'), 'fi': ('fin', 'Finnish'), 'fr': ('fra', 'French'),
    'hu': ('hun', 'Hungarian'), 'it': ('ita', 'Italian'), 'ja': ('jpn', 'Japanese'), 'ko': ('kor', 'Korean'),
    'nb': ('nor', 'Norwegian'), 'nl': ('nld', 'Dutch'), 'pl': ('pol', 'Polish'), 'pt': ('por', 'Portuguese'),
    'ro': ('ron', 'Romanian'), 'ru': ('rus', 'Russian'), 'sv': ('swe', 'Swedish'), 'tr': ('tur', 'Turkish'),
    'uk': ('ukr', 'Ukrainian'), 'zh': ('chi_sim', 'Chinese (simplified)'),
}
NOT_LANGUAGES = ('osd', 'equ')         # Tesseract data for orientation and equations
BOLD_RATIO = 1.25             # stroke thickness vs the page's typical text for bold
LARGE_RATIO = 1.3             # line height vs the typical line for larger text
UNCERTAIN = 60                # confidence (0..100) below which a word counts as uncertain
JUNK_LETTERS = 4              # all-uncertain lines with fewer letters/digits are dropped
JUNK_CONFIDENCE = 30          # lines with lower average confidence are dropped (scribbles, drawings)
SPECK_CONFIDENCE = 80         # isolated 1-2 character words below this confidence are specks
TAB_GAP = 3.0                 # gaps wider than this many typical word gaps become a tab
BLANK_LINE_GAP = 1.35         # line spacing beyond this many typical spacings leaves a blank line
INDENT_STEP = 0.5             # indentation in units of the typical character width ...
MAX_INDENT = 40               # ... capped at this many spaces


@dataclass
class Word:
    text: str
    box: tuple                # left, top, width, height in page pixels
    confidence: float
    bold: bool = False


@dataclass
class Line:
    words: list
    indent: int = 0           # spaces before the line
    large: bool = False
    blank_before: bool = False
    gaps: list = field(default_factory=list)   # separator before each word after the first: ' ' or '\t'


def available():
    return shutil.which('tesseract') is not None


@functools.cache
def installed_languages():
    """Tesseract's codes of the installed languages, sorted; empty if Tesseract is missing.
    Asked once: languages installed later show after a restart."""
    if not available():
        return ()
    try:
        out = subprocess.run(['tesseract', '--list-langs'], capture_output=True, text=True, timeout=10).stdout
    except (OSError, subprocess.SubprocessError):
        return ()
    # the first line is a heading; script data ("script/Latin") isn't offered
    return tuple(sorted(code for code in out.split('\n')[1:]
                        if code and '/' not in code and code not in NOT_LANGUAGES))


def language_name(code):
    return next((name for c, name in LANGUAGES.values() if c == code), code)


def languages(chosen=None):
    """The languages to recognise, e.g. 'deu+eng', or None if none is installed.

    `chosen`: the user's choice (codes); without one, the system's language and
    PREFERRED_LANGUAGES, as far as installed."""
    installed = installed_languages()
    use = [code for code in chosen or () if code in installed]
    if not use:
        # the system's language first, unless it is one of the preferred ones anyway
        system = LANGUAGES.get((locale.getlocale()[0] or '')[:2].lower(), (None,))[0]
        first = () if system in PREFERRED_LANGUAGES else (system,)
        use = [code for code in first + PREFERRED_LANGUAGES if code in installed]
    return '+'.join(use) if use else None


def recognise(gray, langs=None):
    """Lines of the page (uint8 greyscale, upright), in reading order."""
    langs = langs or languages()
    if langs is None:
        return []
    with tempfile.NamedTemporaryFile(suffix='.png') as f:
        cv2.imwrite(f.name, gray)
        try:
            out = subprocess.run(['tesseract', f.name, 'stdout', '-l', langs, '--psm', '3', 'tsv'],
                                 capture_output=True, text=True, timeout=300).stdout
        except (OSError, subprocess.SubprocessError):
            return []                           # Tesseract failed or took too long
    rows = list(csv.DictReader(io.StringIO(out), delimiter='\t', quoting=csv.QUOTE_NONE))
    lines, current, key = [], None, None
    for r in rows:
        if r.get('level') != '5' or not (r.get('text') or '').strip():
            continue
        k = (r['block_num'], r['par_num'], r['line_num'])
        if k != key:
            current, key = Line(words=[]), k
            lines.append(current)
        box = tuple(int(r[name]) for name in ('left', 'top', 'width', 'height'))
        current.words.append(Word(r['text'], box, float(r['conf'])))
    _drop_specks(lines)
    lines = [line for line in lines if line.words and not _junk(line)]
    _styles(gray, lines)
    _layout(lines)
    return lines


def _drop_specks(lines):
    """Remove specks on the paper that Tesseract read as letters or punctuation.

    A line is split into clusters at wide gaps; a cluster of only 1-2 character
    pieces goes if it is all punctuation (real punctuation touches its word) or
    uncertain.
    """
    gaps = [b.box[0] - (a.box[0] + a.box[2]) for line in lines for a, b in zip(line.words, line.words[1:])]
    if not gaps:
        return
    far = TAB_GAP * float(np.median(gaps))
    for line in lines:
        clusters = [[line.words[0]]] if line.words else []
        for a, b in zip(line.words, line.words[1:]):
            if b.box[0] - (a.box[0] + a.box[2]) > far:
                clusters.append([])
            clusters[-1].append(b)
        keep = []
        for cluster in clusters:
            short = all(len(w.text) <= 2 for w in cluster)
            punctuation = not any(ch.isalnum() for w in cluster for ch in w.text)
            uncertain = np.mean([w.confidence for w in cluster]) < SPECK_CONFIDENCE
            if not (short and (punctuation or uncertain)):
                keep += cluster
        line.words = keep


def _junk(line):
    """Lines made only of uncertain bits with hardly any letters: specks, scribbles."""
    letters = sum(ch.isalnum() for w in line.words for ch in w.text)
    if letters == 0 or np.mean([w.confidence for w in line.words]) < JUNK_CONFIDENCE:
        return True
    return all(w.confidence < UNCERTAIN for w in line.words) and letters < JUNK_LETTERS


def _stroke_width(ink):
    """Typical stroke thickness of the ink in a word box: 2 x area / outline length."""
    area = int(ink.sum())
    if area < 20:
        return None
    outline = area - int(cv2.erode(ink, np.ones((3, 3), np.uint8)).sum())
    return 2.0 * area / max(outline, 1)


def _styles(gray, lines):
    ink = (gray < 128).astype(np.uint8)
    widths = {}
    for line in lines:
        for w in line.words:
            x, y, bw, bh = w.box
            sw = _stroke_width(ink[y:y + bh, x:x + bw])
            if sw is not None and w.confidence >= UNCERTAIN:
                widths[id(w)] = sw
    if widths:
        typical = float(np.median(list(widths.values())))
        for line in lines:
            for w in line.words:
                w.bold = widths.get(id(w), 0) >= BOLD_RATIO * typical
    heights = [np.median([w.box[3] for w in line.words]) for line in lines if line.words]
    if heights:
        typical = float(np.median(heights))
        for line, h in zip(lines, heights):
            line.large = h >= LARGE_RATIO * typical


def _layout(lines):
    if not lines:
        return
    lefts = [line.words[0].box[0] for line in lines]
    margin = float(np.percentile(lefts, 10))
    char_widths = [w.box[2] / len(w.text) for line in lines for w in line.words if len(w.text) >= 3]
    char = float(np.median(char_widths)) if char_widths else 10.0
    gaps = [b.box[0] - (a.box[0] + a.box[2]) for line in lines for a, b in zip(line.words, line.words[1:])]
    space = float(np.median(gaps)) if gaps else char
    tops = [line.words[0].box[1] for line in lines]
    pitches = [b - a for a, b in zip(tops, tops[1:]) if b > a]
    # the line spacing inside paragraphs: the tight end of all spacings (a page can
    # have more widely spaced lines, like an address block, than paragraph lines)
    pitch = float(np.percentile(pitches, 25)) if pitches else 0
    for i, line in enumerate(lines):
        line.indent = int(min(MAX_INDENT, max(0, round((lefts[i] - margin) / char * INDENT_STEP))))
        line.gaps = ['\t' if b.box[0] - (a.box[0] + a.box[2]) > TAB_GAP * space else ' '
                     for a, b in zip(line.words, line.words[1:])]
        if i and pitch and tops[i] - tops[i - 1] > BLANK_LINE_GAP * pitch:
            line.blank_before = True


def searchable_pdf(gray, path, dpi, langs=None):
    """Write the page as a PDF with an invisible text layer (searchable, copyable).
    Returns False if Tesseract isn't available or fails."""
    langs = langs or languages()
    if langs is None:
        return False
    with tempfile.TemporaryDirectory() as tmp:
        image = f'{tmp}/page.png'
        cv2.imwrite(image, gray)
        try:
            subprocess.run(['tesseract', image, f'{tmp}/page', '-l', langs, '--dpi', str(dpi), 'pdf'],
                           capture_output=True, timeout=300, check=True)
            shutil.move(f'{tmp}/page.pdf', path)
        except (OSError, subprocess.SubprocessError):
            return False
    return True


def plain_text(lines):
    """The recognised text with its layout, for copying."""
    out = []
    for line in lines:
        if line.blank_before:
            out.append('')
        text = line.words[0].text + ''.join(g + w.text for g, w in zip(line.gaps, line.words[1:]))
        out.append(' ' * line.indent + text)
    return '\n'.join(out) + '\n'
