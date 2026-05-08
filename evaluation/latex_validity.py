import re

def begin_end_ok(text):
    begins = re.findall(r'\\begin\{([^}]*)\}', text)
    ends = re.findall(r'\\end\{([^}]*)\}', text)

    return begins == ends

def balanced_curly(text):
    count = 0

    for c in text:
        if c == "{":
            count += 1
        elif c == "}":
            count -= 1

        if count < 0:
            return False

    return count == 0

import re

def is_latex_valid(text):

    # matching curly braces
    if not balanced_curly(text):
        return 0

    # begin/end
    if not begin_end_ok(text):
        return 0

    # incomplete \frac
    frac_total = text.count(r"\frac")
    frac_valid = len(
        re.findall(r'\\frac\s*\{.*?\}\s*\{.*?\}', text)
    )

    if frac_total != frac_valid:
        return 0

    return 1