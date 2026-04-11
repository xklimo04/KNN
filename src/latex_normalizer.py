from plasTeX.TeX import TeX
import re

UNICODE_MAP = {
    # Greek lowercase
    "α": r"\alpha", "β": r"\beta", "γ": r"\gamma", "δ": r"\delta",
    "ε": r"\varepsilon", "ζ": r"\zeta", "η": r"\eta", "θ": r"\vartheta",
    "ι": r"\iota", "κ": r"\kappa", "λ": r"\lambda", "μ": r"\mu",
    "ν": r"\nu", "ξ": r"\xi", "ο": r"o", "π": r"\pi", "ρ": r"\varrho",
    "σ": r"\sigma", "τ": r"\tau", "υ": r"\upsilon", "φ": r"\varphi",
    "χ": r"\chi", "ψ": r"\psi", "ω": r"\omega",
    
    # Greek uppercase 
    "Α": r"A", "Β": r"B", "Γ": r"\Gamma", "Δ": r"\Delta",
    "Ε": r"E", "Ζ": r"Z", "Η": r"H", "Θ": r"\Theta", "Ι": r"I",
    "Κ": r"K", "Λ": r"\Lambda", "Μ": r"M", "Ν": r"N", "Ξ": r"\Xi",
    "Ο": r"O", "Π": r"\Pi", "Ρ": r"P", "Σ": r"\Sigma", "Τ": r"T",
    "Υ": r"\Upsilon", "Φ": r"\Phi", "Χ": r"X", "Ψ": r"\Psi", "Ω": r"\Omega",
    
    # Math symbols
    "∞": r"\infty", "≤": r"\le", "≥": r"\ge", "≠": r"\neq",
    "∩": r"\cap", "∪": r"\cup", "∑": r"\sum", "∏": r"\prod",
    "∫": r"\int", "∂": r"\partial", "∇": r"\nabla", "≈": r"\approx",
    "≃": r"\simeq", "≡": r"\equiv", "…": r"\dots", "±": r"\pm",
    "∓": r"\mp", "×": r"\times", "÷": r"/",
    "·": r"\cdot", "∗": r"*", "‐": r"-", "−": r"-",
    "|": r"|", "‖": r"\parallel", r"'": r"\prime", r"∋": r"\owns",
    "–": r"-", "—": r"-", "‘": r"'"
}

# Freq < 3 
UNCOMMON_COMMANDS = [r"\ae", r"\mathord", r"\bigcirc", r"\nearrow", "\\\"", r"\coprod", r"\textcircled", r"\rgroup",
                     r"\lgroup", r"\mathclose", r"\mathopen", r"\overwithdelims", r"\of", r"\triangleleft", r"\root",
                     r"\vline", r"\succ", r"\ooalign", r"\verb", r"\bigvee", r"\brack", r"\textnormal", r"\atopwithdelims",
                     r"\symbol", r"\mathrel", r"\skew", r"\null", r"\SS", r"\parbox", r"\rightarrowfill", r"\smile",
                     r"\oslash", r"\em", r"\c"]

CANONICAL_MAP = {
    r"\leq": r"\le", r"\geq": r"\ge", r"\tfrac": r"\frac", r"\dfrac": r"\frac", r"\div": r"/", r"\to" : r"\rightarrow", r"\gets": r"\leftarrow",
    r"\varepsilon": r"\epsilon", r"\varphi": r"\phi" , r"\varrho": r"\rho", r"\vartheta":r"\theta", r"\varsigma": r"\sigma",
    r"\vert": "|", r"\mid": "|", r"\ast": "*", r"\sp": r"^", r"\sb": r"_", r"\boldmath": r"\mathbf", r"\bf": r"\mathbf", r"\textbf": r"\mathbf",
    r"\it": r"\mathit", r"\textit": r"\mathit", r"\tt": r"\mathtt", r"\texttt": r"\mathtt", r"\sf": r"\mathsf", r"\textsf": r"\mathsf", r"\cal" : r"\mathcal",
    r"\sc": r"\mathrm", r"\scshape": r"\mathrm", r"\colon": r":", r"\dag": r"\dagger", r"\ddag": r"\ddagger", r"\ne": r"\neq", r"cdotp": r"cdot",
    r"\lbrack": r"[", r"\rbrack": r"]", r"\ni": r"\owns", r"\Longrightarrow": r"\implies", r"\Longleftrightarrow": r"\iff", r"\not=": r"\neq",
    r"\arrowvert": r"|", r"\longrightarrow": r"\rightarrow", r"\longleftarrow": r"\leftarrow", r"\longmapsto": r"\mapsto", r"\longleftrightarrow": r"\leftrightarrow",
    r"\hookrightarrow": r"\rightarrow", r"\Rightarrow": r"\implies", r"\Leftarrow": r"\impliedby", r"\Leftrightarrow": r"\iff", r"\land": r"\wedge",
    r"\lor": r"\vee", r"\Updownarrow": r"\updownarrow", r"\|": r"\parallel", r"\Longleftarrow": r"\impliedby", r"\bigcirc": r"\circ"
}

SPACING_COMMANDS = [r"\quad", r"\qquad", r"\,", r"\:", r"\;", r"\!", r"~", r"\*", r"\-", r"\/"]

IGNORED_COMMANDS = [
    r"\label", r"\ref", r"\cite", r"\nonumber", r"\notag", 
    r"\protect", r"\separate", r"\stepcounter", r"\setcounter",
    r"\setlength", r"\unitlength", r"\vspace", r"\hspace",
    r"\rule", r"\vrule", r"\hrule", r"\strut", r"\relax",
    r"\centerline", r"\left.", r"\right.", r"\active",
    r"\mathversion", r"\labelsep", r"\extracolsep",
    r"\Huge", r"\huge", r"\LARGE", r"\Large", 
    r"\large", r"\normalsize", r"\small", r"\tiny",
    r"\scriptsize", r"\footnotesize", r"\hfil", r"\hss", r"\vss", r"\vskip", r"\kern", r"\mkern", r"\mskip",
    r"\thicklines", r"\linethickness", r"\unitlength", r"\arraycolsep",
    r"\special", r"\leavevmode", r"\relax", r"\do", r"\crcr",
    r"\protectE", r"\protectZ", r"\protecte", r"\protectm", r"\protectu",
    r"\noalign", r"\nonumber", r"\label", r"\cite", r"\ref",
    r"\mathstrut", r"\strut", r"\vphantom", r"\hphantom", r"\phantom",
    r"\footnotemark", r"\enskip", r"\enspace", r"\thinspace", r"\medspace", r"\thickspace", r"\hfill", r"\vfill",
    r"\textstyle", r"\displaystyle", r"\scriptstyle", r"\scriptscriptstyle",
    r"\smallskip", r"\medskip", r"\bigskip", r"\space", r"\nolinebreak", r"\renewcommand"

]

BRACKETS = {
    r"\left(": "(", r"\right)": ")",
    r"\left[": "[", r"\right]": "]",
    r"\left\{": r"\{", r"\right\}": r"\}",
    r"\left|": r"|", r"\right|": r"|",
    r"\left.": "",  r"\right.": "",
    r"\right(": "(",
    r"\left)": ")",
}

MATRIX_BRACKETS_MAP = {
        'pmatrix': (r'(', r')'),
        'bmatrix': (r'[', r']'),
        'vmatrix': (r'|', r'|'),
        'matrix':  ('', ''),
        'cases':   (r'\{', '.')
}

def preprocess_matrices(latex: str) -> str:
    for env, (left, right) in MATRIX_BRACKETS_MAP.items():

        pattern = r'\\begin\{' + env + r'\}(.*?)\\end\{' + env + r'\}'
        
        def replace_with_array(match):
            content = match.group(1).strip()
            
            # Get colspec
            rows = content.split(r'\\')
            max_cols = 0
            for row in rows:
                if row.strip():
                    cols = row.count('&') + 1
                    max_cols = max(max_cols, cols)
            
            if max_cols == 0: max_cols = 1
            col_spec = 'c' * max_cols
            
            res = ""
            if left: res += f"\\left{left} "
            res += f"\\begin{{array}}{{{col_spec}}} {content} \\end{{array}}"
            if right: res += f" \\right{right}"
            return res

        latex = re.sub(pattern, replace_with_array, latex, flags=re.DOTALL)
    
    return latex

def normalize_dots(latex: str) -> str:
    latex = re.sub(r'\.\s*\.\s*\.\s*', r'\\dots ', latex)

    latex = latex.replace(r'\cdots', r'\dots')
    latex = latex.replace(r'\ldots', r'\dots')
    
    latex = re.sub(r'(\\dots\s*){2,}', r'\\dots ', latex)
    
    return latex

def clean_latex(latex: str) -> str:
    """
    Returns a latex string cleaned from spacing commands, right and left and with unified commands. 
    """
    latex = normalize_dots(latex)
    latex = preprocess_matrices(latex)
    latex = re.sub(r'\\\\\s*$', '', latex)
    latex = re.sub(r'\\vphantom\*?\s{.*?\}', '', latex)
    latex = re.sub(r'\\hphantom\*?\s{.*?\}', '', latex)
    latex = re.sub(r'\\phantom\*?\s{.*?\}', '', latex)
    latex = re.sub(r'\\hspace\*?\s*\{.*?\}', '', latex)
    latex = re.sub(r'\\vspace\*?\s*\{.*?\}', '', latex)
    latex = re.sub(r'(?<!\\)\\ ', '', latex) # leave \\
    for sp in SPACING_COMMANDS:
        latex = latex.replace(sp, '')
    for u, l in UNICODE_MAP.items():
        latex = latex.replace(u, l)
    for x, y in BRACKETS.items():
        latex = latex.replace(x, y)
    return latex.strip()

def canonicalize_token(token: str) -> str:
    """
    Returns unified version of a token, if exists, else returns token itself.
    """
    return CANONICAL_MAP.get(token, token)

def serialize_node(node):
    """
    Returns a list of tokens that represents a node.
    """
    tokens = []
    if node is None:
        return tokens

    # Text nodes
    if getattr(node, "nodeType", None) == getattr(node, "TEXT_NODE", None):
        text = getattr(node, "textContent", "").strip()
        if text:
            # Tokenize text
            tokens.extend(list(text.replace(" ", "")))
        return tokens

    original_name = getattr(node, "nodeName", "")
    name = original_name.lower()
    command_name = "\\" + original_name

    # Ignored commands
    if command_name in IGNORED_COMMANDS or f"\\{name}" in IGNORED_COMMANDS:
        return []

    # Arrays
    if name == "array":
        tokens.append(f"\\begin{{{name}}}")
        
        if name == "array" and hasattr(node, "attributes"):
            # Get columns specification
            colspec_raw = node.attributes.get("colspec", "")
            if isinstance(colspec_raw, list) or "[" in str(colspec_raw):
                colspec_parsed = ['c' for c in str(colspec_raw) if c in "lcr"]
                if colspec_parsed:
                    colspec_parsed = " ".join(colspec_parsed)
                    tokens.append("{")
                    tokens.append(colspec_parsed)
                    tokens.append("}")
            elif colspec_raw:
                content = str(colspec_raw).strip()
                inner = " ".join(list(content.replace(" ", "")))
                tokens.append("{")
                tokens.append(inner)
                tokens.append("}")

        if hasattr(node, "childNodes"):
            rows = [c for c in node.childNodes if getattr(c, "nodeName", "").lower() == "arrayrow"]
            for i, row in enumerate(rows):
                cells = [c for c in row.childNodes if getattr(c, "nodeName", "").lower() == "arraycell"]
                for j, cell in enumerate(cells):
                    tokens += serialize_node(cell)
                    # Separate cells
                    if j < len(cells) - 1:
                        tokens.append("&")
                # Separate rows
                if i < len(rows) - 1:
                    tokens.append(r"\\")
        tokens.append(f"\\end{{{name}}}")
        return tokens

    # Active
    if name == "active":
        char = getattr(node, "attributes", {}).get("char")
        tokens.append(char)
        arg_node = node.attributes.get("self") if hasattr(node, "attributes") else None
        if arg_node:
            tokens.append("{")
            tokens += serialize_node(arg_node)
            tokens.append("}")
        return tokens

    # Fractions
    if name in ["frac", "tfrac", "dfrac"]:
        tokens.append("\\frac")
        for arg in ["numer", "denom"]:
            val = node.attributes.get(arg)
            tokens.append("{")
            tokens += serialize_node(val)
            tokens.append("}")
        return tokens

    # Styles
    if name in ["cal", "bf", "it", "mathbf", "mathcal", "mathrm", "mathit", "mathsf", "mathtt", "boldmath", "textbf", "textit", "tt", "mathtt", "texttt", "sf", "mathsf", "textsf", "sc", "mathrm", "scshape"]: 
        canon_style = canonicalize_token(command_name)
        if canon_style:
            tokens.append(canon_style)
            
            inner_tokens = []
            for c in node.childNodes:
                inner_tokens += serialize_node(c)
            
            if inner_tokens and inner_tokens[0] == "{" and inner_tokens[-1] == "}":
                tokens += inner_tokens
            else:
                tokens.append("{")
                tokens += inner_tokens
                tokens.append("}")
        return tokens
    
    # Groups
    if name in ["bgroup", "group"]:
        tokens.append("{")
        for c in node.childNodes:
            tokens += serialize_node(c)
        tokens.append("}")
        return tokens
    
    # Not
    if name == "not":
        tokens.append(f"\\{name}")
        arg_symbol = node.attributes.get("symbol")
        tokens.append("{")
        if arg_symbol:
            tokens += serialize_node(arg_symbol)
        tokens.append("}")
        return tokens
    
    # Stackrel
    if name == "stackrel":
        tokens.append(f"\\{name}")
        for arg_name in ["top", "bottom"]:
            arg_node = node.attributes.get(arg_name)
            tokens.append("{")
            if arg_node:
                tokens += serialize_node(arg_node)
            tokens.append("}")
        return tokens
    
    # Size commands
    if name in ["big", "Big", "bigg", "Bigg", "bigl", "Bigl", "biggl", "Biggl", "bigr", "Bigr", "biggr", "Biggr", "Biggm", "Bigm", "biggm", "bigm"]:
        arg_char = node.attributes.get("char") if hasattr(node, "attributes") and node.attributes else None
        
        if arg_char:
            attrs = getattr(arg_char, "attributes", None)
            if attrs is not None and "modifier" in attrs:
                tokens.append(attrs["modifier"])
            else:
                tokens += serialize_node(arg_char)
        return tokens

    # Brackets
    if name in ["left", "right"]:
        arg_char = node.attributes.get("char") if hasattr(node, "attributes") and node.attributes else None
        if arg_char:
            tokens += serialize_node(arg_char)
        
        elif hasattr(node, "childNodes"):
            for c in node.childNodes:
                tokens += serialize_node(c)
                
        return tokens
    
    # Raise / Raisebox
    if name in ["raise", "raisebox"]:
        if hasattr(node, "childNodes"):
            for c in node.childNodes:
                tokens += serialize_node(c)
        return tokens
    
    if command_name in UNCOMMON_COMMANDS or f"\\{name}" in UNCOMMON_COMMANDS:
        print("found uncommon")
        raise Exception
    
    # Generic command fallback
    if name and not name.startswith('#') and name not in ["math", "dom-document", "arrayrow", "arraycell", "par"]:
        command_name = "\\" + original_name
        tokens.append(canonicalize_token(command_name))
        
        arg_self = getattr(node, "attributes", {}).get("self")
        if arg_self:
            tokens.append("{")
            tokens += serialize_node(arg_self)
            tokens.append("}")
        elif hasattr(node, "childNodes") and len(node.childNodes) > 0:
            has_content = any(getattr(c, "nodeType", 0) != 3 or c.textContent.strip() for c in node.childNodes)
            if has_content:
                is_lr = name in ["left", "right"]
                if not is_lr: tokens.append("{")
                for c in node.childNodes:
                    tokens += serialize_node(c)
                if not is_lr: tokens.append("}")
        return tokens

    if hasattr(node, "childNodes"):
        for c in node.childNodes:
            tokens += serialize_node(c)

    return tokens

def normalize_latex_tokens(latex: str) -> str:
    """
    Returns normalized version of latex.
    """
    latex = clean_latex(latex)
    tex = TeX()
    tex.input(f"${latex}$")
    doc = tex.parse()
    all_tokens = []
    for n in doc.childNodes:
        all_tokens += serialize_node(n)
    latex = re.sub(r'\\active::', '', " ".join(all_tokens))
    latex = latex.replace("  ", " ")
    return latex