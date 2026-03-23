from plasTeX.TeX import TeX
import re

UNICODE_MAP = {
    # Greek lowercase
    "α": r"\alpha","β": r"\beta","γ": r"\gamma","δ": r"\delta",
    "ε": r"\epsilon","ζ": r"\zeta","η": r"\eta","θ": r"\theta",
    "ι": r"\iota","κ": r"\kappa","λ": r"\lambda","μ": r"\mu",
    "ν": r"\nu","ξ": r"\xi","ο": r"o","π": r"\pi","ρ": r"\rho",
    "σ": r"\sigma","τ": r"\tau","υ": r"\upsilon","φ": r"\phi",
    "χ": r"\chi","ψ": r"\psi","ω": r"\omega",
    # Greek uppercase
    "Α": r"A","Β": r"B","Γ": r"\Gamma","Δ": r"\Delta",
    "Ε": r"E","Ζ": r"Z","Η": r"H","Θ": r"\Theta","Ι": r"I",
    "Κ": r"K","Λ": r"\Lambda","Μ": r"M","Ν": r"N","Ξ": r"\Xi",
    "Ο": r"O","Π": r"\Pi","Ρ": r"P","Σ": r"\Sigma","Τ": r"T",
    "Υ": r"\Upsilon","Φ": r"\Phi","Χ": r"X","Ψ": r"\Psi","Ω": r"\Omega",
    # Math symbols
    "∞": r"\infty","≤": r"\le","≥": r"\ge","≠": r"\neq",
    "∩": r"\cap","∪": r"\cup","∑": r"\sum","∏": r"\prod",
    "∫": r"\int","∂": r"\partial","∇": r"\nabla","≈": r"\approx",
    "≃": r"\simeq","≡": r"\equiv","…": r"\dots","±": r"\pm",
    "∓": r"\mp","×": r"\times","÷": r"\div"
}

CANONICAL_MAP = {
    r"\leq": r"\le", r"\geq": r"\ge", r"\tfrac": r"\frac", r"\dfrac": r"\frac", r"/": r"\div", r"\to" : r"\rightarrow"
}

SPACING_COMMANDS = [r"\quad", r"\qquad", r"\,", r"\:", r"\;", r"\!", r"\enspace", r"\thinspace", r"\medspace", r"\thickspace"]

BRACKETS = {
    r"\left(": "(", r"\right)": ")",
    r"\left[": "[", r"\right]": "]",
    r"\left\{": "{", r"\right\}": "}",
    r"\left|": "|", r"\right|": "|",
    r"\left.": "",  r"\right.": "",
    r"\right(": "(",
    r"\left)": ")",
}

def clean_latex(latex: str) -> str:
    """
    Returns a latex string cleaned from spacing commands, \right and \left and with unified commands. 
    """
    latex = re.sub(r'\\hspace\*?\s*\{[^{}]*\}', '', latex)
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

    name = getattr(node, "nodeName", "").lower()

    # Arrays and matrices
    if name in ["array", "matrix", "pmatrix", "bmatrix", "vmatrix"]:
        tokens.append(f"\\begin{{{name}}}")
        
        if name == "array" and hasattr(node, "attributes"):
            # Get columns specification
            colspec_raw = node.attributes.get("colspec", "")
            if isinstance(colspec_raw, list) or "[" in str(colspec_raw):
                colspec_parsed = [c for c in str(colspec_raw) if c in "lcr"]
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
    if name in ["cal", "bf", "it", "mathbf", "mathcal"]:
        tokens.append(f"\\{name}")
        tokens.append("{")
        for c in node.childNodes:
            tokens += serialize_node(c)
        tokens.append("}")
        return tokens
    
    # Groups
    if name in ["bgroup", "group"]:
        tokens.append("{")
        for c in node.childNodes:
            tokens += serialize_node(c)
        tokens.append("}")
        return tokens

    # Generic command fallback
    if name and not name.startswith('#') and name not in ["math", "dom-document", "arrayrow", "arraycell", "par"]:
        command_name = "\\" + name
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