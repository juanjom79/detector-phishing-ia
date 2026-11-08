import re, math, ipaddress
from collections import Counter
from urllib.parse import urlparse

PALABRAS_SOSPECHOSAS = ["login", "signin", "verify", "secure", "account", "update",
                        "bank", "password", "confirm", "webscr", "paypal", "wallet",
                        "billing", "support", "alert"]

NOMBRES = [
    "url_len", "host_len", "path_len", "query_len", "tld_len",
    "n_dots_host", "n_subdomains", "n_labels", "longest_label", "has_www",
    "n_hyphen_host", "n_hyphen_url", "n_digits", "digit_ratio", "n_letters",
    "letter_ratio", "n_equals", "n_qmark", "n_amp", "n_at", "n_slash",
    "n_percent", "n_underscore", "n_other_special", "special_ratio",
    "is_https", "is_ip", "has_port", "host_digit_ratio",
    "host_entropy", "url_entropy", "n_susp_words", "double_slash_path",
]

def _entropia(s):
    if not s:
        return 0.0
    c = Counter(s); n = len(s)
    return -sum((v / n) * math.log2(v / n) for v in c.values())

def _es_ip(h):
    try:
        ipaddress.ip_address(h.strip("[]")); return 1
    except ValueError:
        return 0

def extraer_features(url):
    """Devuelve (dict de features numéricas, tld en minúsculas) a partir de un string URL."""
    url = str(url).strip()
    if not re.match(r"^[a-zA-Z][a-zA-Z0-9+.\-]*://", url):
        url = "http://" + url
    try:
        p = urlparse(url)
        host = (p.hostname or "").lower()
        port = p.port
    except ValueError:
        p = urlparse("http://invalido"); host = ""; port = None

    labels = [l for l in host.split(".") if l]
    n_dots = host.count(".")
    is_ip = _es_ip(host)
    tld = "" if is_ip or not labels else labels[-1]

    L = max(len(url), 1)
    letters = sum(c.isalpha() for c in url)
    digits = sum(c.isdigit() for c in url)
    eq, q, amp = url.count("="), url.count("?"), url.count("&")
    especiales = sum(not c.isalnum() for c in url)
    otros = especiales - eq - q - amp
    low = url.lower()

    f = {
        "url_len": len(url), "host_len": len(host), "path_len": len(p.path),
        "query_len": len(p.query), "tld_len": len(tld),
        "n_dots_host": n_dots, "n_subdomains": 0 if is_ip else max(n_dots - 1, 0),
        "n_labels": len(labels), "longest_label": max((len(l) for l in labels), default=0),
        "has_www": int(host.startswith("www.")),
        "n_hyphen_host": host.count("-"), "n_hyphen_url": url.count("-"),
        "n_digits": digits, "digit_ratio": digits / L,
        "n_letters": letters, "letter_ratio": letters / L,
        "n_equals": eq, "n_qmark": q, "n_amp": amp, "n_at": url.count("@"),
        "n_slash": url.count("/"), "n_percent": url.count("%"),
        "n_underscore": url.count("_"), "n_other_special": otros,
        "special_ratio": especiales / L,
        "is_https": int(p.scheme.lower() == "https"), "is_ip": is_ip,
        "has_port": int(port is not None),
        "host_digit_ratio": (sum(c.isdigit() for c in host) / len(host)) if host else 0.0,
        "host_entropy": _entropia(host), "url_entropy": _entropia(url),
        "n_susp_words": sum(low.count(w) for w in PALABRAS_SOSPECHOSAS),
        "double_slash_path": int("//" in p.path),
    }
    return f, tld
