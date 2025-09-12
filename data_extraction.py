import requests, pathlib, sys
from xml.etree import ElementTree as ET

PMCIDS = ["PMC10952103", "PMC11906441", "PMC11150048", "PMC11512351", "PMC12197743","PMC11768232", "PMC11974452",
          "PMC10468816","PMC10796032","PMC10556267","PMC12135148","PMC10186953","PMC10303212","PMC11810083",
          "PMC9863601","PMC11703581","PMC10057056","PMC10376530","PMC11779990","PMC12310597","PMC10052442",
          "PMC11647977","PMC10154003","PMC11680248","PMC10060859"]


def fetch_xml(pmcid: str) -> str:
    url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
    params = {
        "db": "pmc",
        "id": pmcid,
        "rettype": "full",
        "retmode": "xml",
    }
    r = requests.get(url, params=params, timeout=60)
    r.raise_for_status()
    return r.text


def pretty_print_xml(xml_text: str) -> str:
    # Pretty-print with stdlib (fast + no extra deps)
    # Remove DOCTYPE first (ElementTree can choke on it)
    import re
    xml_text = re.sub(r"<!DOCTYPE[^>]*>", "", xml_text, flags=re.IGNORECASE)
    root = ET.fromstring(xml_text)
    # ET.tostring doesn't pretty print; add newlines between tags as a simple legibility hack
    rough = ET.tostring(root, encoding="unicode")
    # insert a newline between close><open to avoid mega-lines
    pretty = rough.replace("><", ">\n<")
    return pretty


def extract_plain_text(xml_text: str) -> str:
    import re
    xml_text = re.sub(r"<!DOCTYPE[^>]*>", "", xml_text, flags=re.IGNORECASE)
    root = ET.fromstring(xml_text)

    # strip namespaces so tags are just 'sec', 'p', etc.
    for el in root.iter():
        if "}" in el.tag:
            el.tag = el.tag.split("}", 1)[1]

    def text_of(node):
        parts = []
        if node.text:
            parts.append(node.text)
        for c in list(node):
            parts.append(text_of(c))
            if c.tail:
                parts.append(c.tail)
        return "".join(parts)

    out = []

    # Title
    t = root.find(".//front/article-meta/title-group/article-title")
    if t is not None:
        title = text_of(t).strip()
        if title:
            out.append(title)

    # Abstract(s)
    for ab in root.findall(".//abstract"):
        head = ab.find("./title")
        head_txt = text_of(head).strip() if head is not None else "Abstract"
        paras = []
        for child in list(ab):
            if child.tag == "title":
                continue
            if child.tag == "p":
                txt = text_of(child).strip()
                if txt:
                    paras.append(txt)
            elif child.tag == "sec":
                # structured abstract: title + paragraphs
                sec_title = child.find("./title")
                sec_head = text_of(sec_title).strip() if sec_title is not None else ""
                sec_paras = [text_of(p).strip() for p in child.findall("./p")]
                sec_block = (sec_head + "\n\n" if sec_head else "") + "\n\n".join([p for p in sec_paras if p])
                if sec_block.strip():
                    paras.append(sec_block)
        if paras:
            out.append(head_txt + "\n\n" + "\n\n".join(paras))

    # Body sections
    body = root.find(".//body")
    if body is not None:
        def walk_sec(sec):
            title_el = sec.find("./title")
            if title_el is not None:
                head = text_of(title_el).strip()
                if head:
                    out.append(head.upper())

            # paragraphs directly under this sec
            for p in sec.findall("./p"):
                txt = text_of(p).strip()
                if txt:
                    out.append(txt)

            # lists
            for lst in sec.findall("./list"):
                for li in lst.findall("./list-item"):
                    li_txt = text_of(li).strip()
                    if li_txt:
                        out.append("• " + li_txt)

            # nested sections
            for sub in sec.findall("./sec"):
                walk_sec(sub)

        for top in body.findall("./sec"):
            walk_sec(top)

    # Join with blank lines between blocks
    text = "\n\n".join(out)

    # Normalize whitespace a bit
    text = text.replace("\r", "")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r" +\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)

    # === Remove all bracketed/parenthesized content ===
    text = re.sub(r"\[[^\]]*\]", "", text)   # remove [stuff]
    text = re.sub(r"\([^)]*\)", "", text)    # remove (stuff)

    # === Remove standalone numbers followed by commas ===
    text = re.sub(r"\b\d+,\s*", "", text)    # remove numbers like 50, 62,

    # Debug: Check text after cleaning
    print("After cleaning:", text[:500])

    # === Clean up punctuation/spaces left behind by removals ===
    text = re.sub(r"\s{2,}", " ", text)
    text = re.sub(r"\s+([.,;:!?])", r"\1", text)
    text = re.sub(r"\s+\.", ".", text)
    text = re.sub(r"(?m)^\s*•\s*$\n?", "", text)
    text = re.sub(r"(?m)^[ \t]+", "", text)
    text = re.sub(r"(?m)[ \t]+$", "", text)
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip().lower()

def main():
    pathlib.Path("data").mkdir(exist_ok=True)

    for pmcid in PMCIDS:
        try:
            print(f"Fetching {pmcid}…")
            xml_text = fetch_xml(pmcid)

            # Save compact XML
            raw_xml_path = pathlib.Path("data") / f"{pmcid}.xml"
            raw_xml_path.write_text(xml_text, encoding="utf-8")

            # Save pretty XML
            pretty = pretty_print_xml(xml_text)
            pretty_path = pathlib.Path("data") / f"{pmcid}.pretty.xml"
            pretty_path.write_text(pretty, encoding="utf-8")

            # Save extracted plain text
            plain = extract_plain_text(xml_text)
            txt_path = pathlib.Path("data") / f"{pmcid}.plain.txt"
            txt_path.write_text(plain, encoding="utf-8")

            print(f"  Saved {pmcid} → plain {txt_path}")
        except Exception as e:
            print(f"  ERROR {pmcid}: {e}")

    print(
        f"Saved:\n  - raw XML   → {raw_xml_path}\n  - pretty XML→ {pretty_path}\n  - plain text→ {txt_path}\n"
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        sys.stderr.write(f"ERROR: {e}\n")
        sys.exit(1)
