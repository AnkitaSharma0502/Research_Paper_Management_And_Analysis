"""Convert INTERVIEW_STUDY_GUIDE.md to a self-contained printable HTML.

Run:  python docs/_build_html.py

Output: docs/INTERVIEW_STUDY_GUIDE.html — open in any browser, Ctrl+P → Save as PDF.
"""

import os
import markdown

ROOT = os.path.dirname(os.path.abspath(__file__))
SRC  = os.path.join(ROOT, "INTERVIEW_STUDY_GUIDE.md")
DST  = os.path.join(ROOT, "INTERVIEW_STUDY_GUIDE.html")

CSS = """
@page { size: A4; margin: 18mm 14mm; }
* { box-sizing: border-box; }
html, body {
  font-family: -apple-system, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
  font-size: 11pt;
  line-height: 1.55;
  color: #222;
  max-width: 920px;
  margin: 0 auto;
  padding: 24px;
  background: #fff;
}
h1, h2, h3, h4 {
  color: #1a3a5c;
  margin-top: 1.6em;
  margin-bottom: 0.5em;
  page-break-after: avoid;
}
h1 { font-size: 22pt; border-bottom: 2px solid #1a3a5c; padding-bottom: 6px; }
h2 { font-size: 17pt; border-bottom: 1px solid #ccc; padding-bottom: 4px; }
h3 { font-size: 13pt; }
h4 { font-size: 11.5pt; }
p, li { orphans: 3; widows: 3; }
code {
  font-family: "SFMono-Regular", Consolas, "Liberation Mono", Menlo, monospace;
  font-size: 0.92em;
  background: #f4f5f7;
  padding: 1px 4px;
  border-radius: 3px;
  color: #c0392b;
}
pre {
  background: #1e1e2e;
  color: #e6e6e6;
  padding: 12px 14px;
  border-radius: 6px;
  font-size: 9.5pt;
  line-height: 1.4;
  overflow-x: auto;
  page-break-inside: avoid;
}
pre code { background: transparent; color: inherit; padding: 0; }
table {
  border-collapse: collapse;
  width: 100%;
  margin: 0.6em 0 1em;
  font-size: 0.95em;
  page-break-inside: avoid;
}
th, td {
  border: 1px solid #d0d7de;
  padding: 6px 10px;
  text-align: left;
  vertical-align: top;
}
th { background: #f0f3f7; font-weight: 600; }
blockquote {
  border-left: 4px solid #6c8ebf;
  background: #f3f6fb;
  padding: 8px 14px;
  margin: 0.8em 0;
  color: #1a3a5c;
  font-style: italic;
  page-break-inside: avoid;
}
a { color: #1a5fb4; text-decoration: none; }
a:hover { text-decoration: underline; }
hr { border: 0; border-top: 1px dashed #aaa; margin: 1.5em 0; }
ul, ol { margin: 0.4em 0 0.8em 1.6em; }
li { margin: 0.2em 0; }
.toc {
  background: #f8fafc;
  border: 1px solid #d0d7de;
  border-radius: 6px;
  padding: 12px 18px;
  margin: 0 0 24px;
  page-break-after: always;
}
@media print {
  body { padding: 0; }
  pre { font-size: 8pt; }
  table { font-size: 0.88em; }
  h1 { page-break-before: always; }
  h1:first-of-type { page-break-before: auto; }
  blockquote { background: #f8f9fa; }
}
"""

HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Research Intelligence System — Study Guide</title>
<style>
{css}
</style>
</head>
<body>
{body}
<hr>
<p style="text-align:center; color:#888; font-size:9pt; margin-top: 30px;">
  Research Intelligence System — Study Guide • Generated for revision
</p>
</body>
</html>"""

def main():
    with open(SRC, "r", encoding="utf-8") as f:
        md = f.read()

    body_html = markdown.markdown(
        md,
        extensions=["extra", "tables", "fenced_code", "toc", "sane_lists"],
        output_format="html5",
    )

    html = HTML_TEMPLATE.format(css=CSS, body=body_html)

    with open(DST, "w", encoding="utf-8") as f:
        f.write(html)

    size_kb = os.path.getsize(DST) / 1024
    print(f"  Wrote {DST} ({size_kb:.1f} KB)")

if __name__ == "__main__":
    main()
