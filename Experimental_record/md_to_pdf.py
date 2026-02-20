"""Markdown to PDF converter with Japanese support using fpdf2."""
import re
from pathlib import Path
from fpdf import FPDF


class MarkdownPDF(FPDF):
    FONT_DIR = "C:/Windows/Fonts"

    def __init__(self):
        super().__init__()
        # Register Japanese fonts (Yu Gothic)
        self.add_font("YuGothic", "", f"{self.FONT_DIR}/YuGothR.ttc", uni=True)
        self.add_font("YuGothic", "B", f"{self.FONT_DIR}/YuGothB.ttc", uni=True)
        self.set_auto_page_break(auto=True, margin=20)

    def header(self):
        pass

    def footer(self):
        self.set_y(-15)
        self.set_font("YuGothic", "", 8)
        self.set_text_color(150, 150, 150)
        self.cell(0, 10, f"Page {self.page_no()}/{{nb}}", align="C")

    def add_title(self, text):
        self.set_font("YuGothic", "B", 16)
        self.set_text_color(0, 0, 0)
        self.multi_cell(0, 9, text)
        self.ln(2)

    def add_h2(self, text):
        self.ln(4)
        self.set_font("YuGothic", "B", 13)
        self.set_text_color(30, 30, 30)
        self.multi_cell(0, 8, text)
        self.set_draw_color(200, 200, 200)
        self.line(self.l_margin, self.get_y(), self.w - self.r_margin, self.get_y())
        self.ln(3)

    def add_h3(self, text):
        self.ln(3)
        self.set_font("YuGothic", "B", 11)
        self.set_text_color(50, 50, 50)
        self.multi_cell(0, 7, text)
        self.ln(2)

    def add_paragraph(self, text):
        self.set_font("YuGothic", "", 9)
        self.set_text_color(0, 0, 0)
        self.multi_cell(0, 5.5, text)
        self.ln(2)

    def add_bold_paragraph(self, text):
        self.set_font("YuGothic", "B", 9)
        self.set_text_color(0, 0, 0)
        # Handle bold markers
        text = text.replace("**", "")
        self.multi_cell(0, 5.5, text)
        self.ln(2)

    def add_list_item(self, text, indent=0):
        self.set_font("YuGothic", "", 9)
        self.set_text_color(0, 0, 0)
        x = self.l_margin + indent * 5
        self.set_x(x)
        text = text.replace("**", "")
        self.multi_cell(self.w - self.r_margin - x, 5.5, f"  \u2022 {text}")
        self.ln(1)

    def add_numbered_item(self, number, text):
        self.set_font("YuGothic", "", 9)
        self.set_text_color(0, 0, 0)
        text = text.replace("**", "")
        self.multi_cell(0, 5.5, f"  {number}. {text}")
        self.ln(1)

    def add_table(self, headers, rows):
        """Render a markdown table."""
        self.set_font("YuGothic", "", 8)

        # Calculate column widths based on content
        usable_w = self.w - self.l_margin - self.r_margin
        n_cols = len(headers)

        # Measure max content width per column
        col_widths = []
        for i in range(n_cols):
            max_len = len(headers[i])
            for row in rows:
                if i < len(row):
                    max_len = max(max_len, len(row[i]))
            col_widths.append(max_len)

        total = sum(col_widths)
        col_widths = [max(w / total * usable_w, 15) for w in col_widths]

        # Adjust to fit
        scale = usable_w / sum(col_widths)
        col_widths = [w * scale for w in col_widths]

        row_h = 6

        # Header
        self.set_font("YuGothic", "B", 8)
        self.set_fill_color(240, 240, 240)
        for i, header in enumerate(headers):
            self.cell(col_widths[i], row_h, header.strip(), border=1, fill=True, align="C")
        self.ln()

        # Rows
        self.set_font("YuGothic", "", 8)
        for row in rows:
            # Check page break
            if self.get_y() + row_h > self.h - 25:
                self.add_page()
                # Re-draw header
                self.set_font("YuGothic", "B", 8)
                self.set_fill_color(240, 240, 240)
                for i, header in enumerate(headers):
                    self.cell(col_widths[i], row_h, header.strip(), border=1, fill=True, align="C")
                self.ln()
                self.set_font("YuGothic", "", 8)

            for i in range(n_cols):
                val = row[i].strip() if i < len(row) else ""
                val = val.replace("**", "")
                self.cell(col_widths[i], row_h, val, border=1, align="C")
            self.ln()
        self.ln(3)

    def add_separator(self):
        self.ln(2)
        self.set_draw_color(180, 180, 180)
        y = self.get_y()
        self.line(self.l_margin, y, self.w - self.r_margin, y)
        self.ln(4)

    def add_meta_line(self, text):
        self.set_font("YuGothic", "", 9)
        self.set_text_color(80, 80, 80)
        text = text.replace("**", "")
        self.multi_cell(0, 5.5, text)
        self.ln(1)

    def add_image(self, img_path, md_dir=None):
        """Insert an image, fitting it to page width."""
        # Resolve relative paths against markdown file directory
        p = Path(img_path)
        if not p.is_absolute() and md_dir:
            p = Path(md_dir) / p
        if not p.exists():
            self.add_paragraph(f"[Image not found: {img_path}]")
            return
        usable_w = self.w - self.l_margin - self.r_margin
        # Check if we need a page break (estimate image height)
        if self.get_y() + 60 > self.h - 25:
            self.add_page()
        self.image(str(p), x=self.l_margin, w=usable_w)
        self.ln(5)


def parse_table(lines, start_idx):
    """Parse a markdown table starting at start_idx. Returns (headers, rows, end_idx)."""
    header_line = lines[start_idx]
    headers = [h.strip() for h in header_line.strip().strip("|").split("|")]

    # Skip separator line
    row_start = start_idx + 2
    rows = []
    idx = row_start
    while idx < len(lines) and lines[idx].strip().startswith("|"):
        cells = [c.strip() for c in lines[idx].strip().strip("|").split("|")]
        rows.append(cells)
        idx += 1

    return headers, rows, idx


def convert_md_to_pdf(md_path: str, pdf_path: str):
    """Convert markdown file to PDF."""
    with open(md_path, "r", encoding="utf-8") as f:
        content = f.read()

    md_dir = str(Path(md_path).parent)
    lines = content.split("\n")
    pdf = MarkdownPDF()
    pdf.alias_nb_pages()
    pdf.add_page()

    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        # Skip empty lines
        if not stripped:
            i += 1
            continue

        # Horizontal rule
        if stripped == "---":
            pdf.add_separator()
            i += 1
            continue

        # Image: ![alt](path)
        img_match = re.match(r"^!\[.*?\]\((.+?)\)$", stripped)
        if img_match:
            pdf.add_image(img_match.group(1), md_dir=md_dir)
            i += 1
            continue

        # H1
        if stripped.startswith("# ") and not stripped.startswith("## "):
            pdf.add_title(stripped[2:])
            i += 1
            continue

        # H2
        if stripped.startswith("## "):
            pdf.add_h2(stripped[3:])
            i += 1
            continue

        # H3
        if stripped.startswith("### "):
            pdf.add_h3(stripped[4:])
            i += 1
            continue

        # Table
        if "|" in stripped and i + 1 < len(lines) and re.match(r"^\|[\s\-:|]+\|$", lines[i + 1].strip()):
            headers, rows, end_idx = parse_table(lines, i)
            pdf.add_table(headers, rows)
            i = end_idx
            continue

        # Numbered list
        m = re.match(r"^(\d+)\.\s+(.*)", stripped)
        if m:
            pdf.add_numbered_item(m.group(1), m.group(2))
            i += 1
            continue

        # Unordered list
        if stripped.startswith("- "):
            pdf.add_list_item(stripped[2:])
            i += 1
            continue

        # Bold paragraph (starts with **)
        if stripped.startswith("**"):
            pdf.add_bold_paragraph(stripped)
            i += 1
            continue

        # Meta lines (like date, dataset info at top)
        if stripped.startswith("**") and ":" in stripped:
            pdf.add_meta_line(stripped)
            i += 1
            continue

        # Regular paragraph
        pdf.add_paragraph(stripped)
        i += 1

    pdf.output(pdf_path)
    print(f"PDF saved to: {pdf_path}")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        md_file = Path(sys.argv[1])
    else:
        md_file = Path(r"C:\Users\ykita\ROP_AI_project\Experimental_record\report_rwrop_threshold_optimization.md")
    pdf_file = md_file.with_suffix(".pdf")
    convert_md_to_pdf(str(md_file), str(pdf_file))
