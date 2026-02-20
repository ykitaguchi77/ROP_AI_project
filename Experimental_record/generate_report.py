# -*- coding: utf-8 -*-
from fpdf import FPDF
import os

class PDF(FPDF):
    def __init__(self):
        super().__init__()
        # Add Japanese font (Meiryo)
        font_path = "C:/Windows/Fonts/meiryo.ttc"
        if os.path.exists(font_path):
            self.add_font("Meiryo", "", font_path, uni=True)
            self.add_font("Meiryo", "B", "C:/Windows/Fonts/meiryob.ttc", uni=True)
            self.jp_font = "Meiryo"
        else:
            font_path = "C:/Windows/Fonts/YuGothM.ttc"
            if os.path.exists(font_path):
                self.add_font("YuGothic", "", font_path, uni=True)
                self.jp_font = "YuGothic"
            else:
                self.jp_font = "Helvetica"

    def header(self):
        self.set_font(self.jp_font, "B", 10)
        self.set_text_color(100, 100, 100)
        self.cell(0, 8, "ROP AI Project - Experimental Report", 0, 1, "R")
        self.line(10, 18, 200, 18)
        self.ln(5)

    def footer(self):
        self.set_y(-15)
        self.set_font(self.jp_font, "", 8)
        self.set_text_color(128, 128, 128)
        self.cell(0, 10, f"Page {self.page_no()}", 0, 0, "C")

    def section_title(self, title):
        self.set_font(self.jp_font, "B", 14)
        self.set_text_color(30, 80, 150)
        self.set_fill_color(240, 245, 255)
        self.cell(0, 10, title, 0, 1, "L", fill=True)
        self.ln(3)

    def subsection_title(self, title):
        self.set_font(self.jp_font, "B", 11)
        self.set_text_color(50, 50, 50)
        self.cell(0, 8, title, 0, 1, "L")
        self.ln(1)

    def body_text(self, text):
        self.set_font(self.jp_font, "", 10)
        self.set_text_color(0, 0, 0)
        self.multi_cell(0, 6, text)
        self.ln(2)

    def add_table(self, headers, data, col_widths=None):
        self.set_font(self.jp_font, "B", 9)
        self.set_fill_color(70, 130, 180)
        self.set_text_color(255, 255, 255)

        if col_widths is None:
            col_widths = [190 / len(headers)] * len(headers)

        for i, header in enumerate(headers):
            self.cell(col_widths[i], 7, header, 1, 0, "C", fill=True)
        self.ln()

        self.set_font(self.jp_font, "", 9)
        self.set_text_color(0, 0, 0)
        fill = False
        for row in data:
            self.set_fill_color(245, 250, 255) if fill else self.set_fill_color(255, 255, 255)
            for i, cell in enumerate(row):
                self.cell(col_widths[i], 6, str(cell), 1, 0, "C", fill=True)
            self.ln()
            fill = not fill
        self.ln(3)

    def highlight_box(self, text, color="blue"):
        if color == "blue":
            self.set_fill_color(230, 240, 255)
            self.set_text_color(30, 60, 120)
        elif color == "green":
            self.set_fill_color(230, 255, 230)
            self.set_text_color(30, 100, 30)
        elif color == "orange":
            self.set_fill_color(255, 245, 230)
            self.set_text_color(150, 80, 0)

        self.set_font(self.jp_font, "", 9)
        self.multi_cell(0, 6, text, border=1, fill=True)
        self.set_text_color(0, 0, 0)
        self.ln(2)

# Create PDF
pdf = PDF()
pdf.set_auto_page_break(auto=True, margin=15)

# ===== Page 1: Title =====
pdf.add_page()
pdf.ln(30)
pdf.set_font(pdf.jp_font, "B", 24)
pdf.set_text_color(30, 60, 120)
pdf.cell(0, 15, "実験記録レポート", 0, 1, "C")
pdf.set_font(pdf.jp_font, "", 16)
pdf.set_text_color(80, 80, 80)
pdf.cell(0, 10, "2026年1月28日", 0, 1, "C")
pdf.ln(20)

pdf.set_font(pdf.jp_font, "", 12)
pdf.set_text_color(0, 0, 0)
contents = [
    "1. ROP分類器 画質サブセット比較 結果分析",
    "2. 症例レベルアンサンブルによるメトリクス改善検討"
]
pdf.cell(0, 10, "【目次】", 0, 1, "C")
for c in contents:
    pdf.cell(0, 8, c, 0, 1, "C")

pdf.ln(30)
pdf.set_font(pdf.jp_font, "", 10)
pdf.set_text_color(100, 100, 100)
pdf.cell(0, 8, "ROP AI Project - Multicenter Study", 0, 1, "C")

# ===== Page 2: Section 1 =====
pdf.add_page()
pdf.section_title("1. ROP分類器 画質サブセット比較")

pdf.subsection_title("概要")
pdf.body_text("EfficientNet-B0ベースのマルチタスクROP分類器を、異なる画質サブセットで学習・評価。\n- good_only: Good画像のみ (n=3,453)\n- good_fair: Good + Fair画像 (n=4,978)\n5-fold交差検証、患者レベル層化分割を使用。")

pdf.subsection_title("Zone分類 結果")
pdf.add_table(
    ["指標", "good_only", "good_fair", "差分"],
    [
        ["Accuracy", "0.7118", "0.7137", "+0.002"],
        ["QW Kappa", "0.4530", "0.5024", "+0.049"],
        ["Macro F1", "0.6216", "0.6323", "+0.011"],
    ],
    [40, 50, 50, 50]
)

pdf.subsection_title("Zone クラス別性能 (good_fair)")
pdf.add_table(
    ["Class", "n", "Sensitivity", "Specificity", "PPV", "F1"],
    [
        ["Zone I", "651", "0.458", "0.980", "0.776", "0.576"],
        ["Zone II", "3,127", "0.860", "0.485", "0.738", "0.794"],
        ["Zone III", "1,200", "0.473", "0.898", "0.595", "0.527"],
    ],
    [30, 25, 35, 35, 30, 30]
)

pdf.subsection_title("Stage分類 結果")
pdf.add_table(
    ["指標", "good_only", "good_fair", "差分"],
    [
        ["Accuracy", "0.6432", "0.6531", "+0.010"],
        ["QW Kappa", "0.6497", "0.6674", "+0.018"],
        ["Macro F1", "0.6347", "0.6386", "+0.004"],
    ],
    [40, 50, 50, 50]
)

pdf.subsection_title("Stage クラス別性能 (good_fair)")
pdf.add_table(
    ["Class", "n", "Sensitivity", "Specificity", "PPV", "F1"],
    [
        ["Stage 0", "1,969", "0.749", "0.743", "0.657", "0.700"],
        ["Stage 1", "1,355", "0.555", "0.873", "0.622", "0.586"],
        ["Stage 2", "823", "0.447", "0.944", "0.614", "0.518"],
        ["Stage 3", "809", "0.795", "0.937", "0.711", "0.751"],
    ],
    [30, 25, 35, 35, 30, 30]
)

# ===== Page 3: Binary tasks =====
pdf.add_page()
pdf.subsection_title("Aggressive ROP 検出")
pdf.add_table(
    ["指標", "good_only", "good_fair", "差分"],
    [
        ["Accuracy", "0.9615", "0.9681", "+0.007"],
        ["QW Kappa", "0.3698", "0.5465", "+0.177"],
        ["AUC-ROC", "0.6386", "0.7202", "+0.082"],
        ["Sensitivity", "0.286", "0.447", "+0.161"],
    ],
    [40, 50, 50, 50]
)

pdf.highlight_box("Fair画像追加でAggressive ROPの検出力が大幅改善 (Kappa +0.18, AUC +0.08, Sens +0.16)", "green")

pdf.subsection_title("Treatment 予測")
pdf.add_table(
    ["指標", "good_only", "good_fair", "差分"],
    [
        ["Accuracy", "0.9401", "0.9413", "+0.001"],
        ["QW Kappa", "0.7052", "0.7050", "-0.000"],
        ["AUC-ROC", "0.8350", "0.8184", "-0.017"],
        ["Sensitivity", "0.696", "0.654", "-0.042"],
    ],
    [40, 50, 50, 50]
)

pdf.highlight_box("Treatmentはほぼ変化なし（やや低下傾向）", "orange")

pdf.subsection_title("考察")
pdf.body_text("・Fair画像追加でAggressive ROPの検出力が大幅改善\n・Zone, StageはFair追加で微小な改善\n・Zone I/III, Stage 2, Aggressive ROP(Yes)の感度が0.45前後と低い点が課題")

# ===== Page 4: Section 2 - Ensemble =====
pdf.add_page()
pdf.section_title("2. 症例レベルアンサンブル検討")

pdf.subsection_title("概要")
pdf.body_text("同一症例の全Good画像を多数決アンサンブルした場合の症例レベルメトリクスをシミュレーション。\n・症例あたり画像数: good_only 平均14.7枚、good_fair 平均19.2枚\n・アンサンブル方法: 多数決 (majority vote)")

pdf.subsection_title("Multi-classタスク: 改善")
pdf.add_table(
    ["Task", "Metric", "good_only (Img->Case)", "good_fair (Img->Case)"],
    [
        ["Zone", "Accuracy", "0.711->0.740 (+2.9%)", "0.714->0.753 (+3.9%)"],
        ["Zone", "Kappa", "0.453->0.458 (+0.5%)", "0.502->0.537 (+3.5%)"],
        ["Stage", "Accuracy", "0.643->0.654 (+1.1%)", "0.653->0.667 (+1.4%)"],
        ["Stage", "Kappa", "0.650->0.687 (+3.7%)", "0.667->0.719 (+5.2%)"],
    ],
    [30, 30, 65, 65]
)
pdf.highlight_box("多数決アンサンブルでZone/Stageとも改善。特にKappaの改善が顕著 (Stage +3.7~5.2%)", "green")

pdf.subsection_title("Binaryタスク: 単純多数決では悪化")
pdf.add_table(
    ["Task", "Condition", "Image->Case Sens", "原因"],
    [
        ["Aggressive ROP", "good_only", "0.286->0.111 (-17.5%)", "陽性率中央値5.6%"],
        ["Aggressive ROP", "good_fair", "0.447->0.444 (-0.3%)", "陽性率中央値40%"],
        ["Treatment", "good_only", "0.696->0.645 (-5.1%)", "陽性率中央値80%"],
        ["Treatment", "good_fair", "0.654->0.636 (-1.8%)", "陽性率中央値75%"],
    ],
    [45, 35, 55, 55]
)
pdf.highlight_box("クラス不均衡により、陽性症例内でも陽性予測が少数派となり、多数決で陰性に倒れる", "orange")

# ===== Page 5: Threshold optimization =====
pdf.add_page()
pdf.subsection_title("Binaryタスク: 閾値最適化で大幅改善")
pdf.body_text("陽性画像比率 (pos_ratio) をスコアとし、閾値を下げることで検出力が大幅改善:")

pdf.add_table(
    ["Task", "Condition", "最適閾値", "Sens改善", "Case F1", "Case AUC"],
    [
        ["Aggressive ROP", "good_only", "0.05", "0.286->0.667", "0.791", "0.822"],
        ["Aggressive ROP", "good_fair", "0.15", "0.447->0.778", "0.863", "0.991"],
        ["Treatment", "good_only", "0.20", "0.696->0.806", "0.852", "0.916"],
        ["Treatment", "good_fair", "0.05", "0.654->0.909", "0.870", "0.939"],
    ],
    [40, 30, 25, 35, 30, 30]
)

pdf.highlight_box("good_fairのAggressive ROP Case-AUC 0.991 は非常に高い判別能力を示す", "green")

pdf.subsection_title("結論・推奨")
pdf.body_text("1. Multi-classタスク (Zone, Stage): 多数決アンサンブルで安定して改善\n2. Binaryタスク: 閾値を下げたアンサンブル、またはsoft voting (確率平均) を推奨\n3. good_fair > good_only: 画像数が多いほどアンサンブル効果が大きい\n4. 実装推奨: logit/softmax確率を保存したsoft votingが理想的")

# Save
output_path = "C:/Users/ykita/ROP_AI_project/Experimental_record/20260128_report.pdf"
pdf.output(output_path)
print(f"PDF saved to: {output_path}")
