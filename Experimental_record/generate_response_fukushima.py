# -*- coding: utf-8 -*-
"""
福嶋先生への回答PDF生成スクリプト
4項目: (a)動画IDリスト (b)Treatment FN (c)PR-AUC (d)Top-5画像不足
"""
import os
from fpdf import FPDF
from pathlib import Path

OUTPUT_DIR = Path(r"C:\Users\ykita\ROP_AI_project\Experimental_record")


class PDF(FPDF):
    def __init__(self):
        super().__init__()
        font_path = "C:/Windows/Fonts/meiryo.ttc"
        if os.path.exists(font_path):
            self.add_font("Meiryo", "", font_path)
            self.add_font("Meiryo", "B", "C:/Windows/Fonts/meiryob.ttc")
            self.jp_font = "Meiryo"
        else:
            self.jp_font = "Helvetica"

    def header(self):
        self.set_font(self.jp_font, "B", 9)
        self.set_text_color(100, 100, 100)
        self.cell(0, 8, "ROP AI Project - Response to Dr. Fukushima (2026-02-13)",
                  new_x="LMARGIN", new_y="NEXT", align="R")
        self.line(10, 18, 200, 18)
        self.ln(5)

    def footer(self):
        self.set_y(-15)
        self.set_font(self.jp_font, "", 8)
        self.set_text_color(128, 128, 128)
        self.cell(0, 10, f"Page {self.page_no()}/{self.pages_count}", align="C")

    def section_title(self, title):
        self.set_font(self.jp_font, "B", 14)
        self.set_text_color(30, 80, 150)
        self.set_fill_color(240, 245, 255)
        self.cell(0, 10, title, new_x="LMARGIN", new_y="NEXT", align="L", fill=True)
        self.ln(3)

    def subsection_title(self, title):
        self.set_font(self.jp_font, "B", 11)
        self.set_text_color(50, 50, 50)
        self.cell(0, 8, title, new_x="LMARGIN", new_y="NEXT", align="L")
        self.ln(1)

    def body_text(self, text):
        self.set_font(self.jp_font, "", 10)
        self.set_text_color(0, 0, 0)
        self.multi_cell(0, 6, text)
        self.ln(2)

    def small_text(self, text):
        self.set_font(self.jp_font, "", 9)
        self.set_text_color(80, 80, 80)
        self.multi_cell(0, 5, text)
        self.ln(1)

    def add_table(self, headers, data, col_widths=None, highlight_rows=None):
        self.set_font(self.jp_font, "B", 9)
        self.set_fill_color(70, 130, 180)
        self.set_text_color(255, 255, 255)
        if col_widths is None:
            col_widths = [190 / len(headers)] * len(headers)
        for i, h in enumerate(headers):
            self.cell(col_widths[i], 7, h, border=1, align="C", fill=True)
        self.ln()
        self.set_font(self.jp_font, "", 9)
        self.set_text_color(0, 0, 0)
        alt = False
        for ri, row in enumerate(data):
            if highlight_rows and ri in highlight_rows:
                self.set_fill_color(255, 255, 200)
                self.set_font(self.jp_font, "B", 9)
            elif alt:
                self.set_fill_color(245, 250, 255)
                self.set_font(self.jp_font, "", 9)
            else:
                self.set_fill_color(255, 255, 255)
                self.set_font(self.jp_font, "", 9)
            for i, cell in enumerate(row):
                self.cell(col_widths[i], 6, str(cell), border=1, align="C", fill=True)
            self.ln()
            alt = not alt
        self.ln(3)

    def highlight_box(self, text, color="blue"):
        colors = {
            "blue": ((230, 240, 255), (30, 60, 120)),
            "green": ((230, 255, 230), (30, 100, 30)),
            "orange": ((255, 245, 230), (150, 80, 0)),
            "red": ((255, 230, 230), (150, 30, 30)),
        }
        fill_c, text_c = colors.get(color, colors["blue"])
        self.set_fill_color(*fill_c)
        self.set_text_color(*text_c)
        self.set_font(self.jp_font, "B", 10)
        self.multi_cell(0, 6, text, border=1, fill=True)
        self.set_text_color(0, 0, 0)
        self.ln(2)

    def note_box(self, text):
        self.set_fill_color(255, 250, 240)
        self.set_text_color(120, 80, 0)
        self.set_font(self.jp_font, "", 9)
        self.multi_cell(0, 5, text, border=1, fill=True)
        self.set_text_color(0, 0, 0)
        self.ln(2)


def generate_pdf():
    pdf = PDF()
    pdf.set_auto_page_break(auto=True, margin=15)

    # ===== Page 1: Title =====
    pdf.add_page()
    pdf.ln(20)
    pdf.set_font(pdf.jp_font, "B", 22)
    pdf.set_text_color(30, 60, 120)
    pdf.multi_cell(0, 12, "福嶋先生への回答", align="C")
    pdf.ln(3)
    pdf.set_font(pdf.jp_font, "", 14)
    pdf.set_text_color(80, 80, 80)
    pdf.multi_cell(0, 8,
        "Multicenter ROP Study  AI解析結果に関するご質問への回答", align="C")
    pdf.ln(8)

    pdf.set_font(pdf.jp_font, "", 11)
    pdf.set_text_color(0, 0, 0)
    info_lines = [
        "データセット: Multicenter ROP Study (6,448画像, 347動画, 5施設)",
        "モデル: clinical_v3 (EfficientNet-B0 + 臨床データ融合, 5タスク)",
        "対象レポート:",
        "  - report_rwrop_threshold_optimization.pdf (2026-02-06)",
        "  - report_top10_majority_vote.pdf (2026-02-11)",
        "  - report_prauc_supplement.pdf (2026-02-13)",
        "",
        "回答作成日: 2026-02-13",
    ]
    for line in info_lines:
        pdf.cell(0, 7, line, new_x="LMARGIN", new_y="NEXT", align="C")

    pdf.ln(10)
    pdf.set_font(pdf.jp_font, "B", 12)
    pdf.cell(0, 8, "目次", new_x="LMARGIN", new_y="NEXT", align="C")
    pdf.set_font(pdf.jp_font, "", 11)
    toc = [
        "(a) 解析対象動画IDリストと除外動画について",
        "(b) Treatment (TR-ROP) False Negative のIDリスト",
        "(c) PR-AUC (Precision-Recall AUC) の結果",
        "(d) Top-5画像フィルタリングにおける画像不足33動画の詳細",
    ]
    for item in toc:
        pdf.cell(0, 7, item, new_x="LMARGIN", new_y="NEXT", align="C")

    # ===== Page 2-3: (a) 動画IDリスト =====
    pdf.add_page()
    pdf.section_title("(a) 解析対象動画IDリストと除外動画について")

    pdf.subsection_title("概要")
    pdf.add_table(
        ["項目", "動画数"],
        [
            ["患者データ全体", "409"],
            ["解析に使用された動画", "347"],
            ["除外された動画", "62"],
        ],
        col_widths=[120, 70],
    )

    pdf.body_text(
        "解析に使用された全347動画の詳細は training_video_list_clinical_v3.xlsx に記載しています。"
    )

    pdf.subsection_title("施設別内訳")
    pdf.add_table(
        ["施設", "動画数"],
        [
            ["OWCH (母子)", "109"],
            ["FU (福岡)", "90"],
            ["AMU (愛知)", "64"],
            ["KCMC (神奈川)", "43"],
            ["YCH", "41"],
            ["合計", "347"],
        ],
        col_widths=[120, 70],
        highlight_rows={5},
    )

    pdf.subsection_title("除外された62動画の内訳")
    pdf.add_table(
        ["除外理由", "動画数", "説明"],
        [
            ["Bad/Worst only", "47", "トレーニング時点でGood/Fair画像が0枚"],
            ["No Kubota selection", "15", "画像選定データなし (OMPU 8件含む)"],
            ["合計", "62", ""],
        ],
        col_widths=[50, 25, 115],
        highlight_rows={2},
    )

    pdf.note_box(
        "補足: 0098_OWCHは現在Fair画像1枚が存在しますが、トレーニングデータ\n"
        "スナップショット (2026-01-27) 以降に追加された分類バッチであり、\n"
        "トレーニング時点では Bad/Worst only に該当していました。\n"
        "同様に他9動画でも計43枚のFair画像が後から追加されています。"
    )

    pdf.subsection_title("除外動画のうち治療例 (treatment_needed = 1)")
    pdf.highlight_box(
        "7動画が治療例でしたが、いずれもBad/Worst品質のため除外されています。\n"
        "実質的には4患者分 (0016/0017_AMU, 0220/0221_FU, 0261/0262_YCH, 0274_YCH)。"
    )

    pdf.add_table(
        ["video_id", "施設", "GA", "BW", "PMA", "Zone", "Stage", "Plus", "理由"],
        [
            ["0016_AMU", "愛知", "23", "334", "33", "1", "2", "2", "Bad/Worst"],
            ["0017_AMU", "愛知", "23", "334", "33", "1", "2", "2", "Bad/Worst"],
            ["0220_FU", "福岡", "26", "859", "37", "2", "3", "2", "Bad/Worst"],
            ["0221_FU", "福岡", "26", "859", "37", "2", "3", "2", "Bad/Worst"],
            ["0261_YCH", "YCH", "24", "640", "42", "2", "3", "2", "Bad/Worst"],
            ["0262_YCH", "YCH", "24", "640", "42", "2", "3", "2", "Bad/Worst"],
            ["0274_YCH", "YCH", "23", "697", "31", "1", "3", "2", "Bad/Worst"],
        ],
        col_widths=[24, 16, 12, 14, 14, 14, 16, 14, 26],
    )

    pdf.note_box(
        "注: これらの除外はモデルの性能限界ではなく、撮影品質の問題です。\n"
        "Good/Fair品質の画像が得られなかったため、解析対象外としました。\n"
        "詳細は excluded_videos_from_analysis.csv をご参照ください。"
    )

    # ===== Page 3-4: (b) Treatment FN =====
    pdf.add_page()
    pdf.section_title("(b) Treatment (TR-ROP) False Negative のIDリスト")

    pdf.subsection_title("福嶋先生の仮説")
    pdf.note_box(
        "「実際、症例ベースで仮定すると治療例のうち1例見逃し、画像ベースでも\n"
        "治療画像の中で30-40画像が治療すべき画像を見逃したという計算なので\n"
        "同一症例でも正しく判定した画像と見逃した症例が混ざっているだけなのでは？\n"
        "と思っており、すでに実質感度100を達成しているのではないかと期待を持っています。」"
    )

    pdf.highlight_box(
        "結論: 福嶋先生の仮説は正しく、Soft Vote Top-5 / Top-10 で\n"
        "Treatment Sensitivity = 1.000 を達成しています。",
        color="green"
    )

    pdf.subsection_title("Treatment 画像単位の結果 (5-fold CV, default threshold 0.50)")
    pdf.body_text(
        "Treatment 陽性動画数: 33 (627画像)\n"
        "FN画像数: 52 / 627 (8.3%)\n"
        "FN画像を含む動画数: 10 / 33\n"
        "完全見逃し動画 (全画像FN): 0  --- 全33動画で少なくとも1枚はTP"
    )

    pdf.add_table(
        ["video_id", "FN / 全画像", "FN率", "mean P(Treat)", "SV判定"],
        [
            ["0034_KCMC", "13 / 22", "59%", "0.490", "SV AllでFN"],
            ["0259_OWCH", "9 / 22", "41%", "0.554", "SVでTP"],
            ["0009_AMU", "12 / 30", "40%", "0.625", "SVでTP"],
            ["0058_KCMC", "5 / 15", "33%", "0.632", "SVでTP"],
            ["0080_YCH", "1 / 4", "25%", "0.648", "SVでTP"],
            ["0144_OWCH", "6 / 27", "22%", "0.625", "SVでTP"],
            ["0048_KCMC", "3 / 27", "11%", "0.667", "SVでTP"],
            ["0077_OWCH", "1 / 9", "11%", "0.814", "SVでTP"],
            ["0008_AMU", "1 / 30", "3%", "0.822", "SVでTP"],
            ["0014_AMU", "1 / 29", "3%", "0.912", "SVでTP"],
        ],
        col_widths=[30, 30, 20, 35, 75],
        highlight_rows={0},
    )

    pdf.subsection_title("Treatment 患者単位 Soft Vote の結果")

    pdf.add_table(
        ["評価条件", "FN動画数", "Sensitivity", "備考"],
        [
            ["Per-Image (All)", "---", "0.9171", "画像単位: 575/627 TP"],
            ["SV All", "1 / 33", "0.9697", "FN: 0034_KCMC (P=0.490)"],
            ["SV Top-10", "0 / 33", "1.0000", "FN動画なし"],
            ["SV Top-5", "0 / 33", "1.0000", "FN動画なし"],
        ],
        col_widths=[40, 30, 30, 90],
        highlight_rows={2, 3},
    )

    pdf.subsection_title("解釈")
    pdf.body_text(
        "1. 画像単位では52枚のFNが存在するが、全10動画でTPとFNが混在しており、\n"
        "   完全見逃し (全画像FN) の動画は0です。\n\n"
        "2. Soft Vote All で唯一のFN動画は 0034_KCMC (Stage 3, P=0.490)。\n"
        "   22枚中13枚がFNですが、残り9枚のTPにより平均確率は閾値ギリギリ。\n\n"
        "3. Top-5/Top-10 品質フィルタリング後は Treatment Sensitivity = 1.000。\n"
        "   低品質画像を除外し、0034_KCMCの平均確率が閾値を超えました。\n\n"
        "4. 福嶋先生のご指摘通り、TP画像とFN画像が混在しているだけであり、\n"
        "   患者レベルでは実質的に感度100%を達成しています。"
    )

    # Supplementary: RW-ROP FN
    pdf.subsection_title("補足: RW-ROP (Plus OR Stage 3 OR Zone I) の False Negative")
    pdf.body_text(
        "RW-ROPのFNリストは rwrop_false_negatives_summary.md に詳細を記載。\n\n"
        "SV All: RW-ROP陽性 84動画中 11動画がFN (Sensitivity = 0.869)\n"
        "  - Zone I の見逃し (Zone II と誤判定) が最多 (7/11動画)\n"
        "  - 最も困難な症例: 0217_FU (Zone I + Stage 3, P(RW)=0.293)"
    )

    pdf.note_box(
        "Treatment (33動画) は RW-ROP (84動画) のサブセットです。\n"
        "RW-ROP FN 11動画のうち Treatment陽性は 0034_KCMC のみ。\n"
        "残り10動画は「治療不要だが紹介が必要」なケース (例: Zone I のみ) で、\n"
        "臨床的影響度は Treatment FN より低いと言えます。"
    )

    # ===== Page 5: (c) PR-AUC =====
    pdf.add_page()
    pdf.section_title("(c) PR-AUC (Precision-Recall AUC) の結果")

    pdf.body_text(
        "陽性ケースが少ない (Treatment 9.7%, AROP 3.5%, RW-ROP 24.5%) ため、\n"
        "ROC-AUCに加えてPR-AUCを算出しました。\n"
        "詳細は report_prauc_supplement.pdf (8ページ) をご参照ください。"
    )

    pdf.subsection_title("Per-Fold CV 結果 (Mean +/- SD)")
    pdf.add_table(
        ["タスク", "陽性率", "ROC-AUC", "PR-AUC", "差 (ROC-PR)"],
        [
            ["Treatment", "9.8%", "0.975 +/- 0.025", "0.860 +/- 0.125", "+0.116"],
            ["AROP", "3.6%", "0.888 +/- 0.131", "0.614 +/- 0.261", "+0.273"],
            ["RW-ROP", "22.1%", "0.954 +/- 0.018", "0.889 +/- 0.050", "+0.066"],
        ],
        col_widths=[30, 20, 45, 45, 30],
    )

    pdf.subsection_title("患者単位 Soft Vote PR-AUC")
    pdf.add_table(
        ["Task", "Per-Image", "SV All", "SV Top-10", "SV Top-5"],
        [
            ["Treatment", "0.880", "0.925", "0.923", "0.905"],
            ["AROP", "0.649", "0.675", "0.660", "0.617"],
            ["RW-ROP", "0.892", "0.898", "0.891", "0.885"],
        ],
        col_widths=[30, 40, 40, 40, 40],
    )

    pdf.subsection_title("解釈")
    pdf.body_text(
        "1. Treatment: PR-AUC 0.860 はランダム分類器 (陽性率 0.098) の約9倍。\n"
        "   Soft Vote で 0.925 まで改善。実質的に高い性能。\n\n"
        "2. AROP: ROC-AUC (0.888) と PR-AUC (0.614) の乖離が最大 (+0.273)。\n"
        "   陽性率 3.5% の強い不均衡が原因。ROC-AUCほど精度は高くない。\n\n"
        "3. RW-ROP: PR-AUC 0.889 はスクリーニング指標として実用的水準。"
    )

    pdf.note_box(
        "発表用グラフ: report_prauc_supplement.pdf にROC曲線とPR曲線のグラフを\n"
        "含めています。必要に応じて高解像度の図を再出力できます。"
    )

    # ===== Page 6: (d) Top-5 画像不足 =====
    pdf.add_page()
    pdf.section_title("(d) Top-5画像フィルタリング: 画像不足33動画の詳細")

    pdf.highlight_box(
        "結論:\n"
        "1. 画像不足の33動画は除外されず、選出できた画像 (1-4枚) のまま解析されている\n"
        "2. 該当画像が0枚の動画はない (全347動画で1枚以上選出)\n"
        "3. 最小選出数は1枚",
        color="green"
    )

    pdf.subsection_title("Top-5 選出の内訳")
    pdf.add_table(
        ["項目", "値"],
        [
            ["全画像数", "6,448"],
            ["Top-5選出後", "1,650"],
            ["対象動画数", "347 / 347 (全動画カバー)"],
            ["Stage1のみで十分", "301 (86.7%)"],
            ["Stage2補完あり", "13 (3.7%)"],
            ["画像不足 (<5枚)", "33 (9.5%)"],
            ["特徴量なし (0枚)", "0"],
            ["最小選出数", "1枚"],
            ["平均選出数", "4.8枚"],
        ],
        col_widths=[80, 110],
        highlight_rows={5, 6},
    )

    pdf.subsection_title("選出アルゴリズム")
    pdf.body_text(
        "Step 1: 動画ごとの全Good+Fair画像から有効画像を抽出 (retina_ratio > 0)\n"
        "Step 2: disc_edge_coverage_ratio >= 0.80 の画像を品質スコアで降順ソート\n"
        "        品質スコア = 0.4 x retina_ratio + 0.4 x Grad_p90 + 0.2 x mbss_score\n"
        "Step 3: 上位K枚を選出。不足時は残り画像を retina_ratio 降順で補完\n"
        "Step 4: それでもK枚未満 → 「画像不足」カウント (ある分だけで解析)"
    )

    pdf.subsection_title("臨床的留意点")
    pdf.body_text(
        "- 画像枚数が少ない動画では Soft Vote の信頼性が低下する可能性あり\n"
        "- ただし Top-5 でも Treatment Sensitivity = 1.000 を達成しており、\n"
        "  画像不足による見逃しは発生していません"
    )

    # ===== Page 7: Reference files =====
    pdf.add_page()
    pdf.section_title("参照ファイル一覧")

    pdf.add_table(
        ["ファイル名", "内容"],
        [
            ["report_rwrop_threshold_optimization.pdf", "閾値最適化レポート (2026-02-06)"],
            ["report_top10_majority_vote.pdf", "Top-K Majority Vote レポート (2026-02-11)"],
            ["report_prauc_supplement.pdf", "PR-AUC補足解析レポート (2026-02-13)"],
            ["training_video_list_clinical_v3.xlsx", "解析対象347動画リスト"],
            ["excluded_videos_from_analysis.csv", "除外62動画リスト (除外理由付き)"],
            ["rwrop_false_negatives_summary.md", "RW-ROP FN動画IDサマリ"],
            ["rwrop_false_negatives_threshold_opt..csv", "Report 1 FN詳細CSV"],
            ["rwrop_false_negatives_majority_vote.csv", "Report 2 FN詳細CSV"],
            ["top10_quality_filter_insufficient...md", "画像不足動画の詳細レポート"],
        ],
        col_widths=[90, 100],
    )

    # Save
    out_path = OUTPUT_DIR / "response_to_fukushima_dr_20260213.pdf"
    pdf.output(str(out_path))
    print(f"PDF generated: {out_path}")
    return out_path


if __name__ == "__main__":
    generate_pdf()
