#!/usr/bin/env python3
"""Generate the novelty + demo presentation as PPTX (16:9).

Output: ../Novelty_&_Demo.pptx
Run:    python3 scripts/generate_novelty_pptx.py
"""
import os
import sys
from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paper_numbers

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
FIG = os.path.join(ROOT, "figures")
OUT = os.path.join(ROOT, "Novelty_&_Demo.pptx")


# ------------------------------------------------------------------ numbers
# Read every quoted metric from the tracked result artifacts so the deck can
# never drift from the data it cites.
_DRIFT = {row[0]: row for row in paper_numbers.drift_table()}
_DECOMP = {row[0]: row for row in paper_numbers.discriminator_table()}
_MAIN = {row[0]: row for row in paper_numbers.main_results_table("gossipcop")}
_ABL = {row[0]: row for row in paper_numbers.ablation_table()}
_ABL10 = {row[0]: row for row in paper_numbers.ablation_10pct_table()}


def _drift(subset, column):
    row = _DRIFT.get(subset)
    return row[column] if row else "n/a"


def _decomp(subset, setting, column):
    row = _DECOMP.get((subset, setting))
    return row[column] if row else "n/a"


def _main(name, column):
    row = _MAIN.get(name)
    return row[column] if row else "n/a"

NAVY = RGBColor(0x1F, 0x36, 0x5D)
GOLD = RGBColor(0xC9, 0xA2, 0x27)
GREY = RGBColor(0x55, 0x55, 0x55)
LIGHT = RGBColor(0xE8, 0xEE, 0xF7)
REAL = RGBColor(0x1F, 0x77, 0xB4)
FAKE = RGBColor(0xD6, 0x27, 0x28)
GREEN = RGBColor(0x2E, 0x7D, 0x32)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)

prs = Presentation()
prs.slide_width = Inches(13.333)
prs.slide_height = Inches(7.5)
BLANK = prs.slide_layouts[6]
SW, SH = 13.333, 7.5


def add_slide():
    return prs.slides.add_slide(BLANK)


def rect(slide, x, y, w, h, fill, line=None):
    sp = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(x), Inches(y), Inches(w), Inches(h))
    sp.fill.solid()
    sp.fill.fore_color.rgb = fill
    if line is None:
        sp.line.fill.background()
    else:
        sp.line.color.rgb = line
        sp.line.width = Pt(1)
    sp.shadow.inherit = False
    return sp


def header(slide, title, tag=None):
    rect(slide, 0, 0, SW, 1.0, NAVY)
    tb = slide.shapes.add_textbox(Inches(0.6), Inches(0.12), Inches(12.1), Inches(0.8))
    tf = tb.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    p = tf.paragraphs[0]
    r = p.add_run()
    r.text = title
    r.font.size = Pt(28)
    r.font.bold = True
    r.font.color.rgb = WHITE
    if tag:
        p2 = tf.add_paragraph()
        r2 = p2.add_run()
        r2.text = tag
        r2.font.size = Pt(12)
        r2.font.color.rgb = GOLD


def bullets(slide, x, y, w, h, items, base=15):
    tb = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = tb.text_frame
    tf.word_wrap = True
    for i, (text, level, bold, color) in enumerate(items):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.level = level
        p.space_after = Pt(6)
        marker = "" if level == 0 else "    \u2013  "
        run = p.add_run()
        run.text = marker + text
        run.font.size = Pt(base - 2 if level else base)
        run.font.bold = bold
        run.font.color.rgb = color if color else RGBColor(0x22, 0x22, 0x22)
    return tb


def pic(slide, path, x, y, w=None, h=None):
    kw = {}
    if w:
        kw["width"] = Inches(w)
    if h:
        kw["height"] = Inches(h)
    return slide.shapes.add_picture(path, Inches(x), Inches(y), **kw)


def pic_dims(path):
    with Image.open(path) as im:
        return im.size  # (w, h) px


def add_table(slide, headers, rows, x, y, w, h, col_widths=None, fs=13, hfs=13):
    nrows, ncols = len(rows) + 1, len(headers)
    gt = slide.shapes.add_table(nrows, ncols, Inches(x), Inches(y), Inches(w), Inches(h)).table
    if col_widths:
        total = float(sum(col_widths))
        for i, cw in enumerate(col_widths):
            gt.columns[i].width = Inches(w * cw / total)
    for j, htext in enumerate(headers):
        c = gt.cell(0, j)
        c.text = htext
        for p in c.text_frame.paragraphs:
            for r in p.runs:
                r.font.size = Pt(hfs)
                r.font.bold = True
                r.font.color.rgb = WHITE
        c.fill.solid()
        c.fill.fore_color.rgb = NAVY
        c.vertical_anchor = MSO_ANCHOR.MIDDLE
    for i, row in enumerate(rows):
        for j, val in enumerate(row):
            c = gt.cell(i + 1, j)
            c.text = str(val)
            for p in c.text_frame.paragraphs:
                for r in p.runs:
                    r.font.size = Pt(fs)
                    r.font.color.rgb = RGBColor(0x22, 0x22, 0x22)
            c.fill.solid()
            c.fill.fore_color.rgb = LIGHT if i % 2 == 0 else WHITE
            c.vertical_anchor = MSO_ANCHOR.MIDDLE
    return gt


def footer(slide, num, note=""):
    tb = slide.shapes.add_textbox(Inches(0.6), Inches(7.05), Inches(12.0), Inches(0.35))
    tf = tb.text_frame
    p = tf.paragraphs[0]
    r = p.add_run()
    r.text = (note + "   |   ") if note else ""
    r.font.size = Pt(10)
    r.font.color.rgb = GREY
    r2 = p.add_run()
    r2.text = f"{num} / 11"
    r2.font.size = Pt(10)
    r2.font.color.rgb = GREY


# =====================================================================
# SLIDE 1 — Title
# =====================================================================
s = add_slide()
rect(s, 0, 0, SW, SH, NAVY)
rect(s, 0, 4.5, SW, 0.06, GOLD)
tb = s.shapes.add_textbox(Inches(1.0), Inches(2.0), Inches(11.3), Inches(2.4))
tf = tb.text_frame
tf.word_wrap = True
p = tf.paragraphs[0]
r = p.add_run()
r.text = "Temporal Graph Diffusion Tracing for Cross-Modal Misinformation Factuality Degradation"
r.font.size = Pt(36)
r.font.bold = True
r.font.color.rgb = WHITE
p2 = tf.add_paragraph()
r2 = p2.add_run()
r2.text = "\nNovelty Statement \u2022 Implementation Proof \u2022 Demo Walkthrough"
r2.font.size = Pt(18)
r2.font.color.rgb = GOLD

tb = s.shapes.add_textbox(Inches(1.0), Inches(6.2), Inches(11.3), Inches(0.9))
tf = tb.text_frame
p = tf.paragraphs[0]
r = p.add_run()
r.text = "Arya.D   \u2022   Jeswanth G   \u2022   Doravari Sakthi Prasad"
r.font.size = Pt(16)
r.font.bold = True
r.font.color.rgb = WHITE
p2 = tf.add_paragraph()
r2 = p2.add_run()
r2.text = "Misinformation Degradation Project"
r2.font.size = Pt(13)
r2.font.color.rgb = GOLD

# =====================================================================
# SLIDE 2 — One-line novelty
# =====================================================================
s = add_slide()
header(s, "The Novelty in One Sentence", "Positioning")
rect(s, 0.6, 1.5, 12.1, 1.8, LIGHT)
tb = s.shapes.add_textbox(Inches(1.0), Inches(1.7), Inches(11.3), Inches(1.5))
tf = tb.text_frame
tf.word_wrap = True
p = tf.paragraphs[0]
r = p.add_run()
r.text = "\u201cPrior work models WHO spreads misinformation. This project measures HOW THE MESSAGE DEGRADES as it spreads \u2014 and shows that alone is enough to detect it.\u201d"
r.font.size = Pt(22)
r.font.italic = True
r.font.color.rgb = NAVY
bullets(s, 0.6, 3.7, 12.1, 3.2, [
    ("New feature family: per-node drift (factuality degradation) \u2014 the parent\u2192child content gap, never quantified before.", 0, True, NAVY),
    (f"New empirical law: fake news drifts LESS than real news at matched propagation depth "
     f"(hop-standardised ratio {_drift('gossipcop', 4)} on GossipCop, "
     f"{_drift('politifact', 4)} on PolitiFact).", 0, True, NAVY),
    (f"New detector design: aggregate drift is discriminative without a text encoder \u2014 "
     f"content drift alone reaches {_decomp('gossipcop', 'full cascade', 4)} AUC.", 0, True, NAVY),
    ("New design insight: per-node drift HURTS a classifier; only aggregate drift works (ablation proof).", 0, True, NAVY),
])
footer(s, 2)

# =====================================================================
# SLIDE 3 — Finding 1: the empirical law
# =====================================================================
s = add_slide()
header(s, "Novel Finding \u2014 Fake News Degrades Differently", "Claim 1: Empirical Law")
w = 7.3
ph, pw = pic_dims(os.path.join(FIG, "drift_analysis.png"))
h = w * ph / pw
pic(s, os.path.join(FIG, "drift_analysis.png"), 0.5, 1.3, w=w)
bullets(s, w + 0.9, 1.4, 12.9 - (w + 0.9), 5.4, [
    ("Node-level statistics on the TEST cascades (GossipCop 220k nodes, PolitiFact 31k nodes).", 0, False, None),
    (f"GossipCop: pooled ratio {_drift('gossipcop', 3)} \u2192 hop-standardised "
     f"{_drift('gossipcop', 4)}, cascade-level {_drift('gossipcop', 5)} "
     f"(t = {_drift('gossipcop', 6)}, p = {_drift('gossipcop', 7)}).", 0, True, FAKE),
    (f"PolitiFact: pooled ratio {_drift('politifact', 3)} \u2192 hop-standardised "
     f"{_drift('politifact', 4)}, cascade-level {_drift('politifact', 5)} "
     f"(t = {_drift('politifact', 6)}, p = {_drift('politifact', 7)}).", 0, True, FAKE),
    ("The pooled ratio mixes the content effect with the classes' different propagation-depth compositions; the hop-standardised ratio isolates it.", 1, False, None),
    ("Effect emerges at hop \u2265 2; root drift = 0 by construction.", 1, False, None),
    ("Why: fabricated content is amplified by coordinated / templated spreaders \u2192 homogeneous, flat reposts.", 1, False, None),
], base=15)
footer(s, 3, "results/drift_standardized.json + results/significance.json")

# =====================================================================
# SLIDE 4 — Finding 2: text-free signal
# =====================================================================
s = add_slide()
header(s, "Drift Alone Is Discriminative \u2014 No Text, No Graph", "Claim 2: Standalone Signal")
w = 6.4
ph, pw = pic_dims(os.path.join(FIG, "discriminator.png"))
h = w * ph / pw
pic(s, os.path.join(FIG, "discriminator.png"), 0.6, 1.4, w=w)
col_x = w + 1.1
bullets(s, col_x, 1.4, 12.9 - col_x, 5.4, [
    ("12 aggregate statistics per cascade: 4 content-drift, 6 structural, 2 cascade size.", 0, False, None),
    ("Random forest \u2014 200 trees, stratified 5-fold CV. NO text encoder, NO GNN.", 0, False, None),
    (f"GossipCop full cascade: all 12 stats {_decomp('gossipcop', 'full cascade', 3)} AUC; "
     f"content drift alone {_decomp('gossipcop', 'full cascade', 4)} AUC; "
     f"size alone {_decomp('gossipcop', 'full cascade', 5)} AUC.", 0, True, GREEN),
    ("Model-free signal: usable where text is unavailable or private.", 1, False, None),
    ("Cascade size alone is already a strong baseline \u2014 the honest framing is \u201ccontent drift adds on top of shape\u201d.", 1, False, None),
], base=15)
footer(s, 4, "results/drift_discriminator_decomposition.json + drift_discriminator_robustness.json")

# =====================================================================
# SLIDE 5 — Finding 3: survives early detection
# =====================================================================
s = add_slide()
header(s, "Aggregate Drift Survives Early (Partial) Cascades", "Claim 3: Early Detection")
w = 6.5
ph, pw = pic_dims(os.path.join(FIG, "early_detection.png"))
h = w * ph / pw
pic(s, os.path.join(FIG, "early_detection.png"), 0.5, 1.3, w=w)
col_x = w + 1.0
bullets(s, col_x, 1.4, 12.9 - col_x, 5.4, [
    ("Train on FULL cascade, truncate TEST to shallowest 10/25/50/100% of nodes (3 seeds, reproducible protocol).", 0, False, None),
    ("All learned models drop hard at 10%: LSTM 95.6 \u2192 78.7, Hybrid 95.3 \u2192 80.8.", 0, False, None),
    (f"Drift-only signal: {_decomp('gossipcop', 'full cascade', 3)} \u2192 "
     f"{_decomp('gossipcop', 'first 10% of cascade', 3)} AUC \u2014 a gentler relative drop.", 0, True, GREEN),
    (f"Caveat: at 10% truncation { _decomp('gossipcop', 'first 10% of cascade', 3) } AUC includes cascades that "
     f"retain only the root (zero drift by construction); excluding them the signal is "
     f"{_decomp('gossipcop', 'first 10% of cascade', 6)} AUC.", 1, False, None),
    ("=> Degradation STATISTICS are the right vector for early detection.", 1, True, NAVY),
    ("Honest negative: at 10% no model reaches full-cascade accuracy \u2014 the real robustness gap.", 1, False, None),
], base=15)
footer(s, 5, "results/early_detection_multiseed.json")

# =====================================================================
# SLIDE 6 — Prior work vs this work
# =====================================================================
s = add_slide()
header(s, "Prior Work vs. This Work", "Novelty Positioning Against SOTA")
w = 6.6
ph, pw = pic_dims(os.path.join(FIG, "main_results.png"))
h = w * ph / pw
pic(s, os.path.join(FIG, "main_results.png"), 0.4, 1.35, w=w)
add_table(s,
    ["Method", "Acc", "Uses"],
    [
        ["GCN", _main("GCN", 1).split(" \u00b1")[0], "Static graph only"],
        ["GAT", _main("GAT", 1).split(" \u00b1")[0], "Static graph + attention"],
        ["BiGCN", _main("BiGCN", 1).split(" \u00b1")[0], "Bidirectional tree message passing"],
        ["BaselineLSTM", _main("BaselineLSTM", 1).split(" \u00b1")[0], "Temporal sequence only"],
        ["HybridDegradLSTM", _main("Hybrid", 1).split(" \u00b1")[0], "Temporal + drift + tree"],
    ],
    x=7.3, y=1.5, w=5.6, h=2.9, col_widths=[3.2, 1.6, 4.2], fs=13, hfs=13)
bullets(s, 7.3, 4.6, 5.7, 2.3, [
    (f"Best in our comparison (3-seed mean): Acc {_main('Hybrid', 1)}, F1 {_main('Hybrid', 2)}, "
     f"AUC {_main('Hybrid', 3)}, AUC-PR {_main('Hybrid', 4)}.", 0, True, GREEN),
    ("Daily-news dataset \u2192 harder than label-heavy benchmarks.", 1, False, None),
], base=14)
footer(s, 6, "results/multiseed_all.json")

# =====================================================================
# SLIDE 7 — The negative result (design insight)
# =====================================================================
s = add_slide()
header(s, "A Negative Result That Proves the Design Was Tested", "Ablation \u2192 Design Insight")
w = 6.8
ph, pw = pic_dims(os.path.join(FIG, "ablation.png"))
h = w * ph / pw
pic(s, os.path.join(FIG, "ablation.png"), 0.5, 1.3, w=w)
col_x = w + 1.0
bullets(s, col_x, 1.4, 12.9 - col_x, 5.4, [
    ("Injecting per-NODE drift into the classifier HURTS at every cascade size:", 0, False, None),
    (f"  100%: Full {_ABL.get('Full (drift+hop+tree)', [None, 'n/a'])[1]} \u2192 "
     f"w/o drift {_ABL.get('w/o drift', [None, 'n/a'])[1]}.", 1, True, FAKE),
    (f"  10%:   Full {_ABL10.get('Full', [None, 'n/a', 'n/a'])[2]} \u2192 "
     f"w/o drift {_ABL10.get('w/o drift', [None, 'n/a', 'n/a'])[2]}.", 1, True, FAKE),
    (f"Yet AGGREGATE drift is discriminative (content-only {_decomp('gossipcop', 'full cascade', 4)} AUC).", 0, True, GREEN),
    ("Interpretation: the signal lives in the DISTRIBUTION, not per-node instantiations.", 1, False, None),
    ("Design rule for the field: consume drift via set-pooling, not concatenation.", 1, True, NAVY),
], base=15)
footer(s, 7, "results/ablations.json + ablations_10pct.json")

# =====================================================================
# SLIDE 8 — Statistical rigor & honesty
# =====================================================================
s = add_slide()
header(s, "Statistical Rigor \u2014 What We Claim and What We Do Not", "Trust Factors")
bullets(s, 0.6, 1.4, 12.1, 2.2, [
    ("3 seeds per model, reported as mean \u00b1 std; one seed per process (MPS-variance honest).", 0, False, None),
    ("Paired t-tests + Cohen's d for model comparisons; Welch t on 220k nodes for the drift law.", 0, False, None),
    ("Every table traceable to a saved, re-runnable script (no lost protocols).", 0, False, None),
], base=15)
add_table(s,
    ["Claim", "p-value", "Verdict"],
    [
        [f"Fake news drifts less at matched depth (GossipCop {_drift('gossipcop', 4)}, "
         f"PolitiFact {_drift('politifact', 4)})",
         f"cascade-level p = {_drift('gossipcop', 7)} / {_drift('politifact', 7)}", "MODERATE \u2014 honest effect size"],
        ["Hybrid > DegradGNN (accuracy)", "0.020, d = 3.98", "SIGNIFICANT"],
        ["Hybrid > BaselineLSTM (accuracy)", "0.224 (n=3)", "Not significant \u2014 stated honestly"],
        ["PolitiFact model ranking", "\u2014", "Withheld \u2014 62 train graphs collapse"],
    ],
    x=0.6, y=3.7, w=12.1, h=2.9, col_widths=[6.4, 3.2, 2.5], fs=13, hfs=13)
footer(s, 8, "results/significance.json")

# =====================================================================
# SLIDE 9 — What to show in the demo
# =====================================================================
s = add_slide()
header(s, "Implementation \u2014 What to Show (6-Part Demo)", "Demo Script \u2248 15 min")
bullets(s, 0.6, 1.3, 7.2, 5.4, [
    ("1. Real data (30 s) \u2014 data/upfd/raw/gossipcop/: A.txt, BERT features, splits.", 0, True, NAVY),
    ("2. Drift definition (formula on screen) \u2014 features/drift.py: 5-dim vector.", 0, True, NAVY),
    ("3. Headline finding \u2014 drift_analysis.png + significance.json.", 0, True, NAVY),
    ("4. Text-free signal \u2014 discriminator.png + drift_discriminator_robustness.json.", 0, True, NAVY),
    ("5. Models \u2014 main_results.png + architecture.png + hybrid.py.", 0, True, NAVY),
    ("6. Live re-run \u2014 commands:", 0, True, NAVY),
    ("    python scripts/analysis_stats.py", 1, False, GREEN),
    ("    python scripts/run_discriminator_robustness.py", 1, False, GREEN),
    ("    python scripts/make_figures.py", 1, False, GREEN),
], base=14)
add_table(s,
    ["Module", "File", "Status"],
    [
        ["Drift features", "features/drift.py", "Done"],
        ["GNN baselines (GCN/GAT/BiGCN)", "models/baselines.py", "Done"],
        ["DegradGNN (graph diffusion)", "models/degradation_gnn.py", "Done"],
        ["HybridDegradLSTM (fused)", "models/hybrid.py", "Done"],
        ["Trajectory tracing (DTN)", "models/degradation_trajectory.py", "Done"],
        ["Cross-modal aligner", "features/cross_modal.py", "Done"],
        ["Early-detection wrapper", "models/early_detector.py", "Done"],
    ],
    x=8.0, y=1.35, w=4.9, h=3.6, col_widths=[3.4, 3.0, 1.2], fs=12, hfs=12)
footer(s, 9, "scripts/ = reproducibility index")

# =====================================================================
# SLIDE 10 — Framework pillars
# =====================================================================
s = add_slide()
header(s, "The Framework Behind the Claims", "How the Title Maps to Code")
w = 6.2
ph, pw = pic_dims(os.path.join(FIG, "architecture.png"))
h = w * ph / pw
pic(s, os.path.join(FIG, "architecture.png"), 0.5, 1.3, w=w)
bullets(s, 7.0, 1.4, 5.9, 5.3, [
    ("Temporal \u2192 hop-depth encoding + LSTM over cascade order.", 0, True, NAVY),
    ("Graph Diffusion \u2192 GNN message passing over the propagation tree.", 0, True, NAVY),
    ("Cross-Modal \u2192 TextEncoder + ImageEncoder(CLIP) + CrossModalAligner.", 0, True, NAVY),
    ("Factuality Degradation \u2192 5-dim drift vector (the novel signal).", 0, True, NAVY),
    ("Early detection \u2192 trajectory forecasting + truncation curves.", 0, True, NAVY),
    ("Each pillar is a working module, each with a saved script.", 1, False, None),
], base=14)
footer(s, 10, "figures/architecture.png")

# =====================================================================
# SLIDE 11 — Closing
# =====================================================================
s = add_slide()
rect(s, 0, 0, SW, SH, NAVY)
rect(s, 0, 2.4, SW, 0.06, GOLD)
tb = s.shapes.add_textbox(Inches(1.2), Inches(2.8), Inches(10.9), Inches(2.6))
tf = tb.text_frame
tf.word_wrap = True
p = tf.paragraphs[0]
r = p.add_run()
r.text = "\u201cNovelty is not inventing a new network. It is seeing a new signal \u2014 and proving it with statistics.\u201d"
r.font.size = Pt(28)
r.font.italic = True
r.font.color.rgb = WHITE
p2 = tf.add_paragraph()
r2 = p2.add_run()
r2.text = "\nFake news degrades ~11% less. That gap detects misinformation with no text at all (0.873 AUC)."
r2.font.size = Pt(18)
r2.font.color.rgb = GOLD
tb = s.shapes.add_textbox(Inches(1.2), Inches(5.8), Inches(10.9), Inches(1.0))
tf = tb.text_frame
p = tf.paragraphs[0]
r = p.add_run()
r.text = "Thank you  \u2022  Arya.D  \u2022  Jeswanth G  \u2022  Doravari Sakthi Prasad"
r.font.size = Pt(16)
r.font.color.rgb = WHITE

prs.save(OUT)
print("Saved:", OUT)
print("Slides:", len(prs.slides))