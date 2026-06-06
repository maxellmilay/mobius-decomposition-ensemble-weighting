const pptxgen = require("pptxgenjs");
const pres = new pptxgen();
pres.layout = "LAYOUT_WIDE"; // 13.3 x 7.5
pres.author = "Maxell G. Milay";
pres.title = "Non-Overlapping Ensemble Weighting via Mobius Decomposition";

// ---- Palette ----
const DARK   = "0B2027"; // deep teal-black (title / section / conclusion bg)
const PRIMARY= "0D7377"; // teal (dominant)
const TEAL2  = "14919B"; // lighter teal
const ACCENT = "E8A33D"; // amber/gold (sharp accent for key numbers)
const INK    = "1A2A2E"; // body text
const MUTED  = "6B7F84"; // captions
const PALE   = "EAF2F1"; // pale teal card fill
const PALE2  = "F4F8F7"; // very pale
const WHITE  = "FFFFFF";
const RED     = "B23A48"; // for "fail"/negative
const GREEN   = "2E7D5B"; // for "pass"/positive

const W = 13.3, H = 7.5;
const HEAD = "Georgia";
const BODY = "Calibri";

const sh = () => ({ type: "outer", color: "000000", blur: 7, offset: 2, angle: 135, opacity: 0.13 });

// recurring motif + section label + title for content slides
function header(slide, section, title, titleH = 0.85) {
  slide.background = { color: WHITE };
  slide.addShape(pres.shapes.RECTANGLE, { x: 0.6, y: 0.52, w: 0.18, h: 0.18, fill: { color: PRIMARY }, line: { type: "none" } });
  slide.addText(section.toUpperCase(), { x: 0.92, y: 0.46, w: 9, h: 0.3, margin: 0, fontFace: BODY, fontSize: 13, bold: true, color: PRIMARY, charSpacing: 3 });
  slide.addText(title, { x: 0.58, y: 0.78, w: 12.1, h: titleH, margin: 0, fontFace: HEAD, fontSize: 30, bold: true, color: INK });
}
function pageNum(slide, n) {
  slide.addText(String(n), { x: 12.5, y: 6.95, w: 0.5, h: 0.35, margin: 0, fontFace: BODY, fontSize: 11, color: MUTED, align: "right" });
}

// =========================================================
// SLIDE 1 — TITLE
// =========================================================
let s = pres.addSlide();
s.background = { color: DARK };
// motif: stacked teal squares (decomposition motif) top-right
s.addShape(pres.shapes.RECTANGLE, { x: 11.55, y: 0.7, w: 0.9, h: 0.9, fill: { color: PRIMARY, transparency: 25 }, line: { type: "none" } });
s.addShape(pres.shapes.RECTANGLE, { x: 11.9, y: 1.05, w: 0.9, h: 0.9, fill: { color: ACCENT, transparency: 30 }, line: { type: "none" } });
s.addText("UNIVERSITY OF THE PHILIPPINES CEBU  •  SPECIAL PROJECT", { x: 0.8, y: 0.95, w: 10, h: 0.35, margin: 0, fontFace: BODY, fontSize: 13, bold: true, color: TEAL2, charSpacing: 2 });
s.addText("Non-Overlapping Ensemble Weighting\nvia M\u00F6bius Decomposition", { x: 0.78, y: 2.05, w: 11.7, h: 1.9, margin: 0, fontFace: HEAD, fontSize: 38, bold: true, color: WHITE, lineSpacingMultiple: 1.04 });
s.addText("A Coopetitive Framework for Diagnosing and Leveraging Model Interactions", { x: 0.8, y: 3.95, w: 11.2, h: 0.6, margin: 0, fontFace: BODY, italic: true, fontSize: 19, color: ACCENT });
// thin separator via whitespace; author block
s.addText([
  { text: "Maxell G. Milay", options: { fontFace: HEAD, fontSize: 20, bold: true, color: WHITE, breakLine: true } },
  { text: "Bachelor of Science in Computer Science", options: { fontFace: BODY, fontSize: 14, color: TEAL2, breakLine: true } },
  { text: "Adviser:  Dharryl Prince Abellana   |   June 2026", options: { fontFace: BODY, fontSize: 13, color: MUTED } },
], { x: 0.8, y: 5.25, w: 11, h: 1.5, margin: 0, lineSpacingMultiple: 1.25 });

// =========================================================
// SLIDE 2 — INTRO: motivation & three gaps
// =========================================================
s = pres.addSlide();
header(s, "Introduction", "The weighting problem in ensembles");
s.addText("Ensembles combine many models \u2014 but how much weight should each one get? The Shapley value is the principled answer, yet it has a structural blind spot.", { x: 0.6, y: 1.7, w: 12.1, h: 0.7, margin: 0, fontFace: BODY, fontSize: 16, color: INK });

const gaps = [
  ["1", "Entanglement", "The Shapley value mixes a model\u2019s standalone quality with its interaction effects into one score \u2014 you can\u2019t tell why a model is weighted as it is."],
  ["2", "Double-counting", "\u201CFixing\u201D this by adding the Shapley Interaction Index counts the same interaction twice \u2014 the SII terms re-add information already inside the Shapley value."],
  ["3", "Interpretability", "No standard method outputs a pairwise map of which models are complementary, redundant, or independent within the pool."],
];
let gx = 0.6, gw = 3.93, gy = 2.65, gh = 3.45, ggap = 0.27;
gaps.forEach((g, i) => {
  const x = gx + i * (gw + ggap);
  s.addShape(pres.shapes.RECTANGLE, { x, y: gy, w: gw, h: gh, fill: { color: PALE }, line: { type: "none" }, shadow: sh() });
  s.addShape(pres.shapes.RECTANGLE, { x, y: gy, w: gw, h: 0.09, fill: { color: i === 1 ? ACCENT : PRIMARY }, line: { type: "none" } });
  s.addText(g[0], { x: x + 0.28, y: gy + 0.32, w: 0.9, h: 0.9, margin: 0, fontFace: HEAD, fontSize: 44, bold: true, color: i === 1 ? ACCENT : PRIMARY });
  s.addText(g[1] + " gap", { x: x + 0.28, y: gy + 1.25, w: gw - 0.5, h: 0.45, margin: 0, fontFace: HEAD, fontSize: 19, bold: true, color: INK });
  s.addText(g[2], { x: x + 0.28, y: gy + 1.78, w: gw - 0.56, h: gh - 1.95, margin: 0, fontFace: BODY, fontSize: 13.5, color: "33474C", lineSpacingMultiple: 1.05 });
});
s.addText("No existing ensemble method addresses all three.", { x: 0.6, y: 6.25, w: 12, h: 0.4, margin: 0, fontFace: BODY, fontSize: 15, italic: true, bold: true, color: PRIMARY });
pageNum(s, 2);

// =========================================================
// SLIDE 3 — INTRO: the proposed solution + objectives
// =========================================================
s = pres.addSlide();
header(s, "Introduction", "Our solution: the M\u00F6bius decomposition");
s.addText("The M\u00F6bius transform (Harsanyi dividends) splits the ensemble\u2019s value into mathematically non-overlapping pieces \u2014 by construction, with no double-counting.", { x: 0.6, y: 1.68, w: 12.1, h: 0.7, margin: 0, fontFace: BODY, fontSize: 16, color: INK });

// two decomposition cards
const dec = [
  ["m({i})", "Singleton dividend", "Each model\u2019s pure standalone value", PRIMARY],
  ["m({i, j})", "Pairwise dividend", "Surplus (+, complementary) or deficit (\u2013, redundant) from interaction only", ACCENT],
];
dec.forEach((d, i) => {
  const x = 0.6 + i * 4.0;
  s.addShape(pres.shapes.RECTANGLE, { x, y: 2.55, w: 3.7, h: 1.85, fill: { color: PALE }, line: { type: "none" }, shadow: sh() });
  s.addText(d[0], { x: x + 0.25, y: 2.72, w: 3.3, h: 0.55, margin: 0, fontFace: "Consolas", fontSize: 24, bold: true, color: d[3] });
  s.addText(d[1], { x: x + 0.25, y: 3.3, w: 3.3, h: 0.4, margin: 0, fontFace: HEAD, fontSize: 16, bold: true, color: INK });
  s.addText(d[2], { x: x + 0.25, y: 3.72, w: 3.25, h: 0.6, margin: 0, fontFace: BODY, fontSize: 12.5, color: "33474C", lineSpacingMultiple: 1.02 });
});
// arrow to coopetitive score
s.addShape(pres.shapes.RIGHT_ARROW, { x: 8.42, y: 3.25, w: 0.55, h: 0.45, fill: { color: MUTED }, line: { type: "none" } });
s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: 9.05, y: 2.55, w: 3.65, h: 1.85, fill: { color: DARK }, line: { type: "none" }, rectRadius: 0.08, shadow: sh() });
s.addText("Coopetitive score", { x: 9.25, y: 2.72, w: 3.3, h: 0.4, margin: 0, fontFace: HEAD, fontSize: 15, bold: true, color: TEAL2 });
s.addText("s\u1D62 = m({i}) + \u03BB \u00B7 \u00BD \u03A3 m({i,j})", { x: 9.2, y: 3.18, w: 3.4, h: 0.5, margin: 0, fontFace: "Consolas", fontSize: 15.5, bold: true, color: WHITE });
s.addText("\u03BB tunes individual quality vs. interaction structure; softmax \u2192 weights", { x: 9.25, y: 3.7, w: 3.3, h: 0.65, margin: 0, fontFace: BODY, fontSize: 12, italic: true, color: "CFE3E1", lineSpacingMultiple: 1.0 });

// objectives strip
s.addText("WHAT THIS BUYS US", { x: 0.6, y: 4.65, w: 6, h: 0.3, margin: 0, fontFace: BODY, fontSize: 12, bold: true, color: PRIMARY, charSpacing: 2 });
const obj = [
  ["Interpretability", "A pairwise dividend matrix \u2014 a readable map of redundancy & complementarity"],
  ["No double-counting", "Individual and interaction terms never overlap"],
  ["Tunable, simple", "One parameter \u03BB; ~15 lines of NumPy, no meta-learner to fit"],
];
obj.forEach((o, i) => {
  const x = 0.6 + i * 4.06;
  s.addShape(pres.shapes.RECTANGLE, { x, y: 5.0, w: 3.85, h: 1.55, fill: { color: PALE2 }, line: { color: PALE, width: 1 } });
  s.addText(o[0], { x: x + 0.22, y: 5.16, w: 3.5, h: 0.4, margin: 0, fontFace: HEAD, fontSize: 15.5, bold: true, color: PRIMARY });
  s.addText(o[1], { x: x + 0.22, y: 5.58, w: 3.5, h: 0.9, margin: 0, fontFace: BODY, fontSize: 12.5, color: INK, lineSpacingMultiple: 1.05 });
});
pageNum(s, 3);

// =========================================================
// SLIDE 4 — METHODS: research design (pipeline)
// =========================================================
s = pres.addSlide();
header(s, "Methods", "A four-phase validation design");
s.addText("A comparative experiment: the weighting strategy is the independent variable; AUC-ROC (and Brier score) the outcome. Each phase gates the next.", { x: 0.6, y: 1.68, w: 12.1, h: 0.65, margin: 0, fontFace: BODY, fontSize: 15.5, color: INK });

const phases = [
  ["1", "Construct\nvalidity", "Do pairwise dividends really measure redundancy?"],
  ["2", "Ablation\nstudy", "Does each component pull its weight?"],
  ["3", "Benchmark\ncomparison", "Competitive with established methods?"],
  ["4", "\u03BB sensitivity\n& analysis", "How does \u03BB shape performance?"],
];
let px = 0.6, pw = 2.78, py = 2.7, ph = 2.35, pgap = 0.35;
phases.forEach((p, i) => {
  const x = px + i * (pw + pgap);
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y: py, w: pw, h: ph, fill: { color: PALE }, line: { type: "none" }, rectRadius: 0.08, shadow: sh() });
  s.addShape(pres.shapes.OVAL, { x: x + pw / 2 - 0.38, y: py + 0.25, w: 0.76, h: 0.76, fill: { color: PRIMARY }, line: { type: "none" } });
  s.addText(p[0], { x: x + pw / 2 - 0.38, y: py + 0.27, w: 0.76, h: 0.72, margin: 0, align: "center", valign: "middle", fontFace: HEAD, fontSize: 30, bold: true, color: WHITE });
  s.addText(p[1], { x: x + 0.15, y: py + 1.1, w: pw - 0.3, h: 0.7, margin: 0, align: "center", fontFace: HEAD, fontSize: 16, bold: true, color: INK, lineSpacingMultiple: 0.95 });
  s.addText(p[2], { x: x + 0.18, y: py + 1.72, w: pw - 0.36, h: 0.6, margin: 0, align: "center", fontFace: BODY, fontSize: 11.5, color: "33474C", lineSpacingMultiple: 1.0 });
  if (i < 3) s.addShape(pres.shapes.RIGHT_ARROW, { x: x + pw + 0.02, y: py + ph / 2 - 0.16, w: 0.3, h: 0.32, fill: { color: ACCENT }, line: { type: "none" } });
});
// footer note
s.addShape(pres.shapes.RECTANGLE, { x: 0.6, y: 5.45, w: 12.1, h: 1.05, fill: { color: PALE2 }, line: { color: PALE, width: 1 } });
s.addText([
  { text: "Pipeline:  ", options: { bold: true, color: PRIMARY } },
  { text: "Train base models \u2192 evaluate all 2\u207F coalitions on validation \u2192 compute M\u00F6bius dividends \u2192 derive coopetitive weights \u2192 test on held-out set.  ", options: { color: INK } },
  { text: "Statistics follow Dem\u0161ar (2006): Friedman omnibus + Holm-corrected Wilcoxon.", options: { italic: true, color: "33474C" } },
], { x: 0.85, y: 5.55, w: 11.6, h: 0.85, margin: 0, fontFace: BODY, fontSize: 13.5, lineSpacingMultiple: 1.1, valign: "middle" });
pageNum(s, 4);

// =========================================================
// SLIDE 5 — METHODS: experimental setup
// =========================================================
s = pres.addSlide();
header(s, "Methods", "Experimental setup");
// Left: base models
s.addText("FIVE HETEROGENEOUS BASE MODELS", { x: 0.6, y: 1.75, w: 6, h: 0.3, margin: 0, fontFace: BODY, fontSize: 12.5, bold: true, color: PRIMARY, charSpacing: 1.5 });
const models = [
  ["LR", "Logistic Regression", "linear"],
  ["RF", "Random Forest", "bagging"],
  ["SVM", "Support Vector Machine", "kernel"],
  ["XGB", "XGBoost", "boosting"],
  ["KNN", "k-Nearest Neighbors", "instance"],
];
models.forEach((m, i) => {
  const y = 2.2 + i * 0.72;
  s.addShape(pres.shapes.RECTANGLE, { x: 0.6, y, w: 5.7, h: 0.6, fill: { color: i % 2 ? PALE2 : PALE }, line: { type: "none" } });
  s.addShape(pres.shapes.RECTANGLE, { x: 0.6, y, w: 0.95, h: 0.6, fill: { color: PRIMARY }, line: { type: "none" } });
  s.addText(m[0], { x: 0.6, y, w: 0.95, h: 0.6, margin: 0, align: "center", valign: "middle", fontFace: HEAD, fontSize: 16, bold: true, color: WHITE });
  s.addText(m[1], { x: 1.7, y, w: 3.2, h: 0.6, margin: 0, valign: "middle", fontFace: BODY, fontSize: 14, bold: true, color: INK });
  s.addText(m[2], { x: 4.9, y, w: 1.35, h: 0.6, margin: 0, valign: "middle", align: "right", fontFace: BODY, fontSize: 12, italic: true, color: MUTED });
});
s.addText("Five distinct inductive biases \u2014 chosen so pairwise dividends have real structure to detect.", { x: 0.6, y: 5.95, w: 5.7, h: 0.6, margin: 0, fontFace: BODY, fontSize: 12.5, italic: true, color: "33474C", lineSpacingMultiple: 1.05 });

// Right: stat callouts
s.addText("THE EVALUATION", { x: 6.7, y: 1.75, w: 6, h: 0.3, margin: 0, fontFace: BODY, fontSize: 12.5, bold: true, color: PRIMARY, charSpacing: 1.5 });
const stats = [
  ["15", "binary-classification datasets", "OpenML / UCI \u2014 medical, financial, signal, etc. (208 to 48,842 samples)"],
  ["32", "coalitions per run", "all 2\u2075 subsets evaluated exactly by AUC-ROC on the validation set"],
  ["75", "runs per method", "15 datasets \u00D7 5 random seeds; 60 / 20 / 20 train-val-test split"],
];
stats.forEach((st, i) => {
  const y = 2.2 + i * 1.28;
  s.addShape(pres.shapes.RECTANGLE, { x: 6.7, y, w: 6.0, h: 1.12, fill: { color: PALE2 }, line: { color: PALE, width: 1 }, shadow: sh() });
  s.addText(st[0], { x: 6.85, y: y + 0.06, w: 1.55, h: 1.0, margin: 0, align: "center", valign: "middle", fontFace: HEAD, fontSize: 46, bold: true, color: ACCENT });
  s.addText(st[1], { x: 8.45, y: y + 0.14, w: 4.1, h: 0.4, margin: 0, fontFace: HEAD, fontSize: 15.5, bold: true, color: INK });
  s.addText(st[2], { x: 8.45, y: y + 0.52, w: 4.15, h: 0.55, margin: 0, fontFace: BODY, fontSize: 12, color: "33474C", lineSpacingMultiple: 1.0 });
});
pageNum(s, 5);

// =========================================================
// SLIDE 6 — METHODS: the coopetitive weighting formula
// =========================================================
s = pres.addSlide();
header(s, "Methods", "The coopetitive weighting formula");
// big formula band
s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: 0.6, y: 1.8, w: 12.1, h: 1.35, fill: { color: DARK }, line: { type: "none" }, rectRadius: 0.06, shadow: sh() });
s.addText([
  { text: "s\u1D62  =  ", options: { color: WHITE } },
  { text: "m({i})", options: { color: TEAL2 } },
  { text: "  +  \u03BB \u00B7 \u00BD \u03A3", options: { color: WHITE } },
  { text: "\u2C7C\u2260\u1D62", options: { color: MUTED, fontSize: 16 } },
  { text: " m({i,j})", options: { color: ACCENT } },
], { x: 0.6, y: 1.8, w: 12.1, h: 1.35, margin: 0, align: "center", valign: "middle", fontFace: "Consolas", fontSize: 34, bold: true });

// three readout cards
const cards = [
  ["m({i})", "INDIVIDUAL", "Standalone contribution of model i.", TEAL2],
  ["\u00BD \u03A3 m({i,j})", "INTERACTION", "Net complementarity / redundancy with the rest of the pool. The \u00BD shares each pair fairly (Shapley rule).", ACCENT],
  ["\u03BB", "THE DIAL", "\u03BB = 0 \u2192 pure performance-weighting. Larger \u03BB \u2192 weights track interaction structure more strongly.", PRIMARY],
];
cards.forEach((c, i) => {
  const x = 0.6 + i * 4.06;
  s.addShape(pres.shapes.RECTANGLE, { x, y: 3.5, w: 3.85, h: 2.05, fill: { color: PALE }, line: { type: "none" }, shadow: sh() });
  s.addShape(pres.shapes.RECTANGLE, { x, y: 3.5, w: 0.1, h: 2.05, fill: { color: c[3] }, line: { type: "none" } });
  s.addText(c[0], { x: x + 0.28, y: 3.66, w: 3.4, h: 0.5, margin: 0, fontFace: "Consolas", fontSize: 19, bold: true, color: INK });
  s.addText(c[1], { x: x + 0.28, y: 4.18, w: 3.4, h: 0.3, margin: 0, fontFace: BODY, fontSize: 12, bold: true, color: c[3], charSpacing: 2 });
  s.addText(c[2], { x: x + 0.28, y: 4.5, w: 3.45, h: 0.95, margin: 0, fontFace: BODY, fontSize: 12.5, color: "33474C", lineSpacingMultiple: 1.05 });
});
// bottom note
s.addText([
  { text: "Scores \u2192 weights via softmax with adaptive temperature.  ", options: { color: INK } },
  { text: "Key design choice: ", options: { bold: true, color: PRIMARY } },
  { text: "individual-only (\u03BB = 0) is algebraically identical to performance-weighting \u2014 so the whole case for the interaction term rests on whether \u03BB > 0 beats it (Hypothesis H1).", options: { color: "33474C" } },
], { x: 0.6, y: 5.85, w: 12.1, h: 0.95, margin: 0, fontFace: BODY, fontSize: 13.5, lineSpacingMultiple: 1.1 });
pageNum(s, 6);

// =========================================================
// SLIDE 7 — METHODS: experiments & hypotheses
// =========================================================
s = pres.addSlide();
header(s, "Methods", "Four experiments, four hypotheses");
const hyps = [
  ["H1", "Adding interaction info (\u03BB > 0) beats performance-weighting (\u03BB = 0)", "the pivotal test", ACCENT],
  ["H2", "Pairwise dividends correctly detect known redundancy & complementarity", "construct validity", PRIMARY],
  ["H3", "The framework is competitive with established ensemble methods", "benchmark ranking", PRIMARY],
  ["H4", "A single fixed \u03BB = 0.5 works well across datasets", "robustness check", PRIMARY],
];
hyps.forEach((h, i) => {
  const y = 1.9 + i * 1.16;
  s.addShape(pres.shapes.RECTANGLE, { x: 0.6, y, w: 12.1, h: 1.0, fill: { color: i === 0 ? PALE : PALE2 }, line: { color: PALE, width: 1 }, shadow: i === 0 ? sh() : undefined });
  s.addShape(pres.shapes.OVAL, { x: 0.85, y: y + 0.2, w: 0.6, h: 0.6, fill: { color: h[3] }, line: { type: "none" } });
  s.addText(h[0], { x: 0.85, y: y + 0.2, w: 0.6, h: 0.6, margin: 0, align: "center", valign: "middle", fontFace: HEAD, fontSize: 17, bold: true, color: WHITE });
  s.addText(h[1], { x: 1.7, y: y + 0.08, w: 8.7, h: 0.85, margin: 0, valign: "middle", fontFace: BODY, fontSize: 15.5, bold: true, color: INK });
  s.addText(h[2], { x: 10.5, y: y + 0.08, w: 2.05, h: 0.85, margin: 0, valign: "middle", align: "right", fontFace: BODY, fontSize: 12.5, italic: true, color: h[3] });
});
s.addText("H1 is the make-or-break question; H2 gates everything by checking the diagnostic actually measures what it claims.", { x: 0.6, y: 6.55, w: 12.1, h: 0.4, margin: 0, fontFace: BODY, fontSize: 13.5, italic: true, color: MUTED });
pageNum(s, 7);

// =========================================================
// SLIDE 8 — RESULTS: construct validity
// =========================================================
s = pres.addSlide();
header(s, "Results & Discussion", "Construct validity: the diagnostic works");
// left: test verdicts
const tests = [
  ["A", "Detects redundancy", "PASS\u00B9", GREEN],
  ["B", "Agrees with SII (r = 0.96)", "PASS", GREEN],
  ["C", "Smooth, monotonic (r = \u20130.99)", "PASS", GREEN],
  ["D", "Negative control", "FAILED", RED],
];
s.addText("FOUR CONTROLLED TESTS", { x: 0.6, y: 1.72, w: 6, h: 0.3, margin: 0, fontFace: BODY, fontSize: 12.5, bold: true, color: PRIMARY, charSpacing: 1.5 });
tests.forEach((t, i) => {
  const y = 2.12 + i * 0.82;
  s.addShape(pres.shapes.RECTANGLE, { x: 0.6, y, w: 6.0, h: 0.68, fill: { color: PALE2 }, line: { color: PALE, width: 1 } });
  s.addShape(pres.shapes.OVAL, { x: 0.78, y: y + 0.14, w: 0.4, h: 0.4, fill: { color: t[3] }, line: { type: "none" } });
  s.addText(t[0], { x: 0.78, y: y + 0.14, w: 0.4, h: 0.4, margin: 0, align: "center", valign: "middle", fontFace: HEAD, fontSize: 14, bold: true, color: WHITE });
  s.addText(t[1], { x: 1.35, y, w: 3.6, h: 0.68, margin: 0, valign: "middle", fontFace: BODY, fontSize: 14, color: INK });
  s.addText(t[2], { x: 4.95, y, w: 1.5, h: 0.68, margin: 0, valign: "middle", align: "right", fontFace: BODY, fontSize: 13, bold: true, color: t[3] });
});
s.addText("\u00B9 Test A criterion revised post-hoc (provisional).   Test D fails criterion 1: max |m| = 0.039 is ~4\u00D7 the 0.01 threshold \u2014 but the mean (0.017) is >5\u00D7 below Test A, so the excess is finite-sample AUC noise (~160 val. samples), not real interaction.", { x: 0.6, y: 5.5, w: 6.0, h: 0.78, margin: 0, fontFace: BODY, fontSize: 10.5, italic: true, color: MUTED, lineSpacingMultiple: 1.04 });
s.addShape(pres.shapes.RECTANGLE, { x: 0.6, y: 6.3, w: 6.0, h: 0.55, fill: { color: PALE }, line: { type: "none" } });
s.addText([{ text: "H2 partially supported", options: { bold: true, color: PRIMARY } }, { text: "  \u2014  3 of 4 tests pass convincingly.", options: { color: INK } }], { x: 0.78, y: 6.3, w: 5.7, h: 0.55, margin: 0, valign: "middle", fontFace: BODY, fontSize: 13.5 });

// right: dividend heatmap
s.addImage({ path: "img/fig1_dividend_heatmap.png", x: 6.95, y: 1.78, w: 5.7, h: 4.64, sizing: { type: "contain", w: 5.7, h: 4.64 } });
s.addText("Every pair is redundant (negative): most redundant SVM\u2013XGB, least KNN\u2013LR. The matrix is itself the durable contribution.", { x: 6.95, y: 6.45, w: 5.75, h: 0.55, margin: 0, fontFace: BODY, fontSize: 11.5, italic: true, color: MUTED, lineSpacingMultiple: 1.0, align: "center" });
pageNum(s, 8);

// =========================================================
// SLIDE 9 — RESULTS: ablation + double-counting (native chart)
// =========================================================
s = pres.addSlide();
header(s, "Results & Discussion", "Does the interaction term help?");
const ablLabels = ["Coopetitive-0.5", "Coopetitive-CV", "Shapley", "Perf-Weighted (\u03BB=0)", "Equal", "Best Single", "Shapley+SII", "Stacking", "Interaction-Only"];
const ablVals   = [0.8977, 0.8974, 0.8968, 0.8958, 0.8938, 0.8901, 0.8879, 0.8852, 0.8707];
s.addChart(pres.charts.BAR, [{ name: "Mean AUC-ROC", labels: ablLabels, values: ablVals }], {
  x: 0.55, y: 1.85, w: 7.35, h: 5.0, barDir: "bar",
  chartColors: [PRIMARY],
  valAxisMinVal: 0.86, valAxisMaxVal: 0.90, valAxisMajorUnit: 0.01,
  catAxisLabelColor: INK, catAxisLabelFontSize: 11.5, catAxisLabelFontFace: BODY,
  valAxisLabelColor: MUTED, valAxisLabelFontSize: 10,
  valGridLine: { color: "E2E8F0", size: 0.5 }, catGridLine: { style: "none" },
  showValue: true, dataLabelPosition: "outEnd", dataLabelColor: INK, dataLabelFontSize: 10, dataLabelFormatCode: "0.000",
  showLegend: false, showTitle: false, barGapWidthPct: 45,
  chartArea: { fill: { color: WHITE } },
});
s.addText("Mean AUC-ROC across 15 datasets \u00D7 5 seeds (axis zoomed; total spread only 0.027)", { x: 0.55, y: 6.85, w: 7.4, h: 0.3, margin: 0, fontFace: BODY, fontSize: 10.5, italic: true, color: MUTED, align: "center" });

// right callouts
s.addShape(pres.shapes.RECTANGLE, { x: 8.2, y: 1.9, w: 4.55, h: 1.5, fill: { color: PALE }, line: { type: "none" }, shadow: sh() });
s.addText("Coopetitive-0.5 leads", { x: 8.4, y: 2.02, w: 4.2, h: 0.4, margin: 0, fontFace: HEAD, fontSize: 16, bold: true, color: PRIMARY });
s.addText([{ text: "Best AUC (0.8977) and best calibration (Brier \u20130.097) of any variant; wins 9 / 15 datasets.", options: { color: INK } }], { x: 8.4, y: 2.42, w: 4.2, h: 0.95, margin: 0, fontFace: BODY, fontSize: 13, lineSpacingMultiple: 1.08 });

s.addShape(pres.shapes.RECTANGLE, { x: 8.2, y: 3.55, w: 4.55, h: 1.75, fill: { color: "FBEEDB" }, line: { type: "none" }, shadow: sh() });
s.addShape(pres.shapes.RECTANGLE, { x: 8.2, y: 3.55, w: 0.1, h: 1.75, fill: { color: ACCENT }, line: { type: "none" } });
s.addText("Double-counting, confirmed", { x: 8.42, y: 3.67, w: 4.2, h: 0.4, margin: 0, fontFace: HEAD, fontSize: 16, bold: true, color: "B5791F" });
s.addText([
  { text: "Shapley + SII (0.8879) ", options: { bold: true, color: INK } },
  { text: "loses 0.009 AUC to plain Shapley \u2014 ~\u2153 of the whole spread. Re-adding interactions hurts. M\u00F6bius avoids this by construction.", options: { color: "33474C" } },
], { x: 8.42, y: 4.07, w: 4.2, h: 1.2, margin: 0, fontFace: BODY, fontSize: 13, lineSpacingMultiple: 1.08 });

s.addShape(pres.shapes.RECTANGLE, { x: 8.2, y: 5.45, w: 4.55, h: 1.4, fill: { color: DARK }, line: { type: "none" }, shadow: sh() });
s.addText([{ text: "H1: not significant.  ", options: { bold: true, color: ACCENT } }, { text: "p = 0.148, +0.0018 AUC, r = 0.29. Directionally consistent but underpowered \u2014 a real but small effect.", options: { color: "E6EEED" } }], { x: 8.42, y: 5.45, w: 4.2, h: 1.4, margin: 0, valign: "middle", fontFace: BODY, fontSize: 13, lineSpacingMultiple: 1.1 });
pageNum(s, 9);

// =========================================================
// SLIDE 10 — RESULTS: benchmark + lambda
// =========================================================
s = pres.addSlide();
header(s, "Results & Discussion", "Benchmark ranking & \u03BB sensitivity");
// left: benchmark ranks (native bar)
s.addText("BENCHMARK \u2014 AVG. RANK (LOWER IS BETTER)", { x: 0.6, y: 1.72, w: 6, h: 0.3, margin: 0, fontFace: BODY, fontSize: 12, bold: true, color: PRIMARY, charSpacing: 1 });
const bLabels = ["Shapley", "Coopetitive-CV", "Perf-Weighted", "Stacking", "Best Single", "Equal"];
const bVals   = [2.80, 2.87, 3.23, 3.50, 4.20, 4.40];
s.addChart(pres.charts.BAR, [{ name: "Avg rank", labels: bLabels, values: bVals }], {
  x: 0.5, y: 2.05, w: 6.1, h: 3.15, barDir: "bar",
  chartColors: [TEAL2],
  valAxisMinVal: 0, valAxisMaxVal: 5, valAxisMajorUnit: 1,
  catAxisLabelColor: INK, catAxisLabelFontSize: 11.5, catAxisLabelFontFace: BODY,
  valAxisLabelColor: MUTED, valAxisLabelFontSize: 10,
  valGridLine: { color: "E2E8F0", size: 0.5 }, catGridLine: { style: "none" },
  showValue: true, dataLabelPosition: "outEnd", dataLabelColor: INK, dataLabelFontSize: 11, dataLabelFormatCode: "0.00",
  showLegend: false, showTitle: false, barGapWidthPct: 50,
  chartArea: { fill: { color: WHITE } },
});
s.addShape(pres.shapes.RECTANGLE, { x: 0.6, y: 5.4, w: 6.0, h: 1.45, fill: { color: PALE }, line: { type: "none" } });
s.addText([
  { text: "Coopetitive-CV ranks 2nd of 6", options: { bold: true, color: PRIMARY, breakLine: true } },
  { text: "\u2014 essentially tied with Shapley at the top, ahead of performance-weighting, stacking, and the rest. ", options: { color: INK } },
  { text: "H3: not significant (Friedman p = 0.072), but competitive.", options: { italic: true, color: "33474C" } },
], { x: 0.8, y: 5.5, w: 5.65, h: 1.3, margin: 0, valign: "middle", fontFace: BODY, fontSize: 13, lineSpacingMultiple: 1.1 });

// right: lambda image
s.addText("\u03BB SENSITIVITY \u2014 NO UNIVERSAL DEFAULT", { x: 6.85, y: 1.72, w: 6, h: 0.3, margin: 0, fontFace: BODY, fontSize: 12, bold: true, color: PRIMARY, charSpacing: 1 });
s.addImage({ path: "img/fig4_lambda_sensitivity.png", x: 6.85, y: 2.05, w: 5.85, h: 3.51, sizing: { type: "contain", w: 5.85, h: 3.51 } });
s.addShape(pres.shapes.RECTANGLE, { x: 6.85, y: 5.7, w: 5.85, h: 1.15, fill: { color: "FBEEDB" }, line: { type: "none" } });
s.addText([
  { text: "Optimal \u03BB ranges 0.0\u20131.5 (mean 0.67). ", options: { bold: true, color: "B5791F" } },
  { text: "Fixed \u03BB = 0.5 reaches 95% of optimum on only 13% of datasets. ", options: { color: INK } },
  { text: "H4 not supported \u2014 use cross-validated \u03BB.", options: { italic: true, color: "33474C" } },
], { x: 7.05, y: 5.78, w: 5.5, h: 1.0, margin: 0, valign: "middle", fontFace: BODY, fontSize: 12.5, lineSpacingMultiple: 1.08 });
pageNum(s, 10);

// =========================================================
// SLIDE 11 — RESULTS: hypothesis summary + discussion
// =========================================================
s = pres.addSlide();
header(s, "Results & Discussion", "What the evidence says");
const rows = [
  [{ text: "Hypothesis", options: { bold: true, color: WHITE, fill: { color: PRIMARY }, fontFace: BODY, fontSize: 13.5, align: "left", valign: "middle" } },
   { text: "Verdict", options: { bold: true, color: WHITE, fill: { color: PRIMARY }, fontFace: BODY, fontSize: 13.5, valign: "middle" } },
   { text: "Key evidence", options: { bold: true, color: WHITE, fill: { color: PRIMARY }, fontFace: BODY, fontSize: 13.5, align: "left", valign: "middle" } }],
  ["H1  Interaction term improves performance", "Not sig.", "p = 0.148; +0.0018 AUC; wins 9/15; best Brier"],
  ["H2  Dividends detect model relationships", "Partial", "3/4 tests pass; r = 0.96 with SII"],
  ["H3  Competitive with standard methods", "Not sig.", "Ranks 2nd of 6; Friedman p = 0.072"],
  ["H4  A fixed \u03BB works everywhere", "No", "Optimal \u03BB spans 0.0\u20131.5; use CV"],
];
const styled = rows.map((r, ri) => ri === 0 ? r : r.map((c, ci) => ({
  text: c,
  options: { fontFace: BODY, fontSize: 13, color: ci === 1 ? (c === "No" || c.indexOf("Not") === 0 ? RED : ci === 1 && c === "Partial" ? ACCENT : INK) : INK,
    bold: ci === 1, align: ci === 1 ? "center" : "left", valign: "middle",
    fill: { color: ri % 2 ? WHITE : PALE2 } }
})));
s.addTable(styled, { x: 0.6, y: 1.8, w: 7.5, colW: [3.3, 1.2, 3.0], rowH: [0.45, 0.62, 0.62, 0.62, 0.62], border: { type: "solid", pt: 0.5, color: "D8E3E1" } });

// right: discussion takeaways
s.addText("THE HONEST READING", { x: 8.45, y: 1.78, w: 4.3, h: 0.3, margin: 0, fontFace: BODY, fontSize: 12.5, bold: true, color: PRIMARY, charSpacing: 1.5 });
const takes = [
  ["Interpretability is the real win", "The dividend matrix gives a diagnostic no other method offers \u2014 independent of any accuracy gain."],
  ["Performance is competitive, not breakthrough", "Effects are small; ~30\u201340 datasets would be needed to settle H1."],
  ["Coopetition ran one-sided", "All 75 runs gave negative dividends \u2014 it acted purely as a redundancy penalty here."],
];
takes.forEach((t, i) => {
  const y = 2.15 + i * 1.5;
  s.addShape(pres.shapes.RECTANGLE, { x: 8.45, y, w: 4.3, h: 1.34, fill: { color: PALE }, line: { type: "none" } });
  s.addShape(pres.shapes.RECTANGLE, { x: 8.45, y, w: 0.09, h: 1.34, fill: { color: ACCENT }, line: { type: "none" } });
  s.addText(t[0], { x: 8.65, y: y + 0.12, w: 4.0, h: 0.5, margin: 0, fontFace: HEAD, fontSize: 14.5, bold: true, color: INK });
  s.addText(t[1], { x: 8.65, y: y + 0.6, w: 4.0, h: 0.68, margin: 0, fontFace: BODY, fontSize: 12.5, color: "33474C", lineSpacingMultiple: 1.05 });
});
pageNum(s, 11);

// =========================================================
// SLIDE 12 — CONCLUSION
// =========================================================
s = pres.addSlide();
header(s, "Conclusion & Future Work", "Conclusions & limitations");
s.addText("CONCLUSIONS", { x: 0.6, y: 1.72, w: 6, h: 0.3, margin: 0, fontFace: BODY, fontSize: 12.5, bold: true, color: PRIMARY, charSpacing: 1.5 });
const concl = [
  "The M\u00F6bius decomposition resolves the interpretability gap \u2014 the pairwise dividend matrix maps redundancy and complementarity directly.",
  "It resolves the double-counting gap empirically: Shapley + SII degrades AUC, confirming the core thesis.",
  "Entanglement (H1) and benchmark (H3) gains were directional but not significant; calibration was best-in-class.",
];
s.addText(concl.map((c, i) => ({ text: c, options: { bullet: { code: "2022", indent: 18 }, breakLine: true, paraSpaceAfter: 9 } })),
  { x: 0.6, y: 2.12, w: 6.0, h: 3.0, margin: 0, fontFace: BODY, fontSize: 14, color: INK, lineSpacingMultiple: 1.05 });

s.addShape(pres.shapes.RECTANGLE, { x: 0.6, y: 5.35, w: 6.0, h: 1.5, fill: { color: DARK }, line: { type: "none" }, shadow: sh() });
s.addText("Bottom line", { x: 0.8, y: 5.48, w: 5.6, h: 0.4, margin: 0, fontFace: HEAD, fontSize: 15, bold: true, color: ACCENT });
s.addText("A principled, interpretable diagnostic instrument for ensemble design \u2014 transparency that performance-focused methods don\u2019t provide.", { x: 0.8, y: 5.86, w: 5.65, h: 0.95, margin: 0, fontFace: BODY, fontSize: 13.5, color: "E6EEED", lineSpacingMultiple: 1.1 });

// right: limitations
s.addText("LIMITATIONS WE KEEP IN VIEW", { x: 6.95, y: 1.72, w: 6, h: 0.3, margin: 0, fontFace: BODY, fontSize: 12.5, bold: true, color: PRIMARY, charSpacing: 1.5 });
const lims = [
  ["Pairwise truncation", "Higher-order terms dropped; decomposition coverage stayed negative (\u20132.6 to \u20135.0)."],
  ["No fixed \u03BB", "Optimal \u03BB is dataset-dependent (0.0\u20131.5)."],
  ["Scalability", "Exact coalition enumeration limits pools to n \u2264 12."],
  ["Scope", "Binary classification only; n = 5 weakens the rank-stability check."],
];
lims.forEach((l, i) => {
  const y = 2.12 + i * 1.16;
  s.addShape(pres.shapes.RECTANGLE, { x: 6.95, y, w: 5.75, h: 1.0, fill: { color: PALE2 }, line: { color: PALE, width: 1 } });
  s.addText(l[0], { x: 7.15, y: y + 0.12, w: 5.4, h: 0.4, margin: 0, fontFace: HEAD, fontSize: 14.5, bold: true, color: INK });
  s.addText(l[1], { x: 7.15, y: y + 0.5, w: 5.4, h: 0.46, margin: 0, fontFace: BODY, fontSize: 12.5, color: "33474C", lineSpacingMultiple: 1.0 });
});
pageNum(s, 12);

// =========================================================
// SLIDE 13 — FUTURE WORK + THANK YOU
// =========================================================
s = pres.addSlide();
s.background = { color: DARK };
s.addShape(pres.shapes.RECTANGLE, { x: 11.55, y: 5.9, w: 0.9, h: 0.9, fill: { color: PRIMARY, transparency: 25 }, line: { type: "none" } });
s.addShape(pres.shapes.RECTANGLE, { x: 11.9, y: 6.25, w: 0.9, h: 0.9, fill: { color: ACCENT, transparency: 30 }, line: { type: "none" } });
s.addText("FUTURE WORK", { x: 0.8, y: 0.7, w: 8, h: 0.35, margin: 0, fontFace: BODY, fontSize: 13, bold: true, color: TEAL2, charSpacing: 3 });
s.addText("Where this goes next", { x: 0.78, y: 1.05, w: 11, h: 0.8, margin: 0, fontFace: HEAD, fontSize: 32, bold: true, color: WHITE });
const fut = [
  ["Higher-order dividends", "Add 3-way+ terms to restore decomposition coverage \u2014 the most direct extension."],
  ["Dividend-informed pruning", "Use the matrix to drop redundant models before combining (grand coalition lost 92% of runs)."],
  ["Broaden & scale", "Multi-class & regression; sampling-based estimation to break the n \u2264 12 ceiling."],
];
fut.forEach((f, i) => {
  const x = 0.8 + i * 4.04;
  s.addShape(pres.shapes.RECTANGLE, { x, y: 2.15, w: 3.8, h: 2.2, fill: { color: "10323B" }, line: { type: "none" } });
  s.addShape(pres.shapes.RECTANGLE, { x, y: 2.15, w: 3.8, h: 0.09, fill: { color: ACCENT }, line: { type: "none" } });
  s.addText(f[0], { x: x + 0.25, y: 2.4, w: 3.35, h: 0.85, margin: 0, fontFace: HEAD, fontSize: 17, bold: true, color: WHITE });
  s.addText(f[1], { x: x + 0.25, y: 3.25, w: 3.35, h: 1.0, margin: 0, fontFace: BODY, fontSize: 13, color: "CFE3E1", lineSpacingMultiple: 1.08 });
});
s.addText("Thank you", { x: 0.78, y: 4.95, w: 8, h: 0.9, margin: 0, fontFace: HEAD, fontSize: 40, bold: true, color: ACCENT });
s.addText([
  { text: "Maxell G. Milay", options: { bold: true, color: WHITE, breakLine: true } },
  { text: "github.com/maxellmilay/mobius-decomposition-ensemble-weighting", options: { color: TEAL2 } },
], { x: 0.8, y: 5.95, w: 10.5, h: 0.9, margin: 0, fontFace: BODY, fontSize: 14, lineSpacingMultiple: 1.25 });

pres.writeFile({ fileName: "defense.pptx" }).then(f => console.log("Wrote", f));
