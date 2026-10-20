#!/usr/bin/env python3
"""
Fill the Review 1 PPTX with all project content except student details.

Usage:
    python3 fill_pptx.py [path/to/deck.pptx] [--output out.pptx]

The deck path defaults to the "Review 1" deck in the repository root.

Note: this deck was written for the *proposal* stage of the project. Its
"Expected Results" wording predates the results in the paper and is therefore
optimistic relative to the current (honest) claims. Regenerate the current
numbers with ``scripts/generate_novelty_pptx.py`` before presenting.
"""
import argparse
import copy
from pathlib import Path

from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.enum.text import PP_ALIGN
from pptx.dml.color import RGBColor

ROOT = Path(__file__).resolve().parent
DEFAULT_DECK = ROOT / "Review 1 - 22 07 2026.pptx"


def set_text(shape, text, font_size=None, bold=None, color=None):
    """Set text on a shape, preserving existing formatting for first run."""
    tf = shape.text_frame
    tf.clear()
    p = tf.paragraphs[0]
    p.text = text
    if font_size is not None:
        p.runs[0].font.size = Pt(font_size)
    if bold is not None:
        p.runs[0].font.bold = bold
    if color is not None:
        p.runs[0].font.color.rgb = RGBColor(*color)


def add_paragraph(text_frame, text, font_size=14, bold=False, space_before=Pt(4), space_after=Pt(2), level=0):
    """Add a paragraph to a text frame."""
    p = text_frame.add_paragraph()
    p.text = text
    p.level = level
    p.space_before = space_before
    p.space_after = space_after
    for run in p.runs:
        run.font.size = Pt(font_size)
        run.font.bold = bold
    return p


def add_content_slide(prs, title_text, content_items, footer_text="SCOPE", slide_num=None):
    """Add a new slide with title and bulleted content using 'Title and Content' layout."""
    slide_layout = prs.slide_layouts[1]  # Title and Content
    slide = prs.slides.add_slide(slide_layout)

    # Set title
    title_shape = slide.shapes.title
    if title_shape:
        title_shape.text = title_text
        for para in title_shape.text_frame.paragraphs:
            for run in para.runs:
                run.font.size = Pt(30)
                run.font.bold = True

    # Set content
    content_shape = slide.placeholders[1]  # Content placeholder
    tf = content_shape.text_frame
    tf.clear()

    first = True
    for item in content_items:
        if first:
            p = tf.paragraphs[0]
            first = False
        else:
            p = tf.add_paragraph()

        if isinstance(item, tuple):
            text, level, is_bold = item
        else:
            text, level, is_bold = item, 0, False

        p.text = text
        p.level = level
        p.space_before = Pt(6)
        p.space_after = Pt(3)
        for run in p.runs:
            run.font.size = Pt(13 if level > 0 else 14)
            run.font.bold = is_bold

    # Set footer
    for shape in slide.placeholders:
        if hasattr(shape, 'placeholder_format') and shape.placeholder_format.idx == 2:  # Footer
            shape.text = footer_text

    return slide


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("deck", nargs="?", default=str(DEFAULT_DECK))
    parser.add_argument("--output", default=None, help="Write to a different file")
    args = parser.parse_args()

    deck = Path(args.deck)
    if not deck.is_file():
        raise SystemExit(f"Deck not found: {deck}")
    prs = Presentation(str(deck))

    # =========================================================================
    # SLIDE 1: Fill title (leave student details untouched)
    # =========================================================================
    slide1 = prs.slides[0]
    for shape in slide1.shapes:
        if shape.name == "Title 6":
            # Replace placeholder title
            tf = shape.text_frame
            tf.clear()
            p = tf.paragraphs[0]
            # Programme line
            run1 = p.add_run()
            run1.text = "B.Tech / M.Tech\nCourse Code \u2013 Course Title\n"
            run1.font.size = Pt(20)
            run1.font.bold = True
            # Project title
            run2 = p.add_run()
            run2.text = "Temporal Graph Diffusion Tracing for\nCross-Modal Misinformation Factuality Degradation"
            run2.font.size = Pt(32)
            run2.font.bold = True
    # =========================================================================
    # SLIDE 4: Fill Literature Review content placeholder
    # =========================================================================
    slide4 = prs.slides[3]
    for shape in slide4.shapes:
        if shape.name == "Content Placeholder 4" and shape.has_text_frame:
            tf = shape.text_frame
            tf.clear()
            lit_items = [
                ("Dou et al. (2021) \u2014 UPFD: User Preference-aware Fake News Detection (SIGIR)", 0, True),
                ("Proposed user preference features with GNNs on propagation trees. Introduced UPFD benchmark with Politifact and GossipCop subsets.", 1, False),
                ("Bian et al. (2021) \u2014 BiGCN: Bidirectional Graph Convolutional Networks for Fake News Detection (AAAI)", 0, True),
                ("Applied top-down and bottom-up GCN branches on propagation trees with edge dropout.", 1, False),
                ("Lu & Li (2020) \u2014 GCNFN: Fake News Detection with Graph Convolutional Networks over Fake News Networks (WI-IAT)", 0, True),
                ("GAT layers with SELU activations and mean pooling on fake news networks.", 1, False),
                ("Wu et al. (2023) \u2014 GNN-CL: Fake News Detection via Graph Neural Networks with DiffPool (ACM Web Conf.)", 0, True),
                ("Differentiable pooling on propagation graphs for hierarchical feature aggregation.", 1, False),
                ("DPSG (2024) \u2014 Diffusion-based Propagation Structure Graphing (Information Fusion)", 0, True),
                ("Modelled propagation as continuous diffusion process for structural analysis.", 1, False),
                ("DTN (2024) \u2014 Diffusion Trace Network (PeerJ CS)", 0, True),
                ("Traced diffusion paths for fake news detection across propagation hops.", 1, False),
                ("Devlin et al. (2019) \u2014 BERT: Pre-training of Deep Bidirectional Transformers (NAACL-HLT)", 0, True),
                ("Foundation for our Sentence-BERT text embeddings (all-MiniLM-L6-v2, 384-dim).", 1, False),
                ("Radford et al. (2021) \u2014 CLIP: Learning Transferable Visual Models (ICML)", 0, True),
                ("Cross-modal image-text alignment for Path B multimodal extension (ViT-B/32, 512-dim).", 1, False),
            ]
            first = True
            for text, level, is_bold in lit_items:
                if first:
                    p = tf.paragraphs[0]
                    first = False
                else:
                    p = tf.add_paragraph()
                p.text = text
                p.level = level
                p.space_before = Pt(4)
                p.space_after = Pt(2)
                for run in p.runs:
                    run.font.size = Pt(11 if level > 0 else 12)
                    run.font.bold = is_bold

    # =========================================================================
    # NEW SLIDE: Methodology / System Architecture (insert after Abstract)
    # =========================================================================
    slide_layout = prs.slide_layouts[1]
    slide_method = prs.slides.add_slide(slide_layout)
    # Move slide to position 3 (after Abstract, before Literature Review)
    slide_id = slide_method.slide_id
    xml_slides = prs.slides._sldIdLst
    slide_elements = list(xml_slides)
    # Find the element for our new slide and move it
    for idx, elem in enumerate(slide_elements):
        if elem.get('{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id') is None:
            # Use the rId approach
            pass
    # Reorder: move the last slide to index 3
    sldId = None
    for elem in slide_elements:
        rid = elem.get('{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id')
        if rid:
            # Check if this is our newly added slide
            rel = prs.slides._sldIdLst.getparent()
            break
    # Simpler approach: just manipulate the XML list directly
    new_slide_elem = xml_slides[-1]  # Last added
    xml_slides.remove(new_slide_elem)
    # Insert after index 2 (0-based), i.e., 3rd position
    third_elem = list(xml_slides)[2]
    xml_slides.insert(3, new_slide_elem)

    # Set title
    title_shape = slide_method.shapes.title
    if title_shape:
        title_shape.text = "Methodology"
        for para in title_shape.text_frame.paragraphs:
            for run in para.runs:
                run.font.size = Pt(30)
                run.font.bold = True

    content_shape = slide_method.placeholders[1]
    tf = content_shape.text_frame
    tf.clear()
    method_items = [
        ("Data Layer: UPFD Benchmark (FakeNewsNet)", 0, True),
        ("Tree-structured propagation graphs; root = source news, nodes = user engagements, edges = reshare hops.", 1, False),
        ("Splits: 80/10/10 train/val/test; subsets: Politifact & GossipCop.", 1, False),
        ("Feature Engineering Pipeline", 0, True),
        ("Text Encoder: Sentence-BERT (all-MiniLM-L6-v2) \u2192 384-dim embeddings per node.", 1, False),
        ("Image Encoder (Path B): CLIP ViT-B/32 \u2192 512-dim embeddings per node.", 1, False),
        ("Cross-Modal Alignment: cosine similarity between text & image embeddings per node.", 1, False),
        ("Drift Features: \u0394text (cosine dist), \u0394L2, hop depth, branching factor, is_root \u2192 5-dim drift vector.", 1, False),
        ("Model Architecture: Multi-Task Learning", 0, True),
        ("Primary task: binary veracity classification (fake vs. real).", 1, False),
        ("Auxiliary task: per-hop drift prediction (MSE loss) for explainability.", 1, False),
        ("Training: AdamW (lr=1e-3), BCE + MSE loss, early stopping (patience=10), 100 epochs.", 1, False),
    ]
    first = True
    for text, level, is_bold in method_items:
        p = tf.paragraphs[0] if first else tf.add_paragraph()
        first = False
        p.text = text
        p.level = level
        p.space_before = Pt(5)
        p.space_after = Pt(2)
        for run in p.runs:
            run.font.size = Pt(13 if level > 0 else 14)
            run.font.bold = is_bold
    for shape in slide_method.placeholders:
        if hasattr(shape, 'placeholder_format') and shape.placeholder_format.idx == 2:
            shape.text = "SCOPE"

    # =========================================================================
    # NEW SLIDE: Models & Feature Engineering (after Methodology)
    # =========================================================================
    slide_models = prs.slides.add_slide(slide_layout)
    # Move to position 4
    xml_slides = prs.slides._sldIdLst
    all_elems = list(xml_slides)
    new_elem = all_elems[-1]
    xml_slides.remove(new_elem)
    fourth_elem = list(xml_slides)[3]
    xml_slides.insert(4, new_elem)

    title_shape = slide_models.shapes.title
    if title_shape:
        title_shape.text = "Models & Feature Engineering"
        for para in title_shape.text_frame.paragraphs:
            for run in para.runs:
                run.font.size = Pt(28)
                run.font.bold = True

    content_shape = slide_models.placeholders[1]
    tf = content_shape.text_frame
    tf.clear()
    model_items = [
        ("Baseline: LSTM over Node Sequence", 0, True),
        ("Input: BFS-ordered node sequence. LSTM (2-layer, bidirectional, hidden=128) + MLP classifier.", 1, False),
        ("Ignores graph structure; treats propagation as ordered sequence. Serves as lower-bound baseline.", 1, False),
        ("Proposed: DegradationGNN (BiGCN-style)", 0, True),
        ("GCN message passing on propagation tree with temporal hop-depth positional encoding.", 1, False),
        ("Drift features (cosine dist, L2, depth, branching) integrated at each layer via auxiliary projection.", 1, False),
        ("Multi-task output: veracity logits + per-hop drift prediction head.", 1, False),
        ("Trajectory Prediction Head (Transformer Decoder)", 0, True),
        ("Given first k hops of observed drift, predicts degradation for hops k+1 to N.", 1, False),
        ("Enables early detection by extrapolating future degradation curves.", 1, False),
        ("Autoregressive rollout for multi-step prediction.", 1, False),
        ("Early Detection Wrapper", 0, True),
        ("Truncates cascade at 10%, 25%, 50%, 75% and evaluates classification performance.", 1, False),
        ("Measures F1 and accuracy at each truncation point to build early detection curve.", 1, False),
    ]
    first = True
    for text, level, is_bold in model_items:
        p = tf.paragraphs[0] if first else tf.add_paragraph()
        first = False
        p.text = text
        p.level = level
        p.space_before = Pt(5)
        p.space_after = Pt(2)
        for run in p.runs:
            run.font.size = Pt(12 if level > 0 else 13)
            run.font.bold = is_bold
    for shape in slide_models.placeholders:
        if hasattr(shape, 'placeholder_format') and shape.placeholder_format.idx == 2:
            shape.text = "SCOPE"

    # =========================================================================
    # NEW SLIDE: Implementation & Code Structure
    # =========================================================================
    slide_impl = prs.slides.add_slide(slide_layout)
    xml_slides = prs.slides._sldIdLst
    all_elems = list(xml_slides)
    new_elem = all_elems[-1]
    xml_slides.remove(new_elem)
    fifth_elem = list(xml_slides)[4]
    xml_slides.insert(5, new_elem)

    title_shape = slide_impl.shapes.title
    if title_shape:
        title_shape.text = "Implementation & Code Structure"
        for para in title_shape.text_frame.paragraphs:
            for run in para.runs:
                run.font.size = Pt(28)
                run.font.bold = True

    content_shape = slide_impl.placeholders[1]
    tf = content_shape.text_frame
    tf.clear()
    impl_items = [
        ("Project Structure (Modular Pipeline)", 0, True),
        ("data/upfd/loader.py: UPFD dataset loading, BFS ordering, hop depth computation, train/val/test splits.", 1, False),
        ("features/text_encoder.py: Sentence-BERT (all-MiniLM-L6-v2) \u2192 384-dim text embeddings.", 1, False),
        ("features/image_encoder.py: CLIP ViT-B/32 \u2192 512-dim image embeddings (Path B).", 1, False),
        ("features/cross_modal.py: Text-image cosine alignment scoring with learnable temperature.", 1, False),
        ("features/drift.py: Per-hop drift computation (cosine dist, L2, depth, branching, is_root).", 1, False),
        ("models/baseline_lstm.py: Bidirectional LSTM baseline with MLP classifier.", 1, False),
        ("models/degradation_gnn.py: BiGCN-style GNN with temporal encoding + drift integration.", 1, False),
        ("models/trajectory_head.py: Transformer decoder for degradation trajectory prediction.", 1, False),
        ("models/early_detector.py: Cascade truncation wrapper for early detection evaluation.", 1, False),
        ("train.py: Config-driven training loop with AdamW, TensorBoard logging, checkpointing.", 1, False),
        ("evaluate.py: Full evaluation (Accuracy, F1, AUC-ROC, AUC-PR) + early detection + baselines.", 1, False),
        ("Tech Stack", 0, True),
        ("PyTorch 2.0+, PyTorch Geometric, Sentence-BERT, CLIP, TensorBoard, scikit-learn.", 1, False),
    ]
    first = True
    for text, level, is_bold in impl_items:
        p = tf.paragraphs[0] if first else tf.add_paragraph()
        first = False
        p.text = text
        p.level = level
        p.space_before = Pt(4)
        p.space_after = Pt(2)
        for run in p.runs:
            run.font.size = Pt(11 if level > 0 else 13)
            run.font.bold = is_bold
    for shape in slide_impl.placeholders:
        if hasattr(shape, 'placeholder_format') and shape.placeholder_format.idx == 2:
            shape.text = "SCOPE"

    # =========================================================================
    # NEW SLIDE: Expected Results & Evaluation (after Objectives)
    # =========================================================================
    slide_results = prs.slides.add_slide(slide_layout)
    xml_slides = prs.slides._sldIdLst
    all_elems = list(xml_slides)
    new_elem = all_elems[-1]
    xml_slides.remove(new_elem)
    # Insert after slide 6 (Objectives) which is at index 6
    target_idx = 7
    if target_idx < len(list(xml_slides)):
        target_elem = list(xml_slides)[target_idx]
        xml_slides.insert(target_idx, new_elem)
    else:
        xml_slides.append(new_elem)

    title_shape = slide_results.shapes.title
    if title_shape:
        title_shape.text = "Expected Results & Evaluation"
        for para in title_shape.text_frame.paragraphs:
            for run in para.runs:
                run.font.size = Pt(28)
                run.font.bold = True

    content_shape = slide_results.placeholders[1]
    tf = content_shape.text_frame
    tf.clear()
    results_items = [
        ("Evaluation Metrics", 0, True),
        ("Classification: Accuracy, Precision, Recall, F1-Score, AUC-ROC, AUC-PR.", 1, False),
        ("Degradation: MSE between predicted and actual drift per hop.", 1, False),
        ("Early Detection: F1 at 10%, 25%, 50%, 75% cascade completion.", 1, False),
        ("Calibration: Expected Calibration Error (ECE).", 1, False),
        ("Baselines for Comparison", 0, True),
        ("GCN, GAT, GIN (static graph methods from UPFD framework).", 1, False),
        ("BiGCN (bidirectional GCN \u2014 state-of-the-art on UPFD).", 1, False),
        ("Feature-only MLP (no graph structure).", 1, False),
        ("Logistic Regression, SVM (sklearn baselines).", 1, False),
        ("Expected Outcomes", 0, True),
        ("DegradationGNN matches or exceeds BiGCN on classification metrics.", 1, False),
        ("Trajectory head provides explainable degradation curves for fake vs. real news.", 1, False),
        ("Early detector achieves >80% F1 at 50% cascade completion.", 1, False),
        ("Cross-modal Path B shows additional gains on multimodal data.", 1, False),
    ]
    first = True
    for text, level, is_bold in results_items:
        p = tf.paragraphs[0] if first else tf.add_paragraph()
        first = False
        p.text = text
        p.level = level
        p.space_before = Pt(5)
        p.space_after = Pt(2)
        for run in p.runs:
            run.font.size = Pt(13 if level > 0 else 14)
            run.font.bold = is_bold
    for shape in slide_results.placeholders:
        if hasattr(shape, 'placeholder_format') and shape.placeholder_format.idx == 2:
            shape.text = "SCOPE"

    # =========================================================================
    # NEW SLIDE: Timeline & Future Work (after Expected Results)
    # =========================================================================
    slide_timeline = prs.slides.add_slide(slide_layout)
    xml_slides = prs.slides._sldIdLst
    all_elems = list(xml_slides)
    new_elem = all_elems[-1]
    xml_slides.remove(new_elem)
    # Insert before References (last position - 1)
    total = len(list(xml_slides))
    if total > 1:
        target_elem = list(xml_slides)[total - 1]
        xml_slides.insert(total - 1, new_elem)
    else:
        xml_slides.append(new_elem)

    title_shape = slide_timeline.shapes.title
    if title_shape:
        title_shape.text = "Timeline & Future Work"
        for para in title_shape.text_frame.paragraphs:
            for run in para.runs:
                run.font.size = Pt(28)
                run.font.bold = True

    content_shape = slide_timeline.placeholders[1]
    tf = content_shape.text_frame
    tf.clear()
    timeline_items = [
        ("Implementation Timeline", 0, True),
        ("Phase 1 (Weeks 1\u20132): Dataset download, preprocessing, UPFD loader implementation.", 1, False),
        ("Phase 2 (Weeks 3\u20134): Feature engineering \u2014 text encoder, drift computation, cross-modal alignment.", 1, False),
        ("Phase 3 (Weeks 5\u20137): Model implementation \u2014 BaselineLSTM, DegradationGNN, TrajectoryHead.", 1, False),
        ("Phase 4 (Week 8): Training pipeline, hyperparameter tuning, TensorBoard integration.", 1, False),
        ("Phase 5 (Weeks 9\u201310): Evaluation, baseline comparison, early detection analysis.", 1, False),
        ("Phase 6 (Weeks 11\u201312): Paper writing, result visualization, final presentation.", 1, False),
        ("Future Work", 0, True),
        ("Twitter API v2 crawler for custom multimodal propagation data (Path B).", 1, False),
        ("Integration of image features (CLIP) for cross-modal misinformation detection.", 1, False),
        ("Temporal attention mechanisms for adaptive hop-weighting.", 1, False),
        ("Deployment as real-time early warning system for social media platforms.", 1, False),
        ("Extension to video-based misinformation and multi-hop reasoning.", 1, False),
    ]
    first = True
    for text, level, is_bold in timeline_items:
        p = tf.paragraphs[0] if first else tf.add_paragraph()
        first = False
        p.text = text
        p.level = level
        p.space_before = Pt(5)
        p.space_after = Pt(2)
        for run in p.runs:
            run.font.size = Pt(13 if level > 0 else 14)
            run.font.bold = is_bold
    for shape in slide_timeline.placeholders:
        if hasattr(shape, 'placeholder_format') and shape.placeholder_format.idx == 2:
            shape.text = "SCOPE"

    # =========================================================================
    # Save
    # =========================================================================
    output_path = Path(args.output) if args.output else deck
    prs.save(str(output_path))
    print(f"Saved to {output_path}")
    print(f"Total slides: {len(prs.slides)}")


if __name__ == "__main__":
    main()
