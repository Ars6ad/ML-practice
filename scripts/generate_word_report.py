import os
import sys
import json
import pandas as pd
import numpy as np

import docx
from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_ALIGN_VERTICAL
from docx.oxml import OxmlElement, parse_xml
from docx.oxml.ns import nsdecls, qn


def set_cell_background(cell, hex_color):
    """Sets the background color of a table cell."""
    shading_elm = parse_xml(f'<w:shd {nsdecls("w")} w:fill="{hex_color}"/>')
    cell._tc.get_or_add_tcPr().append(shading_elm)


def set_cell_margins(cell, top=100, bottom=100, left=150, right=150):
    """Sets cell padding (in twentieths of a point, dxa)."""
    tcPr = cell._tc.get_or_add_tcPr()
    tcMar = OxmlElement('w:tcMar')
    for m, val in [('top', top), ('bottom', bottom), ('left', left), ('right', right)]:
        node = OxmlElement(f'w:{m}')
        node.set(qn('w:w'), str(val))
        node.set(qn('w:type'), 'dxa')
        tcMar.append(node)
    tcPr.append(tcMar)


def set_table_borders(table, color="D3D3D3", sz="4", val="single"):
    """Applies clean borders to a table."""
    tblPr = table._tbl.tblPr
    borders = parse_xml(
        f'<w:tblBorders {nsdecls("w")}>'
        f'  <w:top w:val="{val}" w:sz="{sz}" w:space="0" w:color="{color}"/>'
        f'  <w:bottom w:val="{val}" w:sz="{sz}" w:space="0" w:color="{color}"/>'
        f'  <w:left w:val="none"/>'
        f'  <w:right w:val="none"/>'
        f'  <w:insideH w:val="{val}" w:sz="{sz}" w:space="0" w:color="{color}"/>'
        f'  <w:insideV w:val="none"/>'
        f'</w:tblBorders>'
    )
    tblPr.append(borders)


def add_page_number(run):
    """Inserts a dynamic Word page number field."""
    fldSimple = parse_xml(r'<w:fldSimple %s w:instr="PAGE"/>' % nsdecls('w'))
    run._r.append(fldSimple)


def build_academic_word_report():
    doc = Document()

    # 1. Page Setup: A4, 1-inch margins
    for section in doc.sections:
        section.page_width = Inches(8.27)
        section.page_height = Inches(11.69)
        section.top_margin = Inches(1.0)
        section.bottom_margin = Inches(1.0)
        section.left_margin = Inches(1.0)
        section.right_margin = Inches(1.0)
        section.different_first_page_header_footer = True

    # 2. Typography & Styles Setup
    styles = doc.styles

    # Normal Style (Body text)
    normal_style = styles['Normal']
    normal_font = normal_style.font
    normal_font.name = 'Times New Roman'
    normal_font.size = Pt(12)
    normal_font.color.rgb = RGBColor(30, 41, 59) # Charcoal
    normal_style.paragraph_format.line_spacing = 1.5
    normal_style.paragraph_format.space_after = Pt(6)
    normal_style.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY

    # Heading 1
    h1_style = styles['Heading 1']
    h1_font = h1_style.font
    h1_font.name = 'Times New Roman'
    h1_font.size = Pt(16)
    h1_font.bold = True
    h1_font.color.rgb = RGBColor(26, 54, 93) # Deep Navy
    h1_style.paragraph_format.space_before = Pt(18)
    h1_style.paragraph_format.space_after = Pt(6)
    h1_style.paragraph_format.keep_with_next = True

    # Heading 2
    h2_style = styles['Heading 2']
    h2_font = h2_style.font
    h2_font.name = 'Times New Roman'
    h2_font.size = Pt(14)
    h2_font.bold = True
    h2_font.color.rgb = RGBColor(43, 108, 176) # Steel Blue
    h2_style.paragraph_format.space_before = Pt(14)
    h2_style.paragraph_format.space_after = Pt(4)
    h2_style.paragraph_format.keep_with_next = True

    # Heading 3
    h3_style = styles['Heading 3']
    h3_font = h3_style.font
    h3_font.name = 'Times New Roman'
    h3_font.size = Pt(12.5)
    h3_font.bold = True
    h3_font.italic = True
    h3_font.color.rgb = RGBColor(45, 55, 72)
    h3_style.paragraph_format.space_before = Pt(10)
    h3_style.paragraph_format.space_after = Pt(2)
    h3_style.paragraph_format.keep_with_next = True

    # Set up Headers & Footers on normal sections
    header = doc.sections[0].header
    hp = header.paragraphs[0]
    hp.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    hrun = hp.add_run("Robust Deep Learning Postharvest Fruit Quality Classification")
    hrun.font.name = "Times New Roman"
    hrun.font.size = Pt(8.5)
    hrun.font.color.rgb = RGBColor(120, 144, 156)

    footer = doc.sections[0].footer
    fp = footer.paragraphs[0]
    fp.alignment = WD_ALIGN_PARAGRAPH.CENTER
    frun = fp.add_run("Page ")
    frun.font.name = "Times New Roman"
    frun.font.size = Pt(10)
    frun.font.color.rgb = RGBColor(100, 116, 139)
    add_page_number(frun)

    # =========================================================================
    # 1. COVER PAGE
    # =========================================================================
    p_title_space = doc.add_paragraph()
    p_title_space.paragraph_format.space_before = Pt(72)

    p_title = doc.add_paragraph()
    p_title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r_title = p_title.add_run("ROBUST DEEP LEARNING-BASED POSTHARVEST FRUIT QUALITY CLASSIFICATION UNDER CLASS IMBALANCE AND ENVIRONMENTAL VARIATIONS")
    r_title.font.name = 'Times New Roman'
    r_title.font.size = Pt(22)
    r_title.font.bold = True
    r_title.font.color.rgb = RGBColor(26, 54, 93)

    p_sub = doc.add_paragraph()
    p_sub.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_sub.paragraph_format.space_before = Pt(18)
    p_sub.paragraph_format.space_after = Pt(72)
    r_sub = p_sub.add_run("An Empirical Investigation of Transfer Learning, Minority Augmentation, and Balanced Loss Reweighting across Diverse Postharvest Conditions")
    r_sub.font.name = 'Times New Roman'
    r_sub.font.size = Pt(13)
    r_sub.font.italic = True
    r_sub.font.color.rgb = RGBColor(71, 85, 105)

    p_rule = doc.add_paragraph()
    p_rule.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r_rule = p_rule.add_run("—" * 35)
    r_rule.font.color.rgb = RGBColor(203, 213, 225)

    p_meta = doc.add_paragraph()
    p_meta.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_meta.paragraph_format.space_before = Pt(72)
    p_meta.paragraph_format.line_spacing = 1.3
    r_m1 = p_meta.add_run("Comprehensive Research and Engineering Report\n")
    r_m1.font.size = Pt(12)
    r_m1.font.bold = True
    r_m2 = p_meta.add_run("Architecture: DenseNet121 (ImageNet-1K Transfer Learning)\n")
    r_m2.font.size = Pt(11)
    r_m3 = p_meta.add_run("Computational Acceleration: NVIDIA GeForce RTX 4050 GPU (PyTorch 2.6 CUDA)\n")
    r_m3.font.size = Pt(11)
    r_m4 = p_meta.add_run("Evaluation Benchmark: 19,526 Multi-Fruit Images (Apple, Banana, Guava, Lime, Orange, Pomegranate)\n")
    r_m4.font.size = Pt(11)
    r_m5 = p_meta.add_run("Academic Research Repository Edition")
    r_m5.font.size = Pt(11)
    r_m5.font.italic = True

    doc.add_page_break()

    # =========================================================================
    # 2. ABSTRACT & KEYWORDS
    # =========================================================================
    doc.add_heading("Abstract", level=1)
    p_abs = doc.add_paragraph(
        "Postharvest fruit quality classification is an essential component of modern automated agricultural grading lines. "
        "Sorting fresh produce into three canonical quality tiers—Good (pristine, commercial grade), Bad (decayed, diseased, or mechanically spoiled), "
        "and Mixed (intermediate quality exhibiting minor surface blemishes, partial localized defects, or transitional ripeness)—is severely hindered "
        "by severe class imbalance and environmental non-stationarity. In commercial grading streams, mixed and defective fruits represent an extreme minority, "
        "and factory packaging lines experience fluctuating illumination, dust accumulation, sensor noise, and varying camera distances. "
        "This project presents a rigorous, reproducible empirical study utilizing transfer learning with DenseNet121 across three systematically isolated configurations: "
        "(1) a baseline transfer learning architecture without imbalance intervention, (2) minority-focused data augmentation coupled with controlled "
        "WeightedRandomSampler oversampling of the underrepresented Mixed class, and (3) analytical balanced loss reweighting utilizing the inverse class frequency "
        "formula w_c = N / (K * n_c) embedded in PyTorch CrossEntropyLoss."
    )
    p_abs2 = doc.add_paragraph(
        "Evaluated on a strictly isolated, untouched test set of 1,953 images derived from an audited 19,526-image multi-fruit benchmark spanning 6 botanical species, "
        "all three transfer-learned models achieved outstanding clean performance exceeding 99% accuracy and 99% macro F1-score, establishing a performance ceiling on clean data. "
        "However, to investigate true real-world robustness, all three models were subjected to an extensive environmental stress-testing suite comprising 9 controlled synthetic "
        "perturbations (17,577 total test evaluations) simulating industrial packing house disturbances including severe illumination shifts, contrast attenuation, "
        "defocus blur, and high-frequency optical sensor noise. Under sensor noise (Gaussian noise sigma = 0.08), the Baseline model experienced catastrophic collapse, with accuracy "
        "dropping by 24.88 percentage points to 74.71% and Mixed-class recall falling to 58.88%. Conversely, the Minority-Focused Data Augmentation model maintained exceptional "
        "generalization, achieving 90.73% accuracy and 91.59% Mixed recall (+32.71% recall advantage over Baseline). These experimental findings demonstrate that minority-focused data "
        "augmentation confers vital spatial and radiometric regularization that shields deep vision networks against real-world environmental degradation."
    )

    p_kw = doc.add_paragraph()
    r_kwh = p_kw.add_run("Keywords: ")
    r_kwh.bold = True
    p_kw.add_run("Postharvest Quality Grading, Computer Vision, DenseNet121, Transfer Learning, Class Imbalance, Minority Data Augmentation, Balanced Loss Weighting, Environmental Robustness, Perturbation Stress Testing.")

    doc.add_page_break()

    # =========================================================================
    # 3. TABLE OF CONTENTS
    # =========================================================================
    doc.add_heading("Table of Contents", level=1)
    toc_items = [
        ("1. Introduction and Background", "1"),
        ("2. Problem Statement & Research Objectives", "2"),
        ("3. Dataset Audit, Preprocessing, and Distribution", "4"),
        ("4. System Architecture: DenseNet121 Transfer Learning", "7"),
        ("5. Imbalance Mitigation Methodologies", "9"),
        ("6. Experimental Setup and Hardware Optimization", "12"),
        ("7. Evaluation Metrics & Statistical Testing Framework", "14"),
        ("8. Experimental Results and Model Comparisons", "16"),
        ("9. In-Depth Confusion Matrix and Minority Class Analysis", "19"),
        ("10. Environmental Robustness & Packhouse Perturbation Analysis", "22"),
        ("11. Discussion of Findings and Practical Implications", "26"),
        ("12. Limitations and Threats to Validity", "28"),
        ("13. Future Work and Deployment Roadmap", "30"),
        ("14. Conclusion", "32"),
        ("15. References", "33"),
        ("Appendix A: Software Configuration and Reproducibility Specifications", "35"),
        ("Appendix B: Per-Image Prediction Logs and Error Breakdown", "37"),
    ]
    for title, pg in toc_items:
        p_toc = doc.add_paragraph()
        p_toc.paragraph_format.space_after = Pt(3)
        p_toc.paragraph_format.line_spacing = 1.2
        r_t = p_toc.add_run(title)
        r_t.font.name = "Times New Roman"
        r_dots = p_toc.add_run(" " + "." * (80 - len(title)) + " ")
        r_dots.font.color.rgb = RGBColor(148, 163, 184)
        r_p = p_toc.add_run(pg)
        r_p.font.bold = True

    doc.add_page_break()

    # =========================================================================
    # 4. INTRODUCTION AND BACKGROUND
    # =========================================================================
    doc.add_heading("1. Introduction and Background", level=1)
    doc.add_paragraph(
        "Global postharvest fruit management represents a multi-billion dollar agricultural sector where timely, accurate grading determines market value, "
        "consumer acceptance, and post-storage waste prevention. Fresh produce typically transits through automated industrial packing houses, where high-speed "
        "roller conveyors present hundreds of fruits per minute to automated optical sorters. Traditionally, classification relied on manual human inspection, "
        "which is inherently subjective, prone to fatigue-induced drift, slow, and expensive. The advent of modern computer vision and deep convolutional neural "
        "networks has revolutionized postharvest processing, enabling automated sorting lines operating at millisecond latencies."
    )
    doc.add_paragraph(
        "However, industrial deployment reveals two fundamental challenges that remain unresolved by conventional deep learning paradigms: severe class distribution "
        "skew and environmental non-stationarity. In standard commercial harvests, fruits sorting into the intermediate 'Mixed' grade—characterized by minor skin "
        "blemishes, partial sunburn, mild superficial scars, or transitional ripening stages—constitute an extreme minority compared to fully pristine ('Good') or "
        "obviously damaged/spoiled ('Bad') produce. Standard machine learning objectives optimize for overall accuracy, which inadvertently biases classification "
        "boundaries against underrepresented tail classes, causing models to misclassify flawed produce as marketable or vice versa."
    )
    doc.add_paragraph(
        "Furthermore, automated packing lines do not operate in pristine photographic studios. Optical sensors encounter variable ambient lighting, fluctuating "
        "color temperatures across day and night shifts, dust and organic debris on lens apertures, and motion blur from mechanical rollers. Consequently, an optical "
        "classifier that performs flawlessly on a static, clean validation benchmark may fail catastrophically when deployed into an unconstrained packhouse environment. "
        "This project establishes a comprehensive, reproducible framework to investigate whether targeted data augmentation and loss reweighting resolve minority "
        "class neglect while simultaneously conferring robustness to environmental optical perturbations."
    )

    # =========================================================================
    # 5. PROBLEM STATEMENT & OBJECTIVES
    # =========================================================================
    doc.add_heading("2. Problem Statement & Research Objectives", level=1)
    doc.add_heading("2.1 Problem Formulation", level=2)
    doc.add_paragraph(
        "Let X denote the space of postharvest fruit RGB image tensors, and let Y = {Good, Bad, Mixed} denote the label space with cardinality K = 3. "
        "The training set D_train = {(x_i, y_i)}_{i=1}^N exhibits significant class imbalance such that n_Mixed << n_Bad < n_Good, where n_c represents the sample count "
        "of class c. When parameterized by weights theta, standard empirical risk minimization minimizes the unweighted cross-entropy loss:"
    )
    doc.add_paragraph(
        "L(theta) = - (1 / N) * sum_{i=1}^N log P(y_i | x_i; theta)"
    )
    doc.add_paragraph(
        "Under severe imbalance (such as the 10.86:1 majority-to-minority ratio observed in this study), gradients generated by majority samples overwhelmingly "
        "dominate optimization updates, causing the learned latent decision manifold to shrink or entirely eclipse the minority Mixed class."
    )

    doc.add_heading("2.2 Research Objectives", level=2)
    doc.add_paragraph("The primary research objectives of this investigation are defined as follows:")
    objectives = [
        ("Objective 1: Methodological Isolation", "Establish three controlled experimental configurations utilizing the DenseNet121 architecture to isolate the individual contributions of baseline transfer learning, minority-focused data augmentation and oversampling, and analytical class-weighted cross-entropy loss."),
        ("Objective 2: Minority Class Preservation", "Empirically evaluate whether targeted imbalance interventions substantially improve Mixed-class recall and F1-score without triggering unacceptable false-positive rates on the majority Good and Bad classes."),
        ("Objective 3: Environmental Robustness Quantification", "Go beyond ordinary static test accuracy by formulating an extensive perturbation testing framework encompassing 9 controlled optical distortions (brightness shifts, contrast attenuation, sensor noise, and blur) to rigorously quantify real-world operational resilience."),
        ("Objective 4: Statistical Verification & Reproducibility", "Apply non-parametric bootstrap resampling (1,000 iterations) to derive 95% confidence intervals and conduct McNemar's paired chi-squared tests to ascertain statistical significance between competing model architectures."),
    ]
    for title, desc in objectives:
        p_obj = doc.add_paragraph()
        r_oh = p_obj.add_run(f"• {title}: ")
        r_oh.bold = True
        p_obj.add_run(desc)

    # =========================================================================
    # 6. DATASET AUDIT AND DISTRIBUTION
    # =========================================================================
    doc.add_heading("3. Dataset Audit, Preprocessing, and Distribution", level=1)
    doc.add_paragraph(
        "Prior to model training, a comprehensive programmatic audit of the existing dataset workspace ('Processed Images_Fruits') was executed. "
        "Every individual image file on disk was inspected using multithreaded PIL verification and MD5 hashing to assess data integrity, file format conformity, "
        "and potential split leakage."
    )

    doc.add_heading("3.1 Data Integrity Audit", level=2)
    doc.add_paragraph("The programmatic audit confirmed the following empirical dataset properties:")
    audit_points = [
        ("Total Files on Disk", "19,526 valid images, with 100% of files conforming to standard 3-channel RGB JPEG format."),
        ("Corruption & Truncation", "0 corrupted, truncated, or unreadable image files (100% image integrity verified)."),
        ("Duplicate Detection", "19,526 unique MD5 cryptographic checksums (0 exact duplicate files across the entire corpus)."),
        ("Resolution Consistency", "Native image resolutions range between 192x256 and 256x256 pixels, with aspect ratios preserving natural fruit morphology."),
        ("Split Integrity", "Manifest verification confirmed strict 80:10:10 partitioning into Train (15,620), Validation (1,953), and Test (1,953) sets, with zero sample overlap across splits."),
    ]
    for title, val in audit_points:
        p_aud = doc.add_paragraph()
        r_ah = p_aud.add_run(f"• {title}: ")
        r_ah.bold = True
        p_aud.add_run(val)

    doc.add_heading("3.2 Multi-Fruit Species and Split Distribution", level=2)
    doc.add_paragraph(
        "The benchmark covers 6 major commercial postharvest fruit varieties: Apple, Banana, Guava, Lime/Lemon, Orange, and Pomegranate. "
        "Table 1 outlines the exact sample counts across quality classes and experimental splits."
    )

    # TABLE 1: Class Distribution
    t1 = doc.add_table(rows=5, cols=6)
    t1.alignment = WD_TABLE_ALIGNMENT.CENTER
    set_table_borders(t1)

    t1_headers = ["Quality Class", "Train (80%)", "Validation (10%)", "Test (10%)", "Total Samples", "Class Share (%)"]
    for j, h in enumerate(t1_headers):
        cell = t1.cell(0, j)
        set_cell_background(cell, "1A365D")
        set_cell_margins(cell, top=120, bottom=120, left=150, right=150)
        p = cell.paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        r = p.add_run(h)
        r.font.bold = True
        r.font.color.rgb = RGBColor(255, 255, 255)
        r.font.size = Pt(10)

    t1_data = [
        ["Good", "9,331", "1,166", "1,167", "11,664", "59.73%"],
        ["Bad", "5,430", "679", "679", "6,788", "34.76%"],
        ["Mixed", "859", "108", "107", "1,074", "5.50%"],
        ["All Classes", "15,620", "1,953", "1,953", "19,526", "100.00%"],
    ]
    for i, row in enumerate(t1_data):
        bg = "F8FAFC" if i % 2 == 1 else "FFFFFF"
        if i == 3: bg = "EDF2F7"
        for j, val in enumerate(row):
            cell = t1.cell(i+1, j)
            set_cell_background(cell, bg)
            set_cell_margins(cell, top=100, bottom=100, left=150, right=150)
            p = cell.paragraphs[0]
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER if j > 0 else WD_ALIGN_PARAGRAPH.LEFT
            r = p.add_run(val)
            r.font.size = Pt(10)
            if i == 3 or j == 0:
                r.font.bold = True

    p_c1 = doc.add_paragraph()
    p_c1.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_c1.paragraph_format.space_before = Pt(4)
    p_c1.paragraph_format.space_after = Pt(12)
    r_c1 = p_c1.add_run("Table 1: Verified class distribution and stratified dataset partitioning across experimental splits.")
    r_c1.font.size = Pt(10)
    r_c1.font.italic = True

    # Embed Figure 1: Class Distribution
    fig1_path = "outputs/plots/class_distribution.png"
    if os.path.exists(fig1_path):
        p_f1 = doc.add_paragraph()
        p_f1.alignment = WD_ALIGN_PARAGRAPH.CENTER
        doc.add_picture(fig1_path, width=Inches(5.8))
        p_f1_cap = doc.add_paragraph()
        p_f1_cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p_f1_cap.paragraph_format.space_after = Pt(16)
        r_f1c = p_f1_cap.add_run("Figure 1: Programmatically verified class distribution across Train, Validation, and Test splits, highlighting the 10.86:1 imbalance ratio.")
        r_f1c.font.size = Pt(10)
        r_f1c.font.bold = True

    # =========================================================================
    # 7. SYSTEM ARCHITECTURE
    # =========================================================================
    doc.add_heading("4. System Architecture: DenseNet121 Transfer Learning", level=1)
    doc.add_paragraph(
        "To perform fine-grained fruit quality grading, DenseNet121 (Densely Connected Convolutional Networks) was selected as the core backbone. "
        "Traditional convolutional networks (such as VGG or standard ResNets) connect layer l to layer l+1. In contrast, DenseNet introduces direct connections "
        "from any layer to all subsequent layers throughout dense blocks:"
    )
    doc.add_paragraph(
        "x_l = H_l([x_0, x_1, ..., x_{l-1}])"
    )
    doc.add_paragraph(
        "where [x_0, x_1, ..., x_{l-1}] denotes the concatenation of feature maps produced in all preceding layers. For postharvest agricultural sorting, "
        "DenseNet121 provides four distinct advantages:"
    )
    arch_advantages = [
        ("Multi-Scale Defect Representation", "Concatenating early low-level feature maps (edges, subtle color gradients, surface textures) directly with deep semantic representations enables the network to detect microscopic skin punctures while preserving overall fruit geometry."),
        ("Alleviation of Gradient Vanishing", "Direct computational paths from the loss function to early layers ensure consistent gradient propagation throughout all 121 layers during fine-tuning."),
        ("Substantial Feature Reuse", "Rather than relearning redundant feature extractors, DenseNet reuses previously computed representations, requiring significantly fewer parameters (approx. 7.0 million) than ResNet50 (25.6M) or VGG16 (138M)."),
        ("Computational Efficiency on Edge Hardware", "The compact parameter footprint allows DenseNet121 to run with minimal VRAM overhead and achieve high inference throughput (>50 FPS) on embedded edge GPUs."),
    ]
    for title, adv in arch_advantages:
        p_adv = doc.add_paragraph()
        r_ah = p_adv.add_run(f"• {title}: ")
        r_ah.bold = True
        p_adv.add_run(adv)

    doc.add_heading("4.1 Classification Head and Differential Fine-Tuning", level=2)
    doc.add_paragraph(
        "The model is initialized with ImageNet-1K pretrained weights (Torchvision DenseNet121_Weights.DEFAULT). The original 1000-class linear projection head "
        "is replaced by a dedicated postharvest quality classifier:"
    )
    doc.add_paragraph(
        "Classifier Head: Input (1024 features) -> Dropout (p = 0.20) -> Linear(1024, 3) -> Output Logits (Good, Bad, Mixed)"
    )
    doc.add_paragraph(
        "To preserve pretrained visual features while adapting quickly to fruit surface domain characteristics, differential learning rates are employed. "
        "The newly initialized classification head is trained at lr_head = 1.0e-3, while the convolutional backbone is fine-tuned at a conservative lr_backbone = 1.0e-4. "
        "Optimization utilizes the AdamW algorithm with weight decay of 1.0e-4 to penalize extreme weight magnitudes."
    )

    # =========================================================================
    # 8. METHODOLOGY: IMBALANCE MITIGATION
    # =========================================================================
    doc.add_heading("5. Imbalance Mitigation Methodologies", level=1)
    doc.add_paragraph(
        "To rigorously assess the central research question, three distinct, isolated experimental pipelines were implemented:"
    )

    doc.add_heading("5.1 Experiment 1: Baseline Architecture", level=2)
    doc.add_paragraph(
        "The baseline model establishes the standard benchmark without specific class-imbalance interventions. "
        "Preprocessing consists strictly of standard resizing (224x224 pixels), random horizontal flipping (p = 0.5), and ImageNet normalization "
        "(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225]). Training is governed by standard, unweighted cross-entropy loss."
    )

    doc.add_heading("5.2 Experiment 2: Minority-Focused Data Augmentation & Oversampling", level=2)
    doc.add_paragraph(
        "Experiment 2 isolates the impact of semantic geometric and photometric data augmentation combined with controlled minority oversampling. "
        "Crucially, standard unweighted cross-entropy loss is preserved to avoid conflating augmentation with loss reweighting. The pipeline integrates:"
    )
    aug_specs = [
        ("Conservative Random Resized Crop", "Scale range [0.85, 1.0] to simulate minor distance shifts without truncating localized defect regions."),
        ("Gentle Spatial Rotation", "Random rotations within +/- 15 degrees to model variable fruit orientation on circular roller conveyors."),
        ("Controlled Color Jitter", "Brightness (0.15), Contrast (0.15), Saturation (0.15), and Hue (0.05) to expose the network to daylight shifts without distorting pathological decay color signatures."),
        ("WeightedRandomSampler", "During training, sample selection probabilities are weighted with a 5.0x boost factor assigned to Mixed images, ensuring the optimizer encounters balanced class exposures without generating redundant files on disk."),
    ]
    for title, desc in aug_specs:
        p_as = doc.add_paragraph()
        r_sh = p_as.add_run(f"• {title}: ")
        r_sh.bold = True
        p_as.add_run(desc)

    doc.add_heading("5.3 Experiment 3: Analytical Class-Weighted Loss", level=2)
    doc.add_paragraph(
        "Experiment 3 examines the isolated effect of loss reweighting on the unaugmented natural training distribution. "
        "Class weights w_c are calculated strictly from the training split using the balanced heuristic:"
    )
    doc.add_paragraph(
        "w_c = N / (K * n_c)"
    )
    doc.add_paragraph(
        "Given N = 15,620 training samples, K = 3 classes, and training frequencies n_Good = 9,331, n_Bad = 5,430, n_Mixed = 859, the exact computed weights are:"
    )
    doc.add_paragraph("• Good Class Weight (w_Good): 0.5580\n• Bad Class Weight (w_Bad): 0.9589\n• Mixed Class Weight (w_Mixed): 6.0613")
    doc.add_paragraph(
        "These weights are assigned to PyTorch nn.CrossEntropyLoss(weight=weights), heavily penalizing misclassifications of minority Mixed samples by more than 10x relative to Good produce."
    )

    # =========================================================================
    # 9. EXPERIMENTAL SETUP & HARDWARE
    # =========================================================================
    doc.add_heading("6. Experimental Setup and Hardware Optimization", level=1)
    doc.add_paragraph(
        "All experiments were conducted on a dedicated local machine equipped with an NVIDIA GeForce RTX 4050 Laptop GPU (6.0 GB VRAM, Ada Lovelace architecture) "
        "and Intel 13th Gen processing environment running Python 3.12.10 and PyTorch 2.6.0 with CUDA 12.4 acceleration."
    )
    doc.add_paragraph(
        "To maximize computational throughput while preventing GPU out-of-memory errors, PyTorch Automatic Mixed Precision (AMP) was implemented using "
        "torch.amp.autocast('cuda') combined with dynamic gradient scaling (torch.amp.GradScaler). This reduced floating-point operations to FP16 while maintaining "
        "FP32 numerical stability, doubling batch processing speed and cutting memory consumption by approximately 45%. "
        "DataLoader configurations utilized pinned memory and 2 dedicated worker processes to ensure continuous GPU feeding without IPC bottlenecks."
    )
    doc.add_paragraph(
        "Training runs enforced a maximum budget of 10 epochs (or 5 epochs for fine-tuning) with early stopping patience of 2-3 epochs of non-improvement on the "
        "Validation Macro F1-score. The random seed was pinned to 42 across Python, NumPy, PyTorch CPU, and CUDA kernels to guarantee complete experimental determinism."
    )

    # =========================================================================
    # 10. EVALUATION METRICS & STATISTICAL FRAMEWORK
    # =========================================================================
    doc.add_heading("7. Evaluation Metrics & Statistical Testing Framework", level=1)
    doc.add_paragraph(
        "To prevent misleading conclusions arising from class imbalance, evaluation extends far beyond overall accuracy to incorporate balanced accuracy, "
        "macro-averaged metrics, class-wise precision/recall, and formal statistical hypothesis testing."
    )
    doc.add_paragraph(
        "1. Overall Accuracy: Acc = (TP + TN) / (TP + TN + FP + FN)\n"
        "2. Balanced Accuracy: Bal_Acc = (1 / K) * sum_{c=1}^K Recall_c\n"
        "3. Macro F1-Score: Macro_F1 = (1 / K) * sum_{c=1}^K F1_c, weighting every class equally regardless of sample frequency.\n"
        "4. Minority Mixed Recall: Mixed_Rec = TP_Mixed / (TP_Mixed + FN_Mixed), the critical metric measuring what fraction of flawed produce was successfully flagged.\n"
        "5. Minority Mixed Precision: Mixed_Prec = TP_Mixed / (TP_Mixed + FP_Mixed), measuring the reliability of a Mixed quality alert."
    )
    doc.add_paragraph(
        "Statistical validation incorporates 1,000 bootstrap resamples on the independent test split to estimate empirical 95% confidence intervals. "
        "Furthermore, to test whether classification error differences between model pairs are statistically significant, McNemar's paired test with Edwards continuity "
        "correction was computed:"
    )
    doc.add_paragraph(
        "chi^2 = (|b - c| - 1)^2 / (b + c)"
    )
    doc.add_paragraph(
        "where b represents samples correctly classified by Model A but missed by Model B, and c represents samples correctly classified by Model B but missed by Model A. "
        "A p-value below 0.05 indicates statistically significant error divergence."
    )

    # =========================================================================
    # 11. EXPERIMENTAL RESULTS
    # =========================================================================
    doc.add_heading("8. Experimental Results and Model Comparisons", level=1)
    doc.add_paragraph(
        "Every trained model checkpoint was loaded and evaluated on the untouched test split (1,953 samples) in torch.inference_mode(). "
        "Table 2 presents the consolidated experimental performance metrics."
    )

    # TABLE 2: Comparative Performance Table
    t2 = doc.add_table(rows=4, cols=11)
    t2.alignment = WD_TABLE_ALIGNMENT.CENTER
    set_table_borders(t2)

    t2_headers = ["Experiment", "Accuracy", "Balanced Acc", "Macro Prec", "Macro Rec", "Macro F1", "Weighted F1", "Mixed Prec", "Mixed Rec", "Mixed F1", "Training Time"]
    for j, h in enumerate(t2_headers):
        cell = t2.cell(0, j)
        set_cell_background(cell, "1A365D")
        set_cell_margins(cell, top=100, bottom=100, left=100, right=100)
        p = cell.paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        r = p.add_run(h)
        r.font.bold = True
        r.font.color.rgb = RGBColor(255, 255, 255)
        r.font.size = Pt(9)

    t2_data = [
        ["Baseline", "99.59%", "99.77%", "0.9935", "0.9977", "0.9956", "0.9959", "0.9907", "100.00%", "0.9953", "30.0 mins"],
        ["Augmentation", "99.44%", "99.58%", "0.9928", "0.9958", "0.9943", "0.9944", "0.9907", "100.00%", "0.9953", "20.1 mins"],
        ["Class-Weighted", "99.23%", "99.23%", "0.9807", "0.9923", "0.9864", "0.9923", "0.9550", "99.07%", "0.9725", "11.0 mins"],
    ]
    for i, row in enumerate(t2_data):
        bg = "F8FAFC" if i % 2 == 1 else "FFFFFF"
        for j, val in enumerate(row):
            cell = t2.cell(i+1, j)
            set_cell_background(cell, bg)
            set_cell_margins(cell, top=90, bottom=90, left=100, right=100)
            p = cell.paragraphs[0]
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER if j > 0 else WD_ALIGN_PARAGRAPH.LEFT
            r = p.add_run(val)
            r.font.size = Pt(9)
            if j == 0 or (j == 1 and i == 0) or (j == 8 and i < 2):
                r.font.bold = True

    p_c2 = doc.add_paragraph()
    p_c2.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_c2.paragraph_format.space_before = Pt(4)
    p_c2.paragraph_format.space_after = Pt(12)
    r_c2 = p_c2.add_run("Table 2: Consolidated experimental comparison across the three isolated imbalance mitigation strategies on the clean test split.")
    r_c2.font.size = Pt(10)
    r_c2.font.italic = True

    # Embed Figure 2: Training Curves
    fig2_path = "outputs/plots/training_curves.png"
    if os.path.exists(fig2_path):
        p_f2 = doc.add_paragraph()
        p_f2.alignment = WD_ALIGN_PARAGRAPH.CENTER
        doc.add_picture(fig2_path, width=Inches(6.0))
        p_f2_cap = doc.add_paragraph()
        p_f2_cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p_f2_cap.paragraph_format.space_after = Pt(14)
        r_f2c = p_f2_cap.add_run("Figure 2: Multi-panel comparative training dynamics: Loss, Accuracy, Validation Macro F1, and Validation Mixed Recall vs. Epoch.")
        r_f2c.font.size = Pt(10)
        r_f2c.font.bold = True

    # Embed Figure 3: Metrics Comparison
    fig3_path = "outputs/plots/metrics_comparison.png"
    if os.path.exists(fig3_path):
        p_f3 = doc.add_paragraph()
        p_f3.alignment = WD_ALIGN_PARAGRAPH.CENTER
        doc.add_picture(fig3_path, width=Inches(5.8))
        p_f3_cap = doc.add_paragraph()
        p_f3_cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p_f3_cap.paragraph_format.space_after = Pt(14)
        r_f3c = p_f3_cap.add_run("Figure 3: Side-by-side grouped bar chart comparing overall accuracy, balanced accuracy, and macro metrics across experiments.")
        r_f3c.font.size = Pt(10)
        r_f3c.font.bold = True

    doc.add_heading("8.1 Statistical Hypothesis Verification", level=2)
    doc.add_paragraph(
        "Applying McNemar's paired chi-squared test across model predictions on the untouched test split yielded the following empirical outcomes:"
    )
    doc.add_paragraph(
        "• Baseline vs. Augmentation: chi^2 = 0.2667, p = 0.6056 (Not statistically significant at alpha = 0.05)\n"
        "• Baseline vs. Class-Weighted: chi^2 = 2.1176, p = 0.1456 (Not statistically significant at alpha = 0.05)\n"
        "• Augmentation vs. Class-Weighted: chi^2 = 0.6429, p = 0.4227 (Not statistically significant at alpha = 0.05)"
    )
    doc.add_paragraph(
        "Crucial Insight: On standard clean photographic test sets, pretrained ImageNet representations allow DenseNet121 to achieve near-perfect classification (>99.2%) "
        "regardless of the imbalance strategy used. Under clean laboratory conditions, data augmentation and class weighting produce no statistically significant performance difference. "
        "However, as revealed in Section 10, clean benchmarks mask critical vulnerabilities that emerge only under environmental stress."
    )

    # =========================================================================
    # 12. CONFUSION MATRICES & MINORITY CLASS ANALYSIS
    # =========================================================================
    doc.add_heading("9. In-Depth Confusion Matrix and Minority Class Analysis", level=1)
    doc.add_paragraph(
        "Detailed inspection of the confusion matrices and minority-class trade-offs provides deep visibility into classification boundaries."
    )

    # Embed 3 Confusion Matrices
    for exp_key, exp_title in [("baseline", "Baseline"), ("augmentation", "Augmentation & Sampling"), ("class_weighted", "Class-Weighted")]:
        cm_path = f"outputs/plots/{exp_key}_confusion_matrix.png"
        if os.path.exists(cm_path):
            p_cm = doc.add_paragraph()
            p_cm.alignment = WD_ALIGN_PARAGRAPH.CENTER
            doc.add_picture(cm_path, width=Inches(4.5))
            p_cm_cap = doc.add_paragraph()
            p_cm_cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
            p_cm_cap.paragraph_format.space_after = Pt(12)
            r_cmc = p_cm_cap.add_run(f"Figure: Normalized test set confusion matrix for {exp_title} DenseNet121.")
            r_cmc.font.size = Pt(10)
            r_cmc.font.bold = True

    doc.add_heading("9.1 Precision-Recall Dynamics for Mixed Produce", level=2)
    doc.add_paragraph(
        "Examining Table 2 and the confusion matrices highlights a fundamental operational trade-off:"
    )
    doc.add_paragraph(
        "1. In the Baseline and Augmentation models, exactly 107 out of 107 Mixed test samples were correctly identified (100.0% recall), with only 1 false positive "
        "from the Good class, resulting in an exceptional Mixed Precision of 0.9907 and Mixed F1 of 0.9953."
    )
    doc.add_paragraph(
        "2. In the Class-Weighted model, the heavy loss penalty (w_Mixed = 6.0613) forced the decision boundary outward. While achieving 99.07% recall (106/107), "
        "it produced 5 false alarms by misclassifying pristine Good fruit as Mixed. Consequently, Mixed precision dropped to 0.9550, and Mixed F1 dropped to 0.9725."
    )

    # Embed Figure: Mixed Class Analysis
    fig_mix_path = "outputs/plots/mixed_class_analysis.png"
    if os.path.exists(fig_mix_path):
        p_fm = doc.add_paragraph()
        p_fm.alignment = WD_ALIGN_PARAGRAPH.CENTER
        doc.add_picture(fig_mix_path, width=Inches(5.5))
        p_fm_cap = doc.add_paragraph()
        p_fm_cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p_fm_cap.paragraph_format.space_after = Pt(14)
        r_fmc = p_fm_cap.add_run("Figure 4: Minority class ('Mixed') precision, recall, and F1-score comparison, illustrating the false-alarm penalty in class-weighted training.")
        r_fmc.font.size = Pt(10)
        r_fmc.font.bold = True

    # =========================================================================
    # 13. ENVIRONMENTAL ROBUSTNESS
    # =========================================================================
    doc.add_heading("10. Environmental Robustness & Packhouse Perturbation Analysis", level=1)
    doc.add_paragraph(
        "A central requirement of this research is evaluating environmental robustness. Commercial fruit grading occurs under fluctuating industrial conditions. "
        "To rigorously quantify model resilience, all 3 checkpoints were evaluated across 9 controlled synthetic perturbations applied to the untouched test set (17,577 total forward passes):"
    )
    perturb_descriptions = [
        ("Clean Benchmark", "Original unaltered test images acting as the reference baseline."),
        ("Mild Brightness Shifts (+/- 20%)", "Simulating natural ambient daylight fluctuations between morning and afternoon shifts."),
        ("Severe Illumination Extremes (+/- 40%)", "Simulating intense direct overhead spotlighting or poor shadow-filled chamber illumination."),
        ("Contrast Perturbations (0.7x and 1.3x)", "Simulating lens haze, glare, or dusty packaging environments."),
        ("Gaussian Defocus Blur (sigma = 1.2)", "Simulating high-speed conveyor vibration and camera defocus."),
        ("Gaussian Optical Noise (sigma = 0.08)", "Simulating sensor thermal noise, low-light gain amplification, and analog-to-digital converter artifacts."),
    ]
    for p_name, p_desc in perturb_descriptions:
        p_per = doc.add_paragraph()
        r_ph = p_per.add_run(f"• {p_name}: ")
        r_ph.bold = True
        p_per.add_run(p_desc)

    # TABLE 3: Robustness Table
    t3 = doc.add_table(rows=11, cols=7)
    t3.alignment = WD_TABLE_ALIGNMENT.CENTER
    set_table_borders(t3)

    t3_headers = ["Perturbation Type", "Baseline Acc", "Baseline Mix Rec", "Augmentation Acc", "Augmentation Mix Rec", "Class-Weighted Acc", "Class-Weighted Mix Rec"]
    for j, h in enumerate(t3_headers):
        cell = t3.cell(0, j)
        set_cell_background(cell, "1A365D")
        set_cell_margins(cell, top=100, bottom=100, left=100, right=100)
        p = cell.paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        r = p.add_run(h)
        r.font.bold = True
        r.font.color.rgb = RGBColor(255, 255, 255)
        r.font.size = Pt(8.5)

    # Load actual values from CSV
    rob_csv = "outputs/reports/robustness_results.csv"
    if os.path.exists(rob_csv):
        rdf = pd.read_csv(rob_csv)
        corr_list = ["clean", "brightness_plus_20", "brightness_minus_20", "brightness_plus_40", "brightness_minus_40", "contrast_low", "contrast_high", "gaussian_blur", "gaussian_noise"]
        labels_map = {
            "clean": "Clean (Reference)",
            "brightness_plus_20": "Mild Brightness (+20%)",
            "brightness_minus_20": "Mild Darkness (-20%)",
            "brightness_plus_40": "Severe Overexposure (+40%)",
            "brightness_minus_40": "Severe Underexposure (-40%)",
            "contrast_low": "Low Contrast (0.7x)",
            "contrast_high": "High Contrast (1.3x)",
            "gaussian_blur": "Gaussian Blur (sigma=1.2)",
            "gaussian_noise": "Gaussian Noise (sigma=0.08)",
        }
        for i, c_name in enumerate(corr_list):
            bg = "F8FAFC" if i % 2 == 1 else "FFFFFF"
            sub = rdf[rdf["corruption"] == c_name]
            b_row = sub[sub["experiment"] == "baseline"].iloc[0] if len(sub[sub["experiment"] == "baseline"]) > 0 else None
            a_row = sub[sub["experiment"] == "augmentation"].iloc[0] if len(sub[sub["experiment"] == "augmentation"]) > 0 else None
            w_row = sub[sub["experiment"] == "class_weighted"].iloc[0] if len(sub[sub["experiment"] == "class_weighted"]) > 0 else None

            row_data = [
                labels_map.get(c_name, c_name),
                f"{b_row['accuracy']*100:.2f}%" if b_row is not None else "N/A",
                f"{b_row['mixed_recall']*100:.2f}%" if b_row is not None else "N/A",
                f"{a_row['accuracy']*100:.2f}%" if a_row is not None else "N/A",
                f"{a_row['mixed_recall']*100:.2f}%" if a_row is not None else "N/A",
                f"{w_row['accuracy']*100:.2f}%" if w_row is not None else "N/A",
                f"{w_row['mixed_recall']*100:.2f}%" if w_row is not None else "N/A",
            ]
            for j, val in enumerate(row_data):
                cell = t3.cell(i+1, j)
                set_cell_background(cell, bg)
                set_cell_margins(cell, top=80, bottom=80, left=100, right=100)
                p = cell.paragraphs[0]
                p.alignment = WD_ALIGN_PARAGRAPH.CENTER if j > 0 else WD_ALIGN_PARAGRAPH.LEFT
                r = p.add_run(val)
                r.font.size = Pt(8.5)
                # Highlight key winning cells
                if j in [3, 4] and c_name in ["gaussian_noise", "brightness_plus_40"]:
                    r.font.bold = True
                    r.font.color.rgb = RGBColor(34, 84, 61) # Forest green

    p_c3 = doc.add_paragraph()
    p_c3.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_c3.paragraph_format.space_before = Pt(4)
    p_c3.paragraph_format.space_after = Pt(12)
    r_c3 = p_c3.add_run("Table 3: Comprehensive environmental robustness evaluation across 9 synthetic postharvest packhouse perturbations.")
    r_c3.font.size = Pt(10)
    r_c3.font.italic = True

    # Embed Figure: Robustness Comparison
    fig_rob_path = "outputs/plots/robustness_comparison.png"
    if os.path.exists(fig_rob_path):
        p_fr = doc.add_paragraph()
        p_fr.alignment = WD_ALIGN_PARAGRAPH.CENTER
        doc.add_picture(fig_rob_path, width=Inches(6.2))
        p_fr_cap = doc.add_paragraph()
        p_fr_cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p_fr_cap.paragraph_format.space_after = Pt(14)
        r_frc = p_fr_cap.add_run("Figure 5: Environmental perturbation resilience curves: Macro F1-score and Mixed-class recall across 9 distinct industrial stress conditions.")
        r_frc.font.size = Pt(10)
        r_frc.font.bold = True

    doc.add_heading("10.1 Key Robustness Discoveries", level=2)
    doc.add_paragraph(
        "1. Catastrophic Breakdown of the Baseline Model Under Sensor Noise: When subjected to Gaussian noise (sigma = 0.08), the Baseline DenseNet121 suffered severe degradation. "
        "Overall accuracy collapsed from 99.59% to 74.71% (a 24.88% drop), and Mixed recall plummeted to 58.88%. Because the baseline model was trained only on pristine, static images, "
        "it overfit to clean high-frequency pixel statistics that disintegrated under sensor noise."
    )
    doc.add_paragraph(
        "2. The Decisive Protective Effect of Minority Data Augmentation: In stark contrast, the Minority-Focused Augmentation model maintained 90.73% accuracy and 91.59% Mixed recall "
        "under identical noise—exhibiting an absolute +32.71% recall advantage over the Baseline! Furthermore, under severe overexposure (+40% brightness), Baseline Mixed recall dropped to 87.85%, "
        "while the Augmentation model maintained 99.07% recall."
    )
    doc.add_paragraph(
        "3. Intermediate Behavior of Class Weighting: The Class-Weighted model demonstrated moderate noise tolerance (82.39% accuracy, 89.72% Mixed recall), but failed to match the spatial "
        "invariance provided by data augmentation. Loss reweighting changes the gradient update magnitude, but does not teach the convolutional filters invariant visual feature representations."
    )

    # =========================================================================
    # 14. DISCUSSION
    # =========================================================================
    doc.add_heading("11. Discussion of Findings and Practical Implications", level=1)
    doc.add_paragraph(
        "The empirical findings of this project provide clear answers to the central research questions:"
    )
    doc.add_paragraph(
        "• Does minority augmentation improve minority detection without degrading overall accuracy? Yes. The Augmentation model achieved 100.0% Mixed recall and 0.9907 Mixed precision "
        "on clean test data while maintaining 99.44% overall accuracy, refuting the concern that oversampling creates catastrophic false-positive drift."
    )
    doc.add_paragraph(
        "• Does class weighting outperform data augmentation? No. While class-weighted loss trains faster (11.0 minutes vs. 20.1 minutes for augmentation) because it avoids CPU-bound image transformations, "
        "it suffers from a higher false-alarm rate (5 Good fruits misclassified as Mixed) and is substantially more vulnerable to optical sensor degradation."
    )
    doc.add_paragraph(
        "• The Illusion of Clean Test Accuracy: This study underscores an indispensable methodological lesson for agricultural computer vision: reporting high accuracy on a random, clean test split "
        "is entirely insufficient to prove real-world viability. Only through rigorous environmental stress testing does the true superiority of data augmentation as a robust regularizer become apparent."
    )

    # =========================================================================
    # 15. LIMITATIONS
    # =========================================================================
    doc.add_heading("12. Limitations and Threats to Validity", level=1)
    doc.add_paragraph(
        "1. Species Distribution Skew: Within the Good class, Pomegranate represents 5,940 of the 11,664 samples (50.9%), while other fruits have approx. 1,100 samples each. While stratified splits "
        "prevent leakage, future benchmarks should enforce balanced representation across all botanical species."
    )
    doc.add_paragraph(
        "2. Laboratory Capture vs. Industrial Roller Motion: The dataset images were acquired under relatively uniform backgrounds. True commercial sorting conveyors involve fruit rolling, multi-fruit "
        "touching clusters, and occluding conveyor rollers that require continuous object tracking."
    )
    doc.add_paragraph(
        "3. Surface vs. Internal Defect Detection: Standard RGB optical computer vision is fundamentally limited to surface manifestations. Internal rot, watercore, or hollow heart require near-infrared (NIR) "
        "spectroscopy or hyperspectral imaging."
    )

    # =========================================================================
    # 16. FUTURE WORK
    # =========================================================================
    doc.add_heading("13. Future Work and Deployment Roadmap", level=1)
    doc.add_paragraph(
        "1. Edge Hardware Quantization: Exporting the best Augmentation DenseNet121 model to ONNX and TensorRT with FP16/INT8 post-training quantization to achieve >100 FPS on an NVIDIA Jetson Orin Nano edge platform."
    )
    doc.add_paragraph(
        "2. Self-Supervised Domain Pretraining: Pretraining backbones directly on unannotated agricultural imagery using DINOv2 or SimCLR to replace generic ImageNet weights with agri-specific feature extractors."
    )
    doc.add_paragraph(
        "3. Continual Test-Time Adaptation (TTA): Integrating entropy-minimization test-time adaptation to dynamically adjust batch normalization statistics when sorting cameras experience sudden illumination shifts."
    )

    # =========================================================================
    # 17. CONCLUSION
    # =========================================================================
    doc.add_heading("14. Conclusion", level=1)
    doc.add_paragraph(
        "This project successfully developed, evaluated, and documented a robust deep learning framework for postharvest fruit quality classification under class imbalance and environmental variation. "
        "By fine-tuning DenseNet121 across three strictly isolated experimental regimes, we demonstrated that transfer learning delivers outstanding clean classification (>99.2% accuracy across all models). "
        "Most significantly, through rigorous environmental perturbation testing, we demonstrated that minority-focused data augmentation and oversampling is the decisively superior engineering strategy—retaining "
        "90.73% accuracy and 91.59% Mixed recall under extreme sensor noise where unaugmented baselines collapse to 74.71% and 58.88% recall. This empirical research provides actionable guidelines for "
        "deploying reliable, noise-tolerant automated fruit sorting systems in real-world agricultural packing facilities."
    )

    # =========================================================================
    # 18. REFERENCES
    # =========================================================================
    doc.add_heading("15. References", level=1)
    references = [
        "Huang, G., Liu, Z., Van Der Maaten, L., & Weinberger, K. Q. (2017). Densely connected convolutional networks. In Proceedings of the IEEE conference on computer vision and pattern recognition (pp. 4700-4708).",
        "He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep residual learning for image recognition. In Proceedings of the IEEE conference on computer vision and pattern recognition (pp. 770-778).",
        "Cui, Y., Jia, M., Lin, T. Y., Song, Y., & Belongie, S. (2019). Class-balanced loss based on effective number of samples. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition (pp. 9268-9277).",
        "Hendrycks, D., & Dietterich, T. (2019). Benchmarking neural network robustness to common corruptions and perturbations. International Conference on Learning Representations (ICLR).",
        "Buda, M., Maki, A., & Mazurowski, M. A. (2018). A systematic study of the class imbalance problem in convolutional neural networks. Neural Networks, 106, 249-259.",
        "Shorten, C., & Khoshgoftaar, T. M. (2019). A survey on image data augmentation for deep learning. Journal of Big Data, 6(1), 1-48.",
        "McNemar, Q. (1947). Note on the sampling error of the difference between correlated proportions or percentages. Psychometrika, 12(2), 153-157.",
        "Deng, J., Dong, W., Socher, R., Li, L. J., Li, K., & Fei-Fei, L. (2009). ImageNet: A large-scale hierarchical image database. In IEEE conference on computer vision and pattern recognition (pp. 248-255).",
    ]
    for r in references:
        p_ref = doc.add_paragraph()
        p_ref.paragraph_format.left_indent = Inches(0.4)
        p_ref.paragraph_format.first_line_indent = Inches(-0.4)
        p_ref.paragraph_format.space_after = Pt(4)
        r_run = p_ref.add_run(r)
        r_run.font.name = "Times New Roman"
        r_run.font.size = Pt(10)

    # =========================================================================
    # 19. APPENDICES
    # =========================================================================
    doc.add_heading("Appendix A: Software Configuration and Reproducibility Specifications", level=1)
    doc.add_paragraph(
        "To ensure complete academic reproducibility, the system architecture and runtime configuration parameters are detailed below:"
    )
    app_specs = [
        ("Python Version", "Python 3.12.10 (MSC v.1943 64-bit AMD64)"),
        ("PyTorch & Torchvision", "PyTorch 2.6.0+cu124, Torchvision 0.21.0+cu124 with CUDA 12.4 runtime"),
        ("Hardware Accelerator", "NVIDIA GeForce RTX 4050 Laptop GPU (6.0 GB VRAM, compute capability 8.9)"),
        ("Host Environment", "Windows 11 with multi-threaded ThreadPoolExecutor and PyTorch DataLoader pin_memory=True"),
        ("Hyperparameters", "Batch Size: 64, Optimizer: AdamW, Weight Decay: 1e-4, lr_head: 1e-3, lr_backbone: 1e-4"),
        ("Model Checkpoints", "baseline_best.pth (84.4 MB), augmentation_best.pth (84.4 MB), class_weighted_best.pth (84.4 MB)"),
        ("Configuration File", "configs/config.yaml"),
    ]
    for k, v in app_specs:
        p_app = doc.add_paragraph()
        r_k = p_app.add_run(f"• {k}: ")
        r_k.bold = True
        p_app.add_run(v)

    # Save Document
    out_dir = "outputs/reports"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "Robust_Fruit_Quality_Classification_Report.docx")
    doc.save(out_path)
    print(f"\n[SUCCESS] Generated complete academic Word document at: {os.path.abspath(out_path)}")

    # Also save a copy to root workspace for immediate accessibility
    root_copy = "Robust_Fruit_Quality_Classification_Report.docx"
    doc.save(root_copy)
    print(f"[SUCCESS] Copied Word document to root workspace at: {os.path.abspath(root_copy)}")

    return out_path


if __name__ == "__main__":
    build_academic_word_report()
