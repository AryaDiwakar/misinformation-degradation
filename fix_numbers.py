#!/usr/bin/env python3
"""Fix slide number placeholders to match slide positions."""
from pptx import Presentation

pptx_path = "/Users/aryadiwakar/misinformation-degradation/Review 1 - 22 07 2026.pptx"
prs = Presentation(pptx_path)

slide_number_map = {
    0: 1,  # Title
    1: 2,  # Aim
    2: 3,  # Abstract
    3: 4,  # Methodology (new)
    4: 5,  # Models & Feature Engineering (new)
    5: 6,  # Implementation & Code Structure (new)
    6: 7,  # Literature Review (old slide 4)
    7: 8,  # Expected Results & Evaluation (new)
    8: 9,  # Research Gap (old slide 5)
    9: 10, # Objectives (old slide 6)
    10: 11, # Timeline & Future Work (new)
    11: 12, # References (old slide 7)
}

for slide_idx, slide in enumerate(prs.slides):
    expected_num = slide_number_map.get(slide_idx)
    if expected_num is None:
        continue
    for shape in slide.shapes:
        if shape.is_placeholder:
            if shape.placeholder_format.idx == 8:  # Slide Number placeholder
                shape.text = str(expected_num)

pptx_path_out = "/Users/aryadiwakar/misinformation-degradation/Review 1 - 22 07 2026.pptx"
prs.save(pptx_path_out)
print(f"Saved slide numbers to {pptx_path_out}")
