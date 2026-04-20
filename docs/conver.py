from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.enum.text import PP_ALIGN
from pptx.dml.color import RGBColor

# Create presentation
prs = Presentation()
prs.slide_width = Inches(10)
prs.slide_height = Inches(7.5)

def add_title_slide(prs, title, subtitle, info):
    """Add title slide"""
    slide = prs.slides.add_slide(prs.slide_layouts[6])  # Blank layout
    
    # Set background color (purple gradient replaced with solid color)
    background = slide.background
    fill = background.fill
    fill.solid()
    fill.fore_color.rgb = RGBColor(102, 126, 234)
    
    # Title
    title_box = slide.shapes.add_textbox(Inches(1), Inches(2.5), Inches(8), Inches(1))
    title_frame = title_box.text_frame
    title_frame.text = title
    title_para = title_frame.paragraphs[0]
    title_para.font.size = Pt(44)
    title_para.font.bold = True
    title_para.font.color.rgb = RGBColor(255, 255, 255)
    title_para.alignment = PP_ALIGN.CENTER
    
    # Subtitle
    subtitle_box = slide.shapes.add_textbox(Inches(1), Inches(3.7), Inches(8), Inches(0.8))
    subtitle_frame = subtitle_box.text_frame
    subtitle_frame.text = subtitle
    subtitle_para = subtitle_frame.paragraphs[0]
    subtitle_para.font.size = Pt(24)
    subtitle_para.font.color.rgb = RGBColor(255, 255, 255)
    subtitle_para.alignment = PP_ALIGN.CENTER
    
    # Info
    info_box = slide.shapes.add_textbox(Inches(1), Inches(4.7), Inches(8), Inches(0.6))
    info_frame = info_box.text_frame
    info_frame.text = info
    info_para = info_frame.paragraphs[0]
    info_para.font.size = Pt(18)
    info_para.font.color.rgb = RGBColor(255, 255, 255)
    info_para.alignment = PP_ALIGN.CENTER

def add_content_slide(prs, title, content_blocks):
    """Add content slide"""
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    
    # Title
    title_box = slide.shapes.add_textbox(Inches(0.5), Inches(0.4), Inches(9), Inches(0.7))
    title_frame = title_box.text_frame
    title_frame.text = title
    title_para = title_frame.paragraphs[0]
    title_para.font.size = Pt(32)
    title_para.font.bold = True
    title_para.font.color.rgb = RGBColor(44, 62, 80)
    
    return slide

# Slide 1: Title Page
add_title_slide(prs, 
    "RAG Retrieval Optimization & System Enhancement",
    "Complete Optimization Process from Retrieval Hit Rate to Prompt Design",
    "Based on Local RAG System for \"Reminiscences of a Stock Operator\"")

# Slide 2: Core Problems
slide2 = add_content_slide(prs, "Core Problems We Faced", [])
y_pos = 1.3
problems = [
    ("Problem 1: Extremely Low Retrieval Hit Rate", [
        "FAISS dense-only: top-1 = 0",
        "BM25-only: top-1 approaches 0",
        "Many questions cannot find correct pages within top-5"
    ]),
    ("Problem 2: LLM Prone to Hallucination", [
        "Mixing in external \"common sense\" and \"reasoning\"",
        "Fabricates answers instead of saying \"I don't know\""
    ]),
    ("Problem 3: Unreasonable Evaluation Metrics", [
        "Too strict for cross-page summary questions",
        "Test questions too simple, failing to expose defects"
    ])
]

for prob_title, items in problems:
    # Problem title
    box = slide2.shapes.add_textbox(Inches(0.7), Inches(y_pos), Inches(8.5), Inches(0.4))
    frame = box.text_frame
    frame.text = prob_title
    para = frame.paragraphs[0]
    para.font.size = Pt(20)
    para.font.bold = True
    para.font.color.rgb = RGBColor(231, 76, 60)
    
    y_pos += 0.5
    # Problem list
    for item in items:
        box = slide2.shapes.add_textbox(Inches(1.2), Inches(y_pos), Inches(8), Inches(0.3))
        frame = box.text_frame
        frame.text = "• " + item
        para = frame.paragraphs[0]
        para.font.size = Pt(16)
        y_pos += 0.35

# Slide 3: Four Optimization Strategies
slide3 = add_content_slide(prs, "Four Major Optimization Strategies", [])
strategies = [
    ("1. Chunk Parameter Tuning", "1500 -> 800 tokens\noverlap: 200", "Improve info concentration\nReduce noise interference"),
    ("2. Hybrid Retriever", "Dense(0.3) + BM25(0.7)", "Combine semantics + keywords\nBM25 most stable for book QA"),
    ("3. Multi-Query Expansion", "1 question -> 4 rewrites", "Improve recall by 20-40%\nHandle colloquial questions"),
    ("4. Strict Prompt Constraints", "Only use Context\nNo info -> explicitly state", "Suppress hallucination\nUnified fallback")
]

x_positions = [0.5, 5.0]
y_positions = [1.5, 4.0]
idx = 0
for strategy in strategies:
    x = x_positions[idx % 2]
    y = y_positions[idx // 2]
    
    # Strategy box
    box = slide3.shapes.add_textbox(Inches(x), Inches(y), Inches(4.2), Inches(2.2))
    frame = box.text_frame
    
    # Title
    p = frame.paragraphs[0]
    p.text = strategy[0]
    p.font.size = Pt(18)
    p.font.bold = True
    p.font.color.rgb = RGBColor(76, 175, 80)
    
    # Code/Configuration
    p = frame.add_paragraph()
    p.text = strategy[1]
    p.font.size = Pt(14)
    p.font.name = 'Courier New'
    p.space_before = Pt(6)
    
    # Description
    p = frame.add_paragraph()
    p.text = strategy[2]
    p.font.size = Pt(14)
    p.space_before = Pt(6)
    
    idx += 1

# Slide 4: Chunk Optimization Details
slide4 = add_content_slide(prs, "Chunk Parameter Optimization", [])
subtitle_box = slide4.shapes.add_textbox(Inches(0.7), Inches(1.2), Inches(8.5), Inches(0.4))
frame = subtitle_box.text_frame
frame.text = "Why Reduce Chunk Size?"
para = frame.paragraphs[0]
para.font.size = Pt(24)
para.font.bold = True

# Left: Problems with 1500 tokens
left_content = [
    "Problems with 1500 tokens:",
    "• Mixed with irrelevant content",
    "• Dense embedding signals diluted",
    "• High noise after BM25 hits",
    "• Not conducive to precise localization"
]
y = 1.8
for line in left_content:
    box = slide4.shapes.add_textbox(Inches(0.7), Inches(y), Inches(4), Inches(0.3))
    frame = box.text_frame
    frame.text = line
    para = frame.paragraphs[0]
    para.font.size = Pt(16)
    if "Problems" in line:
        para.font.bold = True
        para.font.color.rgb = RGBColor(231, 76, 60)
    y += 0.35

# Right: Advantages of 800 tokens
right_content = [
    "Advantages of 800 tokens:",
    "• More concentrated information",
    "• Directly corresponds to original paragraphs",
    "• 200 overlap ensures continuity",
    "• Suitable for location-based retrieval"
]
y = 1.8
for line in right_content:
    box = slide4.shapes.add_textbox(Inches(5.2), Inches(y), Inches(4), Inches(0.3))
    frame = box.text_frame
    frame.text = line
    para = frame.paragraphs[0]
    para.font.size = Pt(16)
    if "Advantages" in line:
        para.font.bold = True
        para.font.color.rgb = RGBColor(76, 175, 80)
    y += 0.35

# Design philosophy box
design_box = slide4.shapes.add_textbox(Inches(1), Inches(4.5), Inches(8), Inches(1.2))
frame = design_box.text_frame
p = frame.paragraphs[0]
p.text = "Design Philosophy"
p.font.size = Pt(20)
p.font.bold = True
p = frame.add_paragraph()
p.text = 'Book QA is more like "location-based retrieval": expecting retrieval results to directly correspond to one sentence or a small paragraph from the original text, rather than a large mixed chunk.'
p.font.size = Pt(16)
p.space_before = Pt(6)

# Slide 5: Hybrid Retriever
slide5 = add_content_slide(prs, "Hybrid Retriever Design", [])
subtitle_box = slide5.shapes.add_textbox(Inches(0.7), Inches(1.2), Inches(8.5), Inches(0.4))
frame = subtitle_box.text_frame
frame.text = "Comparison of Three Modes"
para = frame.paragraphs[0]
para.font.size = Pt(24)
para.font.bold = True

# Table data
table_data = [
    ["Retrieval Mode", "hit@3", "hit@5", "Characteristics"],
    ["dense-only", "2/10", "2/10", "Hard to pinpoint page numbers"],
    ["bm25-only", "3/10", "4/10", "Most stable for book QA"],
    ["hybrid(0.5, 0.5)", "2/10", "3/10", "Slightly better than dense-only"],
    ["hybrid(0.3, 0.7)", "4/10", "4/10", "Best Hybrid configuration"],
    ["hybrid(0.7, 0.3)", "2/10", "3/10", "Too high dense weight degrades"]
]

# Create table
rows, cols = len(table_data), len(table_data[0])
table = slide5.shapes.add_table(rows, cols, Inches(1), Inches(1.8), Inches(8), Inches(3)).table

for i, row_data in enumerate(table_data):
    for j, cell_data in enumerate(row_data):
        cell = table.cell(i, j)
        cell.text = cell_data
        para = cell.text_frame.paragraphs[0]
        para.font.size = Pt(14)
        if i == 0:  # Header
            para.font.bold = True
            cell.fill.solid()
            cell.fill.fore_color.rgb = RGBColor(102, 126, 234)
            para.font.color.rgb = RGBColor(255, 255, 255)

# Final choice box
choice_box = slide5.shapes.add_textbox(Inches(1), Inches(5.2), Inches(8), Inches(1))
frame = choice_box.text_frame
p = frame.paragraphs[0]
p.text = "Final Choice: Dense(0.3) + BM25(0.7)"
p.font.size = Pt(20)
p.font.bold = True
p.font.color.rgb = RGBColor(33, 150, 243)
p = frame.add_paragraph()
p.text = "In single-book + English text scenarios, higher BM25 weight configuration performs most stably"
p.font.size = Pt(16)

# Slide 6: Multi-Query
slide6 = add_content_slide(prs, "Multi-Query: The Recall Booster", [])

# Working principle
principle_box = slide6.shapes.add_textbox(Inches(0.7), Inches(1.2), Inches(8.5), Inches(2.5))
frame = principle_box.text_frame
p = frame.paragraphs[0]
p.text = "Working Principle"
p.font.size = Pt(22)
p.font.bold = True

example_text = '''
Original Question: "Why did he need the tape to be absolutely current?"

LLM generates 4 rewrites:
1. What was the importance of having real-time tape data?
2. Why was timing critical for the tape reading method?
3. How did tape delays affect trading decisions?
4. What role did current prices play in the strategy?

-> Retrieve separately -> Take union
'''
p = frame.add_paragraph()
p.text = example_text
p.font.size = Pt(14)
p.font.name = 'Courier New'

# Metric boxes
metrics_y = 4.0
# Recall improvement
box1 = slide6.shapes.add_textbox(Inches(1.5), Inches(metrics_y), Inches(3), Inches(1.2))
frame1 = box1.text_frame
p = frame1.paragraphs[0]
p.text = "Recall Improvement"
p.font.size = Pt(16)
p.alignment = PP_ALIGN.CENTER
p = frame1.add_paragraph()
p.text = "20-40%"
p.font.size = Pt(36)
p.font.bold = True
p.alignment = PP_ALIGN.CENTER

# Rewritten queries
box2 = slide6.shapes.add_textbox(Inches(5.5), Inches(metrics_y), Inches(3), Inches(1.2))
frame2 = box2.text_frame
p = frame2.paragraphs[0]
p.text = "Rewritten Queries"
p.font.size = Pt(16)
p.alignment = PP_ALIGN.CENTER
p = frame2.add_paragraph()
p.text = "1->4"
p.font.size = Pt(36)
p.font.bold = True
p.alignment = PP_ALIGN.CENTER

# Slide 7: Prompt Design
slide7 = add_content_slide(prs, "Prompt Rewrite: Suppress Hallucination", [])

# Core design principles
principle_box = slide7.shapes.add_textbox(Inches(0.7), Inches(1.2), Inches(8.5), Inches(1.8))
frame = principle_box.text_frame
p = frame.paragraphs[0]
p.text = "Three Strict Rules"
p.font.size = Pt(22)
p.font.bold = True
p.font.color.rgb = RGBColor(76, 175, 80)

rules = [
    "1. Only use information from Context (prohibit external knowledge)",
    "2. Must explicitly state when information is insufficient",
    "   \"Based on the provided information, I cannot fully answer this question.\"",
    "3. Limit answer to 1-3 sentences (avoid excessive summarization)"
]
for rule in rules:
    p = frame.add_paragraph()
    p.text = rule
    p.font.size = Pt(16)
    p.space_before = Pt(4)

# Effect comparison
subtitle_box = slide7.shapes.add_textbox(Inches(0.7), Inches(3.3), Inches(8.5), Inches(0.4))
frame = subtitle_box.text_frame
frame.text = "Effect Comparison"
para = frame.paragraphs[0]
para.font.size = Pt(22)
para.font.bold = True

# Before optimization
before = [
    "Before Optimization",
    "• Mixed with external knowledge",
    "• Fabricates when can't find answer",
    "• Over-summarizes into \"short essays\""
]
y = 3.8
for line in before:
    box = slide7.shapes.add_textbox(Inches(0.7), Inches(y), Inches(4), Inches(0.3))
    frame = box.text_frame
    frame.text = line
    para = frame.paragraphs[0]
    para.font.size = Pt(16)
    if "Before" in line:
        para.font.bold = True
        para.font.color.rgb = RGBColor(231, 76, 60)
    y += 0.35

# After optimization
after = [
    "After Optimization",
    "• Strictly cites original text",
    "• Explicitly states inability when can't answer",
    "• Concise and precise 1-3 sentence answers"
]
y = 3.8
for line in after:
    box = slide7.shapes.add_textbox(Inches(5.2), Inches(y), Inches(4), Inches(0.3))
    frame = box.text_frame
    frame.text = line
    para = frame.paragraphs[0]
    para.font.size = Pt(16)
    if "After" in line:
        para.font.bold = True
        para.font.color.rgb = RGBColor(76, 175, 80)
    y += 0.35

# Slide 8: Evaluation System
slide8 = add_content_slide(prs, "Evaluation System Improvement", [])

# Two types of test sets
subtitle1 = slide8.shapes.add_textbox(Inches(0.7), Inches(1.2), Inches(8.5), Inches(0.4))
frame = subtitle1.text_frame
frame.text = "Two Types of Test Sets"
para = frame.paragraphs[0]
para.font.size = Pt(22)
para.font.bold = True

# Verbatim questions
box1 = slide8.shapes.add_textbox(Inches(0.7), Inches(1.7), Inches(4), Inches(1.3))
frame1 = box1.text_frame
p = frame1.paragraphs[0]
p.text = "Verbatim Questions"
p.font.size = Pt(18)
p.font.bold = True
p = frame1.add_paragraph()
p.text = "Answers are direct quotes from original text\nExample: \"What is stabilising process?\"\n-> Strict page number matching"
p.font.size = Pt(14)

# Summary questions
box2 = slide8.shapes.add_textbox(Inches(5.2), Inches(1.7), Inches(4), Inches(1.3))
frame2 = box2.text_frame
p = frame2.paragraphs[0]
p.text = "Summary Questions"
p.font.size = Pt(18)
p.font.bold = True
p = frame2.add_paragraph()
p.text = "Require summarizing multi-page content\nExample: \"Why did coffee trading fail?\"\n-> Page number +/-1 window tolerance"
p.font.size = Pt(14)

# Page window strategy
subtitle2 = slide8.shapes.add_textbox(Inches(0.7), Inches(3.3), Inches(8.5), Inches(0.4))
frame = subtitle2.text_frame
frame.text = "Page Window Strategy"
para = frame.paragraphs[0]
para.font.size = Pt(22)
para.font.bold = True

strategy_lines = [
    "• Verbatim: Only accept strict page in gold_pages",
    "• Summary: Accept page in [gold_page +/- 1]",
    "• Rationale: Summary questions have info distributed across 2-3 pages"
]
y = 3.8
for line in strategy_lines:
    box = slide8.shapes.add_textbox(Inches(1), Inches(y), Inches(8), Inches(0.3))
    frame = box.text_frame
    frame.text = line
    para = frame.paragraphs[0]
    para.font.size = Pt(16)
    y += 0.35

# Question complexity evolution
subtitle3 = slide8.shapes.add_textbox(Inches(0.7), Inches(5.0), Inches(8.5), Inches(0.4))
frame = subtitle3.text_frame
frame.text = "Question Complexity Evolution"
para = frame.paragraphs[0]
para.font.size = Pt(22)
para.font.bold = True

complexity_box = slide8.shapes.add_textbox(Inches(1), Inches(5.5), Inches(8), Inches(1))
frame = complexity_box.text_frame
complexity_text = '''Level 1: Single fact -> "When did he go to work?"
Level 2: Single causation -> "Why did he say X?"
Level 3: Combinatorial -> "Explain ancient history + real-time need + how delay breaks timing"'''
p = frame.paragraphs[0]
p.text = complexity_text
p.font.size = Pt(14)
p.font.name = 'Courier New'

# Slide 9: Summary
slide9 = add_content_slide(prs, "Optimization Results Summary", [])

# Metric boxes
# Retrieval hit rate
box1 = slide9.shapes.add_textbox(Inches(1.5), Inches(1.3), Inches(3), Inches(1.2))
frame1 = box1.text_frame
p = frame1.paragraphs[0]
p.text = "Retrieval Hit Rate"
p.font.size = Pt(16)
p.alignment = PP_ALIGN.CENTER
p = frame1.add_paragraph()
p.text = "+40%"
p.font.size = Pt(40)
p.font.bold = True
p.alignment = PP_ALIGN.CENTER
p.font.color.rgb = RGBColor(76, 175, 80)
p = frame1.add_paragraph()
p.text = "Significant hit@3 improvement"
p.font.size = Pt(14)
p.alignment = PP_ALIGN.CENTER

# Hallucination rate
box2 = slide9.shapes.add_textbox(Inches(5.5), Inches(1.3), Inches(3), Inches(1.2))
frame2 = box2.text_frame
p = frame2.paragraphs[0]
p.text = "Hallucination Rate"
p.font.size = Pt(16)
p.alignment = PP_ALIGN.CENTER
p = frame2.add_paragraph()
p.text = "-60%"
p.font.size = Pt(40)
p.font.bold = True
p.alignment = PP_ALIGN.CENTER
p.font.color.rgb = RGBColor(33, 150, 243)
p = frame2.add_paragraph()
p.text = "Significant reduction in fabrication"
p.font.size = Pt(14)
p.alignment = PP_ALIGN.CENTER

# Five core achievements
subtitle = slide9.shapes.add_textbox(Inches(0.7), Inches(2.8), Inches(8.5), Inches(0.4))
frame = subtitle.text_frame
frame.text = "Five Core Optimization Achievements"
para = frame.paragraphs[0]
para.font.size = Pt(24)
para.font.bold = True

achievements = [
    "More Accurate Retrieval: BM25 + Multi-Query -> stable page hit rate",
    "More Complete Retrieval: Chunk reform(800) + Multi-Query -> 20-40% recall boost",
    "Safer Answers: New Prompt eliminates hallucination, only uses Context",
    "More Reasonable Evaluation: Page window solves summary question misjudgment",
    "More Controllable System: Explainable, reproducible, tunable"
]
y = 3.4
for achievement in achievements:
    box = slide9.shapes.add_textbox(Inches(1), Inches(y), Inches(8), Inches(0.35))
    frame = box.text_frame
    frame.text = "• " + achievement
    para = frame.paragraphs[0]
    para.font.size = Pt(16)
    para.line_spacing = 1.3
    y += 0.45

# Final summary box
final_box = slide9.shapes.add_textbox(Inches(1), Inches(5.8), Inches(8), Inches(1))
frame = final_box.text_frame
p = frame.paragraphs[0]
p.text = 'Complete Optimization Path from "Working" to "Working Well"'
p.font.size = Pt(22)
p.font.bold = True
p.alignment = PP_ALIGN.CENTER
p.font.color.rgb = RGBColor(102, 126, 234)
p = frame.add_paragraph()
p.text = "Precision UP + Recall UP + Safety UP + Evaluation Scientificity UP"
p.font.size = Pt(16)
p.alignment = PP_ALIGN.CENTER
p.space_before = Pt(8)

# Save presentation
prs.save('RAG_Optimization_Presentation.pptx')
print("PPT file successfully created: RAG_Optimization_Presentation.pptx")
print(f"Contains {len(prs.slides)} slides")
