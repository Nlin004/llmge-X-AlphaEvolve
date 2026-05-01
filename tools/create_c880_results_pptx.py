from __future__ import annotations

import html
import os
import zipfile
from pathlib import Path


OUT = Path("C880_Evolution_Run_Results.pptx")

SLIDE_W = 13_333_333
SLIDE_H = 7_500_000

BG = "0B1020"
FG = "E8F0FF"
MUTED = "9CA3AF"
CYAN = "5EEAD4"
AMBER = "F59E0B"
RED = "EF4444"
SLATE = "1E293B"


slides = [
    {
        "title": "C880 Evolution Run Results",
        "subtitle": "LLM-guided circuit evolution: what worked, what failed, and what changed",
        "bullets": [
            "Target: ISCAS-85 C880 module c880_impl with fixed scalar port interface.",
            "Goal: reduce area and delay while preserving functional correctness.",
            "Key outcome: pipeline now evaluates valid individuals, but LLM generation still times out.",
        ],
        "tag": "Engineering readout",
    },
    {
        "title": "Evaluation Metrics",
        "subtitle": "Every candidate is scored as a fitness tuple",
        "cards": [
            ("Correctness error", "0.0 is best", "Mismatch rate over C880 test vectors."),
            ("Area", "Lower is better", "Synthesis/evaluator gate-area proxy."),
            ("Delay", "Lower is better", "Critical-path delay proxy."),
        ],
        "bullets": [
            "Fitness weights minimize all three objectives.",
            "Partial correctness scoring means failed logic can still be ranked by error rate.",
        ],
    },
    {
        "title": "Initial Failure Mode",
        "subtitle": "The first runs collapsed before evolution could use the population",
        "bullets": [
            "Slurm submitted LLM jobs, but LLM outputs errored before model_<gene>.py files were written.",
            "The controller skipped evaluation when model files were missing.",
            "Population fitness stayed invalid/infinite, then invalid removal collapsed the population.",
        ],
        "callout": "Symptom: Model file does not exist for gene_id, skipping evaluation.",
    },
    {
        "title": "Stabilizing Changes",
        "subtitle": "Changes that converted hard failures into evaluable candidates",
        "cards": [
            ("Partial scoring", "Rank by error rate instead of discarding all failures.", "CYAN"),
            ("Keep model files", "Evaluate generated files even when logs contain warnings.", "CYAN"),
            ("C880 contract checks", "Reject bus/renamed-port variants early.", "AMBER"),
            ("Local fallback", "Apply safe Boolean refactors when the LLM times out.", "AMBER"),
        ],
    },
    {
        "title": "Latest End-To-End Run",
        "subtitle": "c880_partial5 created and evaluated a full population of 8",
        "metrics": [
            ("Population", "8 / 8", "evaluated"),
            ("Correctness", "0.0", "error rate for all 8"),
            ("Best area", "553", "range 553-556"),
            ("Delay", "385", "all observed"),
        ],
        "bullets": [
            "All eight seed individuals produced valid model files.",
            "All eight evaluated as functionally correct under the current evaluator.",
            "Selection favored the lowest-area individuals: area 553 and 554.",
        ],
    },
    {
        "title": "Representative Fitness Results",
        "subtitle": "Observed population after invalid removal",
        "table": [
            ("Gene", "Error", "Area", "Delay"),
            ("xXxia0LY...", "0.0", "553", "385"),
            ("xXxAn5V...", "0.0", "553", "385"),
            ("xXxUb7m...", "0.0", "554", "385"),
            ("xXxkWZ3...", "0.0", "554", "385"),
            ("xXx0pWm...", "0.0", "555", "385"),
            ("xXxyuvj...", "0.0", "556", "385"),
        ],
        "bullets": [
            "Area spread is small because local fallback refactors are conservative.",
            "Delay did not move in this observed population.",
        ],
    },
    {
        "title": "Important Caveat",
        "subtitle": "The run is alive, but not yet truly LLM-driven",
        "bullets": [
            "Every observed LLM generation request timed out before returning a usable individual.",
            "The successful variants came from local correctness-preserving C880 fallback refactors.",
            "So the system currently behaves like a hybrid: local Boolean refactor search plus LLM attempts.",
        ],
        "callout": "Current milestone reached: evaluable evolution. Next milestone: LLM-authored variants.",
    },
    {
        "title": "Why The LLM Timed Out",
        "subtitle": "The bottleneck is serving + prompt size, not the evaluator",
        "bullets": [
            "The client reaches the server hostname, so requests are being sent.",
            "The server can answer tiny test prompts, but C880 prompts exceed practical latency.",
            "Full C880 prompts include hundreds of assign statements, creating high prefill and generation cost.",
            "Llama-70B on constrained L40S-class GPUs leaves little memory and serializes requests with batch size 1.",
        ],
        "callout": "Increasing batch size would likely worsen OOM risk; batch size 1 is safer but slow.",
    },
    {
        "title": "Next Experiment",
        "subtitle": "Make LLM mutation small enough to complete",
        "bullets": [
            "Use cluster prompts: send only 12-24 assign statements to the LLM.",
            "Ask the LLM to return assign statements only, preserving left-hand-side signal order.",
            "Splice the returned cluster into the full generate_seed_verilog block.",
            "Keep local fallback as a safety net, but track whether variants are LLM-authored or fallback-authored.",
        ],
        "tag": "Success criterion: at least one C880 CLUSTER FROM LLM result.",
    },
    {
        "title": "Bottom Line",
        "subtitle": "What we learned",
        "bullets": [
            "The original empty-population failure mode is fixed.",
            "The evaluator now produces useful fitness tuples for valid C880 candidates.",
            "The latest run proves the loop can create, evaluate, select, and mate candidates.",
            "The next unlock is shrinking LLM prompts or using a faster/smaller/better-served model.",
        ],
        "tag": "Evaluable now. LLM-authored next.",
    },
]


def esc(value: str) -> str:
    return html.escape(value, quote=True)


def tx_body(text: str, size: int = 2200, color: str = FG, bold: bool = False) -> str:
    b = '<a:b/>' if bold else ''
    return (
        f'<a:p><a:r><a:rPr lang="en-US" sz="{size}" dirty="0">{b}'
        f'<a:solidFill><a:srgbClr val="{color}"/></a:solidFill></a:rPr>'
        f'<a:t>{esc(text)}</a:t></a:r><a:endParaRPr lang="en-US" sz="{size}"/></a:p>'
    )


def shape_text(x, y, w, h, text, size=2200, color=FG, bold=False, fill=None, line=None, radius=False):
    fill_xml = (
        f'<a:solidFill><a:srgbClr val="{fill}"/></a:solidFill>'
        if fill
        else '<a:noFill/>'
    )
    line_xml = (
        f'<a:ln><a:solidFill><a:srgbClr val="{line}"/></a:solidFill></a:ln>'
        if line
        else '<a:ln><a:noFill/></a:ln>'
    )
    prst = "roundRect" if radius else "rect"
    return f"""
    <p:sp>
      <p:nvSpPr><p:cNvPr id="{shape_text.next_id}" name="TextBox {shape_text.next_id}"/><p:cNvSpPr txBox="1"/><p:nvPr/></p:nvSpPr>
      <p:spPr>
        <a:xfrm><a:off x="{x}" y="{y}"/><a:ext cx="{w}" cy="{h}"/></a:xfrm>
        <a:prstGeom prst="{prst}"><a:avLst/></a:prstGeom>
        {fill_xml}{line_xml}
      </p:spPr>
      <p:txBody><a:bodyPr wrap="square" lIns="160000" tIns="90000" rIns="160000" bIns="90000"/><a:lstStyle/>
        {tx_body(text, size, color, bold)}
      </p:txBody>
    </p:sp>
    """


shape_text.next_id = 10


def bullet_lines(items, x=900_000, y=2_000_000, w=10_900_000, line_h=520_000):
    xml = []
    for idx, item in enumerate(items):
        xml.append(shape_text(x, y + idx * line_h, w, 430_000, f"- {item}", 1800, FG))
        shape_text.next_id += 1
    return "\n".join(xml)


def slide_xml(slide, idx):
    shape_text.next_id = 10
    items = []
    items.append(shape_text(0, 0, SLIDE_W, SLIDE_H, "", fill=BG))
    shape_text.next_id += 1
    items.append(shape_text(700_000, 430_000, 8_900_000, 720_000, slide["title"], 3600, FG, True))
    shape_text.next_id += 1
    items.append(shape_text(730_000, 1_150_000, 10_700_000, 460_000, slide.get("subtitle", ""), 1650, MUTED))
    shape_text.next_id += 1
    items.append(shape_text(10_800_000, 450_000, 1_700_000, 380_000, f"{idx:02d}", 2000, CYAN, True, SLATE, CYAN, True))
    shape_text.next_id += 1

    if "bullets" in slide:
        items.append(bullet_lines(slide["bullets"]))

    if "cards" in slide:
        cards = slide["cards"]
        card_w = 2_850_000 if len(cards) >= 4 else 3_500_000
        start_x = 780_000
        for i, card in enumerate(cards):
            title, stat, desc = card[:3]
            accent = CYAN if len(card) < 4 or card[3] == "CYAN" else AMBER
            x = start_x + i * (card_w + 220_000)
            y = 2_130_000
            items.append(shape_text(x, y, card_w, 1_600_000, "", fill=SLATE, line=accent, radius=True))
            shape_text.next_id += 1
            items.append(shape_text(x + 120_000, y + 120_000, card_w - 240_000, 320_000, title, 1700, accent, True))
            shape_text.next_id += 1
            items.append(shape_text(x + 120_000, y + 510_000, card_w - 240_000, 330_000, stat, 1450, FG, True))
            shape_text.next_id += 1
            items.append(shape_text(x + 120_000, y + 900_000, card_w - 240_000, 530_000, desc, 1250, MUTED))
            shape_text.next_id += 1

    if "metrics" in slide:
        for i, (label, value, desc) in enumerate(slide["metrics"]):
            x = 800_000 + i * 3_050_000
            y = 2_050_000
            items.append(shape_text(x, y, 2_700_000, 1_650_000, "", fill=SLATE, line=CYAN, radius=True))
            shape_text.next_id += 1
            items.append(shape_text(x + 140_000, y + 160_000, 2_420_000, 290_000, label, 1500, MUTED))
            shape_text.next_id += 1
            items.append(shape_text(x + 140_000, y + 520_000, 2_420_000, 520_000, value, 3200, CYAN, True))
            shape_text.next_id += 1
            items.append(shape_text(x + 140_000, y + 1_100_000, 2_420_000, 320_000, desc, 1350, FG))
            shape_text.next_id += 1

    if "table" in slide:
        rows = slide["table"]
        x0, y0 = 850_000, 2_050_000
        widths = [3_300_000, 1_700_000, 1_700_000, 1_700_000]
        row_h = 520_000
        for r, row in enumerate(rows):
            x = x0
            for c, cell in enumerate(row):
                fill = "111827" if r == 0 else SLATE
                color = CYAN if r == 0 else FG
                items.append(shape_text(x, y0 + r * row_h, widths[c], row_h, cell, 1350, color, r == 0, fill, "334155"))
                shape_text.next_id += 1
                x += widths[c]

    if "callout" in slide:
        items.append(shape_text(760_000, 6_350_000, 11_800_000, 520_000, slide["callout"], 1500, AMBER, True, "1F2937", AMBER, True))
        shape_text.next_id += 1

    if "tag" in slide:
        items.append(shape_text(8_450_000, 6_470_000, 3_850_000, 410_000, slide["tag"], 1250, CYAN, True, "132027", CYAN, True))
        shape_text.next_id += 1

    return f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<p:sld xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main"
       xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"
       xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main">
  <p:cSld><p:spTree>
    <p:nvGrpSpPr><p:cNvPr id="1" name=""/><p:cNvGrpSpPr/><p:nvPr/></p:nvGrpSpPr>
    <p:grpSpPr><a:xfrm><a:off x="0" y="0"/><a:ext cx="0" cy="0"/><a:chOff x="0" y="0"/><a:chExt cx="0" cy="0"/></a:xfrm></p:grpSpPr>
    {''.join(items)}
  </p:spTree></p:cSld>
  <p:clrMapOvr><a:masterClrMapping/></p:clrMapOvr>
</p:sld>"""


def rels_xml():
    rels = []
    for i in range(1, len(slides) + 1):
        rels.append(
            f'<Relationship Id="rId{i}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slide" Target="slides/slide{i}.xml"/>'
        )
    rels.append(
        f'<Relationship Id="rId{len(slides)+1}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/theme" Target="theme/theme1.xml"/>'
    )
    return f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">{''.join(rels)}</Relationships>"""


def presentation_xml():
    sld_ids = []
    for i in range(1, len(slides) + 1):
        sld_ids.append(f'<p:sldId id="{255+i}" r:id="rId{i}"/>')
    return f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<p:presentation xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main"
 xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"
 xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main">
 <p:sldMasterIdLst/>
 <p:sldIdLst>{''.join(sld_ids)}</p:sldIdLst>
 <p:sldSz cx="{SLIDE_W}" cy="{SLIDE_H}" type="wide"/>
 <p:notesSz cx="6858000" cy="9144000"/>
</p:presentation>"""


CONTENT_TYPES = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
 <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
 <Default Extension="xml" ContentType="application/xml"/>
 <Override PartName="/ppt/presentation.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.presentation.main+xml"/>
 <Override PartName="/ppt/theme/theme1.xml" ContentType="application/vnd.openxmlformats-officedocument.theme+xml"/>
""" + "".join(
    f' <Override PartName="/ppt/slides/slide{i}.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.slide+xml"/>\n'
    for i in range(1, len(slides) + 1)
) + "</Types>"


ROOT_RELS = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
 <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="ppt/presentation.xml"/>
</Relationships>"""


THEME = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<a:theme xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" name="C880 Results">
 <a:themeElements>
  <a:clrScheme name="C880">
   <a:dk1><a:srgbClr val="0B1020"/></a:dk1><a:lt1><a:srgbClr val="E8F0FF"/></a:lt1>
   <a:dk2><a:srgbClr val="1E293B"/></a:dk2><a:lt2><a:srgbClr val="FFFFFF"/></a:lt2>
   <a:accent1><a:srgbClr val="5EEAD4"/></a:accent1><a:accent2><a:srgbClr val="F59E0B"/></a:accent2>
   <a:accent3><a:srgbClr val="EF4444"/></a:accent3><a:accent4><a:srgbClr val="64748B"/></a:accent4>
   <a:accent5><a:srgbClr val="94A3B8"/></a:accent5><a:accent6><a:srgbClr val="CBD5E1"/></a:accent6>
   <a:hlink><a:srgbClr val="5EEAD4"/></a:hlink><a:folHlink><a:srgbClr val="F59E0B"/></a:folHlink>
  </a:clrScheme>
  <a:fontScheme name="Aptos"><a:majorFont><a:latin typeface="Aptos Display"/></a:majorFont><a:minorFont><a:latin typeface="Aptos"/></a:minorFont></a:fontScheme>
  <a:fmtScheme name="C880"><a:fillStyleLst/><a:lnStyleLst/><a:effectStyleLst/><a:bgFillStyleLst/></a:fmtScheme>
 </a:themeElements>
</a:theme>"""


def main() -> None:
    if OUT.exists():
        OUT.unlink()
    with zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("[Content_Types].xml", CONTENT_TYPES)
        z.writestr("_rels/.rels", ROOT_RELS)
        z.writestr("ppt/presentation.xml", presentation_xml())
        z.writestr("ppt/_rels/presentation.xml.rels", rels_xml())
        z.writestr("ppt/theme/theme1.xml", THEME)
        for i, slide in enumerate(slides, start=1):
            z.writestr(f"ppt/slides/slide{i}.xml", slide_xml(slide, i))
            z.writestr(
                f"ppt/slides/_rels/slide{i}.xml.rels",
                '<?xml version="1.0" encoding="UTF-8" standalone="yes"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"/>',
            )
    print(OUT.resolve())


if __name__ == "__main__":
    main()
