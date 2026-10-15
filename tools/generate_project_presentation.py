"""Generate the editable ShopSense project presentation using only Python stdlib."""

from __future__ import annotations

from html import escape
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "ShopSense_Project_Presentation.pptx"
EMU = 914400
SLIDE_W = 13_333_333
SLIDE_H = 7_500_000

NAVY = "0A1A22"
PANEL = "122A34"
PANEL_LIGHT = "193640"
MINT = "70E1BD"
MINT_DARK = "1A705C"
CREAM = "F5F3EA"
MUTED = "A7BBC0"
ORANGE = "FFB86B"
RED = "F27D75"
BLUE = "86B9FF"
LINE = "29434C"
FONT = "Aptos"


def inch(value: float) -> int:
    return round(value * EMU)


class Slide:
    def __init__(self, number: int, title: str = "", kicker: str = "") -> None:
        self.number = number
        self._shape_id = 2
        self.objects: list[str] = []
        if title:
            self.rect(0, 0, 13.333, 7.5, NAVY)
            self.rect(0, 0, 0.12, 7.5, MINT)
            self.text(kicker.upper(), 0.65, 0.36, 8.5, 0.28, 10, MINT, True)
            self.text(title, 0.65, 0.72, 12, 0.62, 27, CREAM, True)
            self.rect(0.65, 1.49, 12.0, 0.012, LINE)
            self.text(
                "SHOPSENSE  /  PROJECT PRESENTATION",
                0.65,
                7.13,
                7.0,
                0.18,
                8,
                MUTED,
                True,
            )
            self.text(f"{number:02d}", 12.15, 7.10, 0.5, 0.22, 9, MINT, True, "r")

    def rect(
        self,
        x: float,
        y: float,
        w: float,
        h: float,
        fill: str,
        *,
        rounded: bool = False,
        line: str | None = None,
        radius: str | None = None,
    ) -> None:
        del radius
        sid = self._shape_id
        self._shape_id += 1
        geometry = "roundRect" if rounded else "rect"
        stroke = (
            f'<a:ln w="12700"><a:solidFill><a:srgbClr val="{line}"/>'
            "</a:solidFill></a:ln>"
            if line
            else "<a:ln><a:noFill/></a:ln>"
        )
        self.objects.append(
            f'<p:sp><p:nvSpPr><p:cNvPr id="{sid}" name="Shape {sid}"/>'
            "<p:cNvSpPr/><p:nvPr/></p:nvSpPr>"
            f'<p:spPr><a:xfrm><a:off x="{inch(x)}" y="{inch(y)}"/>'
            f'<a:ext cx="{inch(w)}" cy="{inch(h)}"/></a:xfrm>'
            f'<a:prstGeom prst="{geometry}"><a:avLst/></a:prstGeom>'
            f'<a:solidFill><a:srgbClr val="{fill}"/></a:solidFill>{stroke}</p:spPr></p:sp>'
        )

    def oval(self, x: float, y: float, w: float, h: float, fill: str) -> None:
        sid = self._shape_id
        self._shape_id += 1
        self.objects.append(
            f'<p:sp><p:nvSpPr><p:cNvPr id="{sid}" name="Circle {sid}"/>'
            "<p:cNvSpPr/><p:nvPr/></p:nvSpPr>"
            f'<p:spPr><a:xfrm><a:off x="{inch(x)}" y="{inch(y)}"/>'
            f'<a:ext cx="{inch(w)}" cy="{inch(h)}"/></a:xfrm>'
            '<a:prstGeom prst="ellipse"><a:avLst/></a:prstGeom>'
            f'<a:solidFill><a:srgbClr val="{fill}"/></a:solidFill>'
            "<a:ln><a:noFill/></a:ln></p:spPr></p:sp>"
        )

    def text(
        self,
        value: str,
        x: float,
        y: float,
        w: float,
        h: float,
        size: int,
        color: str = CREAM,
        bold: bool = False,
        align: str = "l",
        *,
        margin: float = 0.04,
        valign: str = "t",
        italic: bool = False,
    ) -> None:
        sid = self._shape_id
        self._shape_id += 1
        paragraphs: list[str] = []
        for line in value.split("\n"):
            run = (
                f'<a:r><a:rPr lang="en-US" sz="{size * 100}"'
                f' b="{1 if bold else 0}" i="{1 if italic else 0}" dirty="0">'
                f'<a:solidFill><a:srgbClr val="{color}"/></a:solidFill>'
                f'<a:latin typeface="{FONT}"/></a:rPr><a:t xml:space="preserve">'
                f"{escape(line)}</a:t></a:r>"
            )
            paragraphs.append(
                f'<a:p><a:pPr algn="{align}"><a:lnSpc><a:spcPct val="110000"/>'
                f"</a:lnSpc></a:pPr>{run}<a:endParaRPr lang=\"en-US\" sz=\"{size * 100}\"/>"
                "</a:p>"
            )
        body = (
            f'<a:bodyPr wrap="square" lIns="{inch(margin)}" rIns="{inch(margin)}"'
            f' tIns="{inch(margin)}" bIns="{inch(margin)}" anchor="{valign}">'
            "</a:bodyPr><a:lstStyle/>" + "".join(paragraphs)
        )
        self.objects.append(
            f'<p:sp><p:nvSpPr><p:cNvPr id="{sid}" name="Text {sid}"/>'
            "<p:cNvSpPr txBox=\"1\"/><p:nvPr/></p:nvSpPr>"
            f'<p:spPr><a:xfrm><a:off x="{inch(x)}" y="{inch(y)}"/>'
            f'<a:ext cx="{inch(w)}" cy="{inch(h)}"/></a:xfrm>'
            '<a:prstGeom prst="rect"><a:avLst/></a:prstGeom>'
            "<a:noFill/><a:ln><a:noFill/></a:ln></p:spPr>"
            f"<p:txBody>{body}</p:txBody></p:sp>"
        )

    def card(
        self,
        x: float,
        y: float,
        w: float,
        h: float,
        heading: str,
        body: str,
        *,
        accent: str = MINT,
        heading_size: int = 16,
        body_size: int = 12,
    ) -> None:
        self.rect(x, y, w, h, PANEL, rounded=True, line=LINE)
        self.rect(x, y, 0.055, h, accent, rounded=True)
        self.text(heading, x + 0.2, y + 0.18, w - 0.38, 0.38, heading_size, CREAM, True)
        self.text(
            body,
            x + 0.2,
            y + 0.67,
            w - 0.4,
            h - 0.82,
            body_size,
            MUTED,
        )

    def bullet(self, x: float, y: float, body: str, *, color: str = MINT, w: float = 5.2,
               size: int = 13) -> None:
        self.oval(x, y + 0.095, 0.10, 0.10, color)
        self.text(body, x + 0.22, y, w - 0.22, 0.48, size, CREAM)

    def xml(self) -> str:
        tree = (
            '<p:spTree><p:nvGrpSpPr><p:cNvPr id="1" name=""/>'
            "<p:cNvGrpSpPr/><p:nvPr/></p:nvGrpSpPr>"
            "<p:grpSpPr><a:xfrm><a:off x=\"0\" y=\"0\"/><a:ext cx=\"0\" cy=\"0\"/>"
            "<a:chOff x=\"0\" y=\"0\"/><a:chExt cx=\"0\" cy=\"0\"/></a:xfrm></p:grpSpPr>"
            + "".join(self.objects)
            + "</p:spTree>"
        )
        return (
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<p:sld xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" '
            'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships" '
            'xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main">'
            f'<p:cSld name="Slide {self.number}"><p:bg><p:bgPr><a:solidFill>'
            f'<a:srgbClr val="{NAVY}"/></a:solidFill><a:effectLst/></p:bgPr></p:bg>'
            f"{tree}</p:cSld><p:clrMapOvr><a:masterClrMapping/></p:clrMapOvr></p:sld>"
        )


def add_cover(slide: Slide) -> None:
    slide.rect(0, 0, 13.333, 7.5, NAVY)
    slide.rect(0, 0, 0.14, 7.5, MINT)
    slide.text("CS 5903  /  PROJECT PRESENTATION", 0.8, 0.65, 5.5, 0.3, 11, MINT, True)
    slide.text("ShopSense", 0.78, 1.55, 7.0, 1.05, 46, CREAM, True)
    slide.text(
        "Grounded AI for shopping support",
        0.82,
        2.72,
        7.2,
        0.58,
        24,
        MINT,
        True,
    )
    slide.text(
        "A local-first assistant that connects natural conversation to verified "
        "order data and store policy.",
        0.84,
        3.56,
        6.3,
        1.0,
        17,
        MUTED,
    )
    slide.rect(0.84, 5.38, 1.72, 0.42, PANEL_LIGHT, rounded=True)
    slide.text("PROTOTYPE", 0.94, 5.47, 1.52, 0.18, 9, MINT, True, "ctr")
    slide.text("FastAPI  ·  LangGraph  ·  Ollama  ·  Qdrant  ·  Chainlit", 0.84, 6.12,
               6.8, 0.35, 11, MUTED)
    # Chat-card visual, assembled from editable shapes.
    slide.rect(8.25, 1.0, 4.2, 5.65, PANEL, rounded=True, line=LINE)
    slide.text("SHOP SENSE  /  SUPPORT", 8.58, 1.32, 3.4, 0.25, 9, MUTED, True)
    slide.rect(8.6, 2.0, 3.27, 0.76, PANEL_LIGHT, rounded=True)
    slide.text("Can I return my order?", 8.82, 2.22, 2.85, 0.25, 13, CREAM)
    slide.rect(9.16, 3.14, 2.93, 1.43, MINT_DARK, rounded=True)
    slide.text("Your shirt is within the return\nwindow and recorded as unused\nwith tags attached.", 9.38, 3.39,
               2.55, 0.85, 12, CREAM)
    slide.text("Would you like to proceed?", 9.38, 4.37, 2.55, 0.25, 11, MINT, True)
    slide.rect(8.58, 5.35, 3.50, 0.012, LINE)
    slide.oval(8.63, 5.74, 0.22, 0.22, MINT)
    slide.text("Policy-grounded  ·  Order-aware", 8.97, 5.72, 3.15, 0.26, 10, MUTED, True)
    slide.text("01", 12.15, 7.10, 0.5, 0.22, 9, MINT, True, "r")


def build_slides() -> list[Slide]:
    slides: list[Slide] = []
    cover = Slide(1)
    add_cover(cover)
    slides.append(cover)

    s = Slide(2, "The support gap", "01  /  Problem & motivation")
    s.text("Customer questions are simple.\nThe context behind them is not.", 0.78, 1.8,
           5.5, 0.95, 24, CREAM, True)
    s.text("A useful support assistant must understand intent, remember the right context, "
           "and answer from trustworthy facts.", 0.8, 3.0, 5.15, 0.88, 15, MUTED)
    s.card(6.45, 1.8, 2.7, 1.8, "Template replies",
           "Rigid answers miss tone and often feel unhelpful.", accent=ORANGE)
    s.card(9.45, 1.8, 2.7, 1.8, "Lost context",
           "A follow-up like “this order” needs conversation-level memory.", accent=RED)
    s.card(6.45, 4.0, 2.7, 1.8, "Unverified claims",
           "Generated answers can misstate an order or policy.", accent=RED)
    s.card(9.45, 4.0, 2.7, 1.8, "Disconnected tools",
           "Policy, order lookup, and refunds need coordinated handling.", accent=BLUE)
    s.text("Project goal", 0.8, 4.55, 2.0, 0.3, 11, MINT, True)
    s.text("Natural answers, grounded in policy and account data.", 0.8, 4.98, 5.0, 0.55,
           18, CREAM, True)
    slides.append(s)

    s = Slide(3, "What ShopSense is designed to do", "02  /  Goals")
    goals = [
        ("01", "Understand", "Use an LLM intent agent to classify the customer’s request from conversation context."),
        ("02", "Ground", "Retrieve policy passages and verify order facts before response generation."),
        ("03", "Remember", "Carry thread context forward; tag each turn with user, thread, session, and turn IDs."),
        ("04", "Respond", "Generate concise customer-facing answers and stream them to the chat UI."),
    ]
    for i, (number, heading, body) in enumerate(goals):
        x = 0.78 + (i % 2) * 6.05
        y = 1.88 + (i // 2) * 2.25
        s.rect(x, y, 5.65, 1.8, PANEL, rounded=True, line=LINE)
        s.oval(x + 0.22, y + 0.24, 0.52, 0.52, MINT_DARK)
        s.text(number, x + 0.27, y + 0.39, 0.42, 0.18, 10, MINT, True, "ctr")
        s.text(heading, x + 0.95, y + 0.24, 4.2, 0.33, 17, CREAM, True)
        s.text(body, x + 0.95, y + 0.74, 4.35, 0.72, 12, MUTED)
    s.text("Design principle", 0.8, 6.48, 1.7, 0.25, 10, MINT, True)
    s.text("The model writes the answer; verified systems supply the facts.", 2.22, 6.41,
           8.8, 0.35, 16, CREAM, True)
    slides.append(s)

    s = Slide(4, "System architecture", "03  /  Proposed approach")
    # Client and gateway.
    s.card(0.75, 2.02, 2.15, 1.45, "Chainlit UI",
           "Demo customer\nStreaming chat", accent=MINT, heading_size=14, body_size=11)
    s.rect(3.12, 2.52, 0.48, 0.12, MINT)
    s.text("SSE", 3.09, 2.18, 0.6, 0.22, 8, MUTED, True, "ctr")
    s.card(3.78, 2.02, 2.15, 1.45, "FastAPI",
           "Manual validation\nLLM safety gate", accent=ORANGE, heading_size=14, body_size=11)
    s.rect(6.16, 2.52, 0.48, 0.12, MINT)
    s.card(6.82, 2.02, 2.15, 1.45, "LangGraph",
           "Checkpointed state\nIntent triage", accent=BLUE, heading_size=14, body_size=11)
    s.rect(9.2, 2.52, 0.48, 0.12, MINT)
    s.card(9.86, 2.02, 2.55, 1.45, "Specialists",
           "Care · Order · Refund", accent=MINT, heading_size=14, body_size=11)
    # Tool layer.
    s.card(1.22, 4.35, 3.05, 1.48, "Policy RAG",
           "Qdrant + local embeddings\nTXT · CSV · PDF sources", accent=BLUE, body_size=11)
    s.card(5.15, 4.35, 3.05, 1.48, "Order + refund",
           "SQLite demo orders\nEligibility + explicit approval", accent=ORANGE, body_size=11)
    s.card(9.05, 4.35, 3.05, 1.48, "Ollama",
           "Local model for intent,\nsafety, and answer generation", accent=MINT, body_size=11)
    s.text("Grounded facts", 4.20, 4.88, 0.9, 0.22, 8, MUTED, True, "ctr")
    s.text("Verified data", 8.20, 4.88, 0.8, 0.22, 8, MUTED, True, "ctr")
    s.text("Data", 0.83, 6.24, 0.6, 0.22, 9, MINT, True)
    s.text("30 seeded orders / 3 demo customers  ·  policy corpus: TXT, CSV, PDF  ·  "
           "checkpoints + per-user memory", 1.55, 6.18, 10.7, 0.32, 11, CREAM)
    slides.append(s)

    s = Slide(5, "A turn from request to response", "04  /  Runtime flow")
    steps = [
        ("01", "Check", "Manual input rules\n+ LLM safety gate"),
        ("02", "Load", "Checkpointed thread\n+ memory summary"),
        ("03", "Triage", "LLM intent + relevant\norder ID extraction"),
        ("04", "Ground", "Policy retrieval and/or\nowned-order lookup"),
        ("05", "Answer", "LLM composes reply;\nSSE streams answer"),
    ]
    for i, (number, heading, body) in enumerate(steps):
        x = 0.74 + i * 2.52
        s.rect(x, 2.16, 2.18, 2.14, PANEL, rounded=True, line=LINE)
        s.oval(x + 0.18, 2.37, 0.48, 0.48, MINT_DARK)
        s.text(number, x + 0.22, 2.52, 0.4, 0.16, 9, MINT, True, "ctr")
        s.text(heading, x + 0.18, 3.05, 1.82, 0.34, 16, CREAM, True)
        s.text(body, x + 0.18, 3.54, 1.82, 0.54, 10, MUTED)
        if i < len(steps) - 1:
            s.rect(x + 2.22, 3.16, 0.23, 0.08, MINT)
    s.rect(0.8, 4.88, 11.75, 0.85, PANEL_LIGHT, rounded=True)
    s.text("Answer generation is downstream of evidence retrieval; the LLM is not the "
           "authority for order status or refund eligibility.", 1.05, 5.12, 11.2, 0.35,
           14, CREAM, True, "ctr")
    s.text("Streaming endpoint: POST /v1/chat/stream  ·  regular JSON endpoint also available",
           1.0, 6.15, 11.1, 0.3, 11, MUTED, False, "ctr")
    slides.append(s)

    s = Slide(6, "Evidence before explanation", "05  /  Example: order-specific return")
    s.text("Demo order", 0.82, 1.83, 2.0, 0.25, 10, MINT, True)
    s.text("ORD-1101  ·  Cotton shirt (Customer 1)", 0.82, 2.18, 6.2, 0.44, 20, CREAM, True)
    s.text("Delivered 2026-10-10  ·  Demo date: 2026-10-15", 0.82, 2.75, 5.8, 0.3,
           12, MUTED)
    s.card(0.82, 3.4, 3.45, 1.55, "Order facts",
           "Delivered; unused; tags attached;\nnot marked final sale.", accent=BLUE,
           heading_size=14, body_size=11)
    s.card(4.68, 3.4, 3.45, 1.55, "Retrieved policy",
           "30 days after delivery; unused;\noriginal condition and tags.", accent=ORANGE,
           heading_size=14, body_size=11)
    s.rect(8.82, 3.4, 3.55, 1.55, MINT_DARK, rounded=True)
    s.text("Decision", 9.08, 3.62, 2.9, 0.32, 14, CREAM, True)
    s.text("Eligible to request a return", 9.08, 4.1, 3.0, 0.5, 17, MINT, True)
    s.rect(0.82, 5.35, 11.55, 0.74, PANEL, rounded=True)
    s.text("The response explains why and asks whether to proceed. It does not claim a "
           "return or refund has been submitted or approved.", 1.05, 5.55, 11.05, 0.35,
           13, CREAM, True, "ctr")
    s.text("Policy source shown to the user: retrieved policy document; order facts are "
           "looked up against the authenticated demo customer.", 0.9, 6.36, 11.3, 0.3,
           10, MUTED, False, "ctr")
    slides.append(s)

    s = Slide(7, "Prototype demonstration", "06  /  Suggested live walkthrough")
    s.rect(0.78, 1.82, 7.1, 4.9, PANEL, rounded=True, line=LINE)
    s.text("CHAT  /  CUSTOMER 1", 1.08, 2.08, 4.0, 0.25, 9, MINT, True)
    bubbles = [
        ("USER", "I want to get detail about my order", False),
        ("ASSISTANT", "Which order do you mean? Please share the order ID.", True),
        ("USER", "ORD-1101", False),
        ("ASSISTANT", "Cotton shirt; delivered on 2026-10-10.", True),
        ("USER", "What is the return policy for this order?", False),
        ("ASSISTANT", "Eligible based on the order and policy. Want to proceed?", True),
    ]
    y = 2.5
    for role, message, assistant in bubbles:
        x, width = (1.08, 5.7) if assistant else (2.66, 4.8)
        color = MINT_DARK if assistant else PANEL_LIGHT
        slide_text = CREAM if assistant else CREAM
        s.text(role, x, y, width, 0.19, 8, MINT if assistant else MUTED, True)
        s.rect(x, y + 0.23, width, 0.46, color, rounded=True)
        s.text(message, x + 0.14, y + 0.35, width - 0.28, 0.19, 10, slide_text)
        y += 0.68
    s.text("Thread ID: 12-char key   ·   Streaming response", 1.1, 6.34, 6.3, 0.22,
           9, MUTED)
    s.text("DEMO SCRIPT", 8.35, 1.92, 3.8, 0.28, 10, MINT, True)
    demo_steps = [
        ("1", "Ask for order details without an ID."),
        ("2", "Provide ORD-1101 and review the lookup."),
        ("3", "Ask return policy for “this order”."),
        ("4", "Show policy source and eligibility reasoning."),
    ]
    y = 2.42
    for num, text in demo_steps:
        s.oval(8.4, y + 0.03, 0.34, 0.34, MINT_DARK)
        s.text(num, 8.48, y + 0.12, 0.18, 0.15, 9, MINT, True, "ctr")
        s.text(text, 8.92, y - 0.01, 3.35, 0.58, 12, CREAM)
        y += 0.94
    s.text("Illustrative chat storyboard based on implemented demo flow.", 8.4, 6.38,
           4.05, 0.2, 8, MUTED, False, italic=True)
    slides.append(s)

    s = Slide(8, "Conversation state and memory", "07  /  Context without accidental carryover")
    s.card(0.82, 1.95, 3.45, 1.7, "Thread checkpoint",
           "LangGraph state persists within a conversation for follow-up questions.",
           accent=BLUE, heading_size=15, body_size=12)
    s.card(4.93, 1.95, 3.45, 1.7, "Tagged turn record",
           "Each stored turn includes user_id, thread_id, session_id, turn_id, message, answer.",
           accent=ORANGE, heading_size=15, body_size=12)
    s.card(9.02, 1.95, 3.45, 1.7, "Long-term summary",
           "LLM consolidates useful history asynchronously into a per-user JSON file.",
           accent=MINT, heading_size=15, body_size=12)
    s.rect(1.0, 4.35, 11.25, 0.92, PANEL_LIGHT, rounded=True)
    s.text("Conversation A", 1.35, 4.65, 1.7, 0.24, 12, CREAM, True)
    s.rect(3.2, 4.74, 1.3, 0.07, MINT)
    s.text("ORD-1101 context retained in thread", 4.72, 4.62, 3.7, 0.27, 12, CREAM)
    s.rect(8.7, 4.74, 1.0, 0.07, LINE)
    s.text("New conversation", 10.0, 4.62, 2.0, 0.27, 11, ORANGE, True)
    s.text("Order context is not auto-selected from shared user memory in a new conversation; "
           "the assistant asks for an order ID when needed.", 1.05, 5.75, 11.0, 0.72,
           14, CREAM, True, "ctr")
    s.text("Memory file: memories/{user_id}/history_summarization.json", 1.05, 6.55,
           11.0, 0.24, 10, MUTED, False, "ctr")
    slides.append(s)

    s = Slide(9, "Prototype validation", "08  /  Experimental results")
    s.rect(0.82, 1.9, 3.0, 3.5, MINT_DARK, rounded=True)
    s.text("83", 1.15, 2.36, 2.2, 1.05, 52, CREAM, True, "ctr")
    s.text("automated tests\npassed", 1.1, 3.5, 2.25, 0.8, 20, MINT, True, "ctr")
    s.text("Full suite at project snapshot", 1.1, 4.75, 2.25, 0.3, 10, MUTED, False,
           "ctr")
    s.text("Coverage themes", 4.38, 1.96, 3.6, 0.34, 15, MINT, True)
    coverage = [
        ("Conversation workflow", "Greeting, order, refund, policy routing"),
        ("Grounding", "Policy ingestion, sources, order ownership"),
        ("Safety", "Input checks, guardrail decisions, response validation"),
        ("State & UI", "Memory, checkpoints, streaming, thread metadata"),
    ]
    y = 2.52
    for heading, body in coverage:
        s.text(heading, 4.4, y, 2.6, 0.25, 12, CREAM, True)
        s.text(body, 7.15, y, 5.05, 0.28, 11, MUTED)
        s.rect(4.4, y + 0.42, 7.85, 0.012, LINE)
        y += 0.73
    s.rect(4.4, 5.75, 7.9, 0.62, PANEL, rounded=True)
    s.text("Evidence is functional regression testing—not a user study or model-quality benchmark.",
           4.62, 5.92, 7.45, 0.26, 11, ORANGE, True, "ctr")
    s.text("Test result is reproducible with: python -m pytest -q", 4.45, 6.58,
           7.8, 0.24, 10, MUTED)
    slides.append(s)

    s = Slide(10, "Trust, safety, and control", "09  /  Guardrails")
    items = [
        ("Input safety", "Manual control-character / injection checks plus an LLM safety decision."),
        ("Verified identity scope", "Order lookup is scoped to the authenticated demo customer."),
        ("LLM role", "LLM classifies and communicates; repositories and policy sources provide facts."),
        ("Refund control", "Eligibility is checked; a refund requires an explicit approval step."),
    ]
    y = 1.9
    for i, (heading, body) in enumerate(items):
        color = (MINT, BLUE, ORANGE, MINT)[i]
        s.oval(0.95, y + 0.03, 0.5, 0.5, MINT_DARK if color == MINT else PANEL_LIGHT)
        s.text(f"0{i + 1}", 1.04, y + 0.19, 0.31, 0.16, 9, color, True, "ctr")
        s.text(heading, 1.72, y, 3.05, 0.33, 15, CREAM, True)
        s.text(body, 4.65, y, 7.5, 0.55, 13, MUTED)
        y += 1.05
    s.rect(0.9, 6.18, 11.6, 0.5, PANEL, rounded=True)
    s.text("Return-policy answer ≠ return submission  ·  Refund offer ≠ refund execution",
           1.15, 6.31, 11.1, 0.22, 12, MINT, True, "ctr")
    slides.append(s)

    s = Slide(11, "Limitations of the prototype", "10  /  What is not solved yet")
    limitations = [
        ("Demo commerce data", "Seeded SQLite orders and fixed demo identities; no live retailer integration."),
        ("Local model dependency", "Ollama availability and model behavior determine response quality and latency."),
        ("Evaluation gap", "Automated tests cover behavior, but no customer study or quantitative quality benchmark."),
        ("Workflow boundary", "Return eligibility can be explained; a full return-submission process is not implemented."),
        ("Production readiness", "Needs operational monitoring, privacy/retention policy, stronger auth, and scale testing."),
    ]
    y = 1.86
    for i, (head, body) in enumerate(limitations):
        s.rect(0.86, y, 11.6, 0.79, PANEL, rounded=True, line=LINE)
        s.text(f"0{i + 1}", 1.08, y + 0.24, 0.5, 0.2, 10,
               (RED, ORANGE, BLUE, ORANGE, RED)[i], True)
        s.text(head, 1.78, y + 0.13, 2.65, 0.28, 13, CREAM, True)
        s.text(body, 4.4, y + 0.13, 7.75, 0.46, 11, MUTED)
        y += 0.93
    slides.append(s)

    s = Slide(12, "Next steps", "11  /  Future work")
    roadmap = [
        ("01", "Measure", "Build a gold set of realistic support conversations; track routing accuracy, groundedness, refusal quality, and latency."),
        ("02", "Integrate", "Connect authenticated commerce APIs and a policy publishing / versioning workflow."),
        ("03", "Operationalize", "Add privacy controls, memory retention/deletion, observability, and load testing."),
        ("04", "Complete returns", "Implement a secure return-request workflow with explicit confirmation and idempotent state changes."),
    ]
    y = 1.86
    for number, heading, body in roadmap:
        s.oval(0.96, y + 0.08, 0.56, 0.56, MINT_DARK)
        s.text(number, 1.05, y + 0.27, 0.38, 0.17, 9, MINT, True, "ctr")
        s.text(heading, 1.82, y + 0.06, 2.4, 0.31, 15, CREAM, True)
        s.text(body, 4.15, y + 0.03, 7.75, 0.63, 12, MUTED)
        if number != "04":
            s.rect(1.23, y + 0.69, 0.04, 0.33, LINE)
        y += 1.12
    s.text("Target: move from a validated local prototype to a measured, secure service.",
           1.0, 6.47, 11.3, 0.34, 15, MINT, True, "ctr")
    slides.append(s)

    s = Slide(13, "ShopSense", "12  /  Takeaway")
    s.text("A customer-support prototype that joins", 0.9, 2.0, 11.5, 0.5, 23, CREAM,
           True, "ctr")
    takeaways = [
        ("Conversation", "Intent + thread context"),
        ("Evidence", "Policies + verified orders"),
        ("Control", "Guardrails + explicit approval"),
        ("Experience", "Natural answers + streaming"),
    ]
    for i, (head, body) in enumerate(takeaways):
        x = 0.78 + i * 3.15
        s.rect(x, 3.05, 2.78, 1.5, PANEL, rounded=True, line=LINE)
        s.text(head, x + 0.18, 3.34, 2.42, 0.33, 15, MINT, True, "ctr")
        s.text(body, x + 0.18, 3.9, 2.42, 0.3, 11, MUTED, False, "ctr")
    s.text("Thank you", 0.9, 5.2, 11.5, 0.55, 27, MINT, True, "ctr")
    s.text("Demo: ask about an unspecified order → provide ORD-1101 → ask about return "
           "eligibility.", 1.0, 6.12, 11.3, 0.3, 12, CREAM, False, "ctr")
    slides.append(s)
    return slides


def xml_package(slides: list[Slide]) -> dict[str, str]:
    slide_overrides = "".join(
        f'<Override PartName="/ppt/slides/slide{i}.xml" '
        'ContentType="application/vnd.openxmlformats-officedocument.presentationml.slide+xml"/>'
        for i in range(1, len(slides) + 1)
    )
    content_types = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
        '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
        '<Default Extension="xml" ContentType="application/xml"/>'
        '<Override PartName="/ppt/presentation.xml" '
        'ContentType="application/vnd.openxmlformats-officedocument.presentationml.presentation.main+xml"/>'
        '<Override PartName="/ppt/slideMasters/slideMaster1.xml" '
        'ContentType="application/vnd.openxmlformats-officedocument.presentationml.slideMaster+xml"/>'
        '<Override PartName="/ppt/slideLayouts/slideLayout1.xml" '
        'ContentType="application/vnd.openxmlformats-officedocument.presentationml.slideLayout+xml"/>'
        '<Override PartName="/ppt/theme/theme1.xml" '
        'ContentType="application/vnd.openxmlformats-officedocument.theme+xml"/>'
        + slide_overrides
        + "</Types>"
    )
    slide_ids = "".join(
        f'<p:sldId id="{255 + i}" r:id="rId{i + 1}"/>'
        for i in range(1, len(slides) + 1)
    )
    slide_rels = "".join(
        f'<Relationship Id="rId{i + 1}" '
        'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slide" '
        f'Target="slides/slide{i}.xml"/>'
        for i in range(1, len(slides) + 1)
    )
    presentation = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<p:presentation xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" '
        'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships" '
        'xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main">'
        '<p:sldMasterIdLst><p:sldMasterId id="2147483648" r:id="rId1"/></p:sldMasterIdLst>'
        f"<p:sldIdLst>{slide_ids}</p:sldIdLst>"
        '<p:sldSz cx="12192000" cy="6858000" type="screen16x9"/>'
        '<p:notesSz cx="6858000" cy="9144000"/>'
        '<p:defaultTextStyle><a:defPPr><a:defRPr lang="en-US"/></a:defPPr></p:defaultTextStyle>'
        "</p:presentation>"
    )
    pres_rels = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        '<Relationship Id="rId1" '
        'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slideMaster" '
        'Target="slideMasters/slideMaster1.xml"/>'
        + slide_rels
        + "</Relationships>"
    )
    master_tree = (
        '<p:spTree><p:nvGrpSpPr><p:cNvPr id="1" name=""/>'
        "<p:cNvGrpSpPr/><p:nvPr/></p:nvGrpSpPr>"
        "<p:grpSpPr><a:xfrm><a:off x=\"0\" y=\"0\"/><a:ext cx=\"0\" cy=\"0\"/>"
        "<a:chOff x=\"0\" y=\"0\"/><a:chExt cx=\"0\" cy=\"0\"/></a:xfrm></p:grpSpPr>"
        "</p:spTree>"
    )
    master = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<p:sldMaster xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" '
        'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships" '
        'xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main">'
        f'<p:cSld>{master_tree}</p:cSld>'
        '<p:clrMap bg1="lt1" tx1="dk1" bg2="lt2" tx2="dk2" '
        'accent1="accent1" accent2="accent2" accent3="accent3" accent4="accent4" '
        'accent5="accent5" accent6="accent6" hlink="hlink" folHlink="folHlink"/>'
        '<p:sldLayoutIdLst><p:sldLayoutId id="1" r:id="rId1"/></p:sldLayoutIdLst>'
        "<p:txStyles><p:titleStyle/><p:bodyStyle/><p:otherStyle/></p:txStyles></p:sldMaster>"
    )
    master_rels = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        '<Relationship Id="rId1" '
        'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slideLayout" '
        'Target="../slideLayouts/slideLayout1.xml"/>'
        '<Relationship Id="rId2" '
        'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/theme" '
        'Target="../theme/theme1.xml"/>'
        "</Relationships>"
    )
    layout = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<p:sldLayout xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" '
        'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships" '
        'xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main" '
        'type="blank" preserve="1"><p:cSld name="Blank">'
        '<p:spTree><p:nvGrpSpPr><p:cNvPr id="1" name=""/>'
        "<p:cNvGrpSpPr/><p:nvPr/></p:nvGrpSpPr>"
        "<p:grpSpPr><a:xfrm><a:off x=\"0\" y=\"0\"/><a:ext cx=\"0\" cy=\"0\"/>"
        "<a:chOff x=\"0\" y=\"0\"/><a:chExt cx=\"0\" cy=\"0\"/></a:xfrm></p:grpSpPr>"
        "</p:spTree></p:cSld><p:clrMapOvr><a:masterClrMapping/></p:clrMapOvr></p:sldLayout>"
    )
    layout_rels = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        '<Relationship Id="rId1" '
        'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slideMaster" '
        'Target="../slideMasters/slideMaster1.xml"/>'
        "</Relationships>"
    )
    theme = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<a:theme xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" name="ShopSense">'
        '<a:themeElements><a:clrScheme name="ShopSense">'
        '<a:dk1><a:srgbClr val="0A1A22"/></a:dk1><a:lt1><a:srgbClr val="F5F3EA"/></a:lt1>'
        '<a:dk2><a:srgbClr val="122A34"/></a:dk2><a:lt2><a:srgbClr val="A7BBC0"/></a:lt2>'
        '<a:accent1><a:srgbClr val="70E1BD"/></a:accent1><a:accent2><a:srgbClr val="FFB86B"/></a:accent2>'
        '<a:accent3><a:srgbClr val="86B9FF"/></a:accent3><a:accent4><a:srgbClr val="F27D75"/></a:accent4>'
        '<a:accent5><a:srgbClr val="1A705C"/></a:accent5><a:accent6><a:srgbClr val="29434C"/></a:accent6>'
        '<a:hlink><a:srgbClr val="70E1BD"/></a:hlink><a:folHlink><a:srgbClr val="86B9FF"/></a:folHlink>'
        '</a:clrScheme><a:fontScheme name="ShopSense">'
        f'<a:majorFont><a:latin typeface="{FONT}"/></a:majorFont>'
        f'<a:minorFont><a:latin typeface="{FONT}"/></a:minorFont></a:fontScheme>'
        '<a:fmtScheme name="ShopSense"><a:fillStyleLst>'
        '<a:solidFill><a:schemeClr val="phClr"/></a:solidFill>'
        '<a:solidFill><a:schemeClr val="phClr"/></a:solidFill>'
        '<a:solidFill><a:schemeClr val="phClr"/></a:solidFill></a:fillStyleLst>'
        '<a:lnStyleLst>'
        '<a:ln w="6350"><a:solidFill><a:schemeClr val="phClr"/></a:solidFill></a:ln>'
        '<a:ln w="12700"><a:solidFill><a:schemeClr val="phClr"/></a:solidFill></a:ln>'
        '<a:ln w="19050"><a:solidFill><a:schemeClr val="phClr"/></a:solidFill></a:ln>'
        '</a:lnStyleLst><a:effectStyleLst>'
        '<a:effectStyle><a:effectLst/></a:effectStyle>'
        '<a:effectStyle><a:effectLst/></a:effectStyle>'
        '<a:effectStyle><a:effectLst/></a:effectStyle></a:effectStyleLst>'
        '<a:bgFillStyleLst>'
        '<a:solidFill><a:schemeClr val="phClr"/></a:solidFill>'
        '<a:solidFill><a:schemeClr val="phClr"/></a:solidFill>'
        '<a:solidFill><a:schemeClr val="phClr"/></a:solidFill></a:bgFillStyleLst>'
        '</a:fmtScheme></a:themeElements></a:theme>'
    )
    root_rels = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        '<Relationship Id="rId1" '
        'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" '
        'Target="ppt/presentation.xml"/>'
        "</Relationships>"
    )
    files: dict[str, str] = {
        "[Content_Types].xml": content_types,
        "_rels/.rels": root_rels,
        "ppt/presentation.xml": presentation,
        "ppt/_rels/presentation.xml.rels": pres_rels,
        "ppt/slideMasters/slideMaster1.xml": master,
        "ppt/slideMasters/_rels/slideMaster1.xml.rels": master_rels,
        "ppt/slideLayouts/slideLayout1.xml": layout,
        "ppt/slideLayouts/_rels/slideLayout1.xml.rels": layout_rels,
        "ppt/theme/theme1.xml": theme,
    }
    for i, slide in enumerate(slides, 1):
        files[f"ppt/slides/slide{i}.xml"] = slide.xml()
        files[f"ppt/slides/_rels/slide{i}.xml.rels"] = (
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
            '<Relationship Id="rId1" '
            'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slideLayout" '
            'Target="../slideLayouts/slideLayout1.xml"/>'
            "</Relationships>"
        )
    return files


def main() -> None:
    files = xml_package(build_slides())
    with ZipFile(OUTPUT, "w", ZIP_DEFLATED) as package:
        for name, contents in files.items():
            package.writestr(name, contents.encode("utf-8"))
    print(f"Created {OUTPUT} with 13 editable slides.")


if __name__ == "__main__":
    main()
