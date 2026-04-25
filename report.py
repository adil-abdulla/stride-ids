from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch, cm
from reportlab.lib.colors import HexColor, white, black, Color
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer,
    Table, TableStyle, HRFlowable, PageBreak)
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_RIGHT
from reportlab.pdfgen import canvas
from reportlab.platypus import BaseDocTemplate, Frame, PageTemplate
from datetime import datetime
import io

# ========================
# COLOR PALETTE
# ========================
C_BG = HexColor('#ffffff')
C_CARD = HexColor('#f1f5f9')
C_CARD2 = HexColor('#e2e8f0')
C_ACCENT = HexColor('#4f6ef7')
C_ACCENT2 = HexColor('#7c3aed')
C_CYAN = HexColor('#06b6d4')
C_GREEN = HexColor('#10b981')
C_RED = HexColor('#ef4444')
C_ORANGE = HexColor('#f97316')
C_YELLOW = HexColor('#eab308')
C_TEXT = HexColor('#0f172a')
C_TEXT2 = HexColor('#334155')
C_TEXT3 = HexColor('#64748b')
C_BORDER = HexColor('#cbd5e1')
C_WHITE = HexColor('#ffffff')

def risk_color(risk):
    return {
        'CRITICAL': C_RED,
        'HIGH': C_ORANGE,
        'MEDIUM': C_YELLOW,
        'LOW': C_GREEN
    }.get(risk, C_TEXT2)

def pred_color(pred):
    return C_RED if pred == 'ATTACK' else C_GREEN

def score_color(score):
    try:
        s = float(score)
        if s > 0.7: return C_RED
        if s > 0.4: return C_ORANGE
        return C_GREEN
    except:
        return C_TEXT2

class NumberedCanvas(canvas.Canvas):
    """Canvas with page numbers"""
    def __init__(self, *args, **kwargs):
        canvas.Canvas.__init__(self, *args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(
            dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_page_elements(num_pages)
            canvas.Canvas.showPage(self)
        canvas.Canvas.save(self)

    def draw_page_elements(self, page_count):
        self.saveState()
        W, H = A4

        # Top accent bar
        self.setFillColor(C_ACCENT)
        self.rect(0, H-4, W, 4, fill=1, stroke=0)

        # Side accent line
        self.setFillColor(C_ACCENT)
        self.setFillAlpha(0.3)
        self.rect(0, 0, 3, H, fill=1, stroke=0)

        # Footer
        self.setFillAlpha(1)
        self.setFillColor(C_CARD)
        self.rect(0, 0, W, 28, fill=1, stroke=0)

        self.setFillColor(C_BORDER)
        self.rect(0, 28, W, 1, fill=1, stroke=0)

        self.setFillColor(C_TEXT3)
        self.setFont('Helvetica', 7)
        self.drawString(
            1.5*cm, 10,
            'STRIDE v2.0 — Confidential Threat Intelligence Report · MBZUAI Research')
        self.drawRightString(
            W - 1.5*cm, 10,
            f'Page {self._pageNumber} of {page_count}')

        self.restoreState()

def make_section_header(text):
    """Create a styled section header"""
    data = [[text]]
    t = Table(data, colWidths=[18*cm])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), C_CARD),
        ('TEXTCOLOR', (0,0), (-1,-1), C_ACCENT),
        ('FONTNAME', (0,0), (-1,-1), 'Helvetica-Bold'),
        ('FONTSIZE', (0,0), (-1,-1), 7),
        ('LEFTPADDING', (0,0), (-1,-1), 12),
        ('RIGHTPADDING', (0,0), (-1,-1), 12),
        ('TOPPADDING', (0,0), (-1,-1), 7),
        ('BOTTOMPADDING', (0,0), (-1,-1), 7),
        ('LINEBELOW', (0,0), (-1,-1), 1, C_ACCENT),
        ('LINEBEFORE', (0,0), (0,-1), 3, C_ACCENT),
    ]))
    return t

def make_stat_card(label, value, color=None):
    """Create a single stat card"""
    vc = color or C_ACCENT
    data = [[value], [label]]
    t = Table(data, colWidths=[4.3*cm])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), C_CARD),
        ('TEXTCOLOR', (0,0), (-1,0), vc),
        ('TEXTCOLOR', (0,1), (-1,1), C_TEXT3),
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
        ('FONTNAME', (0,1), (-1,1), 'Helvetica'),
        ('FONTSIZE', (0,0), (-1,0), 18),
        ('FONTSIZE', (0,1), (-1,1), 7),
        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('TOPPADDING', (0,0), (-1,0), 14),
        ('BOTTOMPADDING', (0,0), (-1,0), 4),
        ('TOPPADDING', (0,1), (-1,1), 4),
        ('BOTTOMPADDING', (0,1), (-1,1), 14),
        ('LINEBELOW', (0,0), (-1,0), 1,
         HexColor('#' + '{:02x}{:02x}{:02x}'.format(
             int(vc.red*255),
             int(vc.green*255),
             int(vc.blue*255)))),
        ('BOX', (0,0), (-1,-1), 0.5, C_BORDER),
    ]))
    return t

def generate_report(scan_results):
    buffer = io.BytesIO()

    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=1.5*cm,
        leftMargin=1.8*cm,
        topMargin=1.8*cm,
        bottomMargin=1.5*cm
    )

    W = A4[0] - 1.5*cm - 1.8*cm  # usable width

    styles = getSampleStyleSheet()

    # Custom styles
    S = {
        'title': ParagraphStyle('t',
            fontSize=32, textColor=C_WHITE,
            fontName='Helvetica-Bold',
            spaceAfter=2, leading=36),
        'subtitle': ParagraphStyle('st',
            fontSize=10, textColor=C_ACCENT,
            fontName='Helvetica',
            spaceAfter=2, letterSpacing=1),
        'eyebrow': ParagraphStyle('ey',
            fontSize=8, textColor=C_TEXT3,
            fontName='Helvetica',
            spaceAfter=16, letterSpacing=2),
        'body': ParagraphStyle('b',
            fontSize=8.5, textColor=C_TEXT,
            fontName='Helvetica',
            spaceAfter=4, leading=13),
        'small': ParagraphStyle('sm',
            fontSize=7.5, textColor=C_TEXT2,
            fontName='Helvetica',
            spaceAfter=3, leading=11),
        'mono': ParagraphStyle('mo',
            fontSize=7, textColor=C_CYAN,
            fontName='Courier', spaceAfter=2),
        'label': ParagraphStyle('lb',
            fontSize=7, textColor=C_TEXT3,
            fontName='Helvetica-Bold',
            letterSpacing=1.5),
    }

    content = []

    # ========================
    # COVER SECTION
    # ========================
    ts = datetime.now().strftime("%B %d, %Y — %H:%M UTC")
    filename = scan_results.get('filename', 'Unknown')

    content.append(Spacer(1, 0.5*cm))
    content.append(Paragraph(
        "ADVANCED THREAT INTELLIGENCE",
        S['eyebrow']))
    content.append(Paragraph("STRIDE", S['title']))
    content.append(Paragraph(
        "Network Intrusion Detection Report",
        S['subtitle']))
    content.append(Spacer(1, 0.3*cm))

    # Divider
    content.append(HRFlowable(
        width="100%", thickness=1,
        color=C_ACCENT, spaceAfter=16,
        spaceBefore=4))

    # Meta info row
    meta = [[
        'FILE ANALYZED', 'DATE GENERATED',
        'FRAMEWORK', 'VERSION'
    ], [
        filename[:35] + ('...' if len(filename) > 35 else ''),
        ts, 'STRIDE / CICIDS-2017', 'v2.0'
    ]]
    meta_t = Table(meta, colWidths=[
        5.5*cm, 5.5*cm, 4.5*cm, 2.5*cm])
    meta_t.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), C_CARD),
        ('BACKGROUND', (0,1), (-1,1), C_CARD2),
        ('TEXTCOLOR', (0,0), (-1,0), C_TEXT3),
        ('TEXTCOLOR', (0,1), (-1,1), C_TEXT),
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
        ('FONTNAME', (0,1), (-1,1), 'Helvetica'),
        ('FONTSIZE', (0,0), (-1,0), 6.5),
        ('FONTSIZE', (0,1), (-1,1), 8),
        ('LEFTPADDING', (0,0), (-1,-1), 10),
        ('RIGHTPADDING', (0,0), (-1,-1), 10),
        ('TOPPADDING', (0,0), (-1,-1), 7),
        ('BOTTOMPADDING', (0,0), (-1,-1), 7),
        ('GRID', (0,0), (-1,-1), 0.5, C_BORDER),
        ('LINEABOVE', (0,0), (-1,0), 2, C_ACCENT),
    ]))
    content.append(meta_t)
    content.append(Spacer(1, 0.6*cm))

    # ========================
    # STAT CARDS ROW
    # ========================
    n_attacks = scan_results.get('attack_sequences', 0)
    n_benign = scan_results.get('benign_sequences', 0)
    n_total = scan_results.get('total_sequences', 0)
    attack_rate = scan_results.get('attack_rate', 0)
    max_score = scan_results.get('max_ensemble_score', 0)
    avg_score = scan_results.get('avg_ensemble_score', 0)

    if attack_rate > 10:
        threat_level = "CRITICAL"
        tcolor = C_RED
    elif attack_rate > 5:
        threat_level = "HIGH"
        tcolor = C_ORANGE
    elif attack_rate > 1:
        threat_level = "MEDIUM"
        tcolor = C_YELLOW
    else:
        threat_level = "LOW"
        tcolor = C_GREEN

    cards_row = [[
        make_stat_card("THREAT LEVEL",
                       threat_level, tcolor),
        make_stat_card("ATTACKS DETECTED",
                       str(n_attacks), C_RED),
        make_stat_card("BENIGN SEQUENCES",
                       f"{n_benign:,}", C_GREEN),
        make_stat_card("MAX THREAT SCORE",
                       str(max_score), C_ORANGE),
    ]]
    cards_t = Table(cards_row,
                    colWidths=[4.3*cm]*4,
                    spaceBefore=0, spaceAfter=0)
    cards_t.setStyle(TableStyle([
        ('LEFTPADDING', (0,0), (-1,-1), 3),
        ('RIGHTPADDING', (0,0), (-1,-1), 3),
        ('TOPPADDING', (0,0), (-1,-1), 0),
        ('BOTTOMPADDING', (0,0), (-1,-1), 0),
    ]))
    content.append(cards_t)
    content.append(Spacer(1, 0.5*cm))

    # ========================
    # MODEL CONSENSUS
    # ========================
    content.append(make_section_header(
        "MODEL CONSENSUS"))
    content.append(Spacer(1, 0.2*cm))

    ms = scan_results.get('model_scores', {})
    model_data = [
        ['MODEL', 'AVG SCORE', 'ROLE', 'WEIGHT'],
        ['Transformer V3',
         str(ms.get('transformer', '—')),
         'Primary anomaly detector — '
         'flow sequence self-supervision',
         '50%'],
        ['MLP Autoencoder',
         str(ms.get('mlp', '—')),
         'Statistical baseline — '
         'reconstruction error scoring',
         '30%'],
        ['Robust Transformer',
         str(ms.get('robust', '—')),
         'Adversarial defense — '
         'PGD-trained for evasion resistance',
         '20%'],
        ['Ensemble', str(ms.get('ensemble', '—')),
         'Weighted combination of all models',
         '100%'],
    ]
    model_t = Table(model_data,
                    colWidths=[4*cm, 2.5*cm,
                               9*cm, 2.5*cm])
    model_style = [
        ('BACKGROUND', (0,0), (-1,0), C_CARD),
        ('BACKGROUND', (0,1), (-1,-1), C_CARD2),
        ('TEXTCOLOR', (0,0), (-1,0), C_TEXT3),
        ('TEXTCOLOR', (0,1), (0,-1), C_TEXT),
        ('TEXTCOLOR', (2,1), (2,-1), C_TEXT2),
        ('TEXTCOLOR', (3,1), (3,-1), C_TEXT3),
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
        ('FONTNAME', (0,1), (0,-1), 'Helvetica-Bold'),
        ('FONTNAME', (1,1), (1,-1), 'Helvetica-Bold'),
        ('FONTNAME', (2,1), (2,-1), 'Helvetica'),
        ('FONTNAME', (3,1), (3,-1), 'Helvetica'),
        ('FONTSIZE', (0,0), (-1,0), 6.5),
        ('FONTSIZE', (0,1), (-1,-1), 8),
        ('LEFTPADDING', (0,0), (-1,-1), 10),
        ('RIGHTPADDING', (0,0), (-1,-1), 10),
        ('TOPPADDING', (0,0), (-1,-1), 8),
        ('BOTTOMPADDING', (0,0), (-1,-1), 8),
        ('GRID', (0,0), (-1,-1), 0.5, C_BORDER),
        ('ROWBACKGROUNDS', (0,1), (-1,-1),
         [C_CARD2, C_CARD]),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('LINEABOVE', (0,0), (-1,0), 2, C_ACCENT2),
    ]
    # Color model scores
    score_colors = [C_ACCENT, C_ACCENT2, C_CYAN, C_GREEN]
    for i, sc in enumerate(score_colors, 1):
        model_style.append(
            ('TEXTCOLOR', (1,i), (1,i), sc))
    model_t.setStyle(TableStyle(model_style))
    content.append(model_t)
    content.append(Spacer(1, 0.5*cm))

    # ========================
    # TOP SUSPICIOUS SEQUENCES
    # ========================
    content.append(make_section_header(
        "TOP SUSPICIOUS SEQUENCES"))
    content.append(Spacer(1, 0.2*cm))

    top = scan_results.get('top_suspicious', [])
    if top:
        headers = ['#', 'TRANSFORMER', 'MLP',
                   'ROBUST', 'ENSEMBLE',
                   'RISK', 'VERDICT', 'TYPE']
        seq_data = [headers]
        for r in top[:10]:
            seq_data.append([
                f"#{r.get('sequence','?')}",
                str(r.get('transformer_score','—')),
                str(r.get('mlp_score','—')),
                str(r.get('robust_score','—')),
                str(r.get('ensemble_score','—')),
                r.get('risk_level','—'),
                r.get('prediction','—'),
                (r.get('attack_type','—') or '—')[:18]
            ])

        seq_t = Table(seq_data,
                      colWidths=[1.4*cm, 2.4*cm,
                                 2.4*cm, 2.4*cm,
                                 2.4*cm, 2*cm,
                                 2.2*cm, 2.8*cm])
        seq_style = [
            ('BACKGROUND', (0,0), (-1,0), C_CARD),
            ('BACKGROUND', (0,1), (-1,-1), C_CARD2),
            ('TEXTCOLOR', (0,0), (-1,0), C_TEXT3),
            ('TEXTCOLOR', (0,1), (-1,-1), C_TEXT),
            ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
            ('FONTNAME', (0,1), (-1,-1), 'Helvetica'),
            ('FONTSIZE', (0,0), (-1,0), 6.5),
            ('FONTSIZE', (0,1), (-1,-1), 7.5),
            ('ALIGN', (1,0), (-1,-1), 'CENTER'),
            ('ALIGN', (0,0), (0,-1), 'CENTER'),
            ('LEFTPADDING', (0,0), (-1,-1), 6),
            ('RIGHTPADDING', (0,0), (-1,-1), 6),
            ('TOPPADDING', (0,0), (-1,-1), 7),
            ('BOTTOMPADDING', (0,0), (-1,-1), 7),
            ('GRID', (0,0), (-1,-1), 0.5, C_BORDER),
            ('ROWBACKGROUNDS', (0,1), (-1,-1),
             [C_CARD2, C_CARD]),
            ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
            ('LINEABOVE', (0,0), (-1,0), 2, C_CYAN),
        ]
        # Color risk and prediction per row
        for i, r in enumerate(top[:10], 1):
            rc = risk_color(r.get('risk_level',''))
            pc = pred_color(r.get('prediction',''))
            sc = score_color(r.get('ensemble_score', 0))
            seq_style += [
                ('TEXTCOLOR', (5,i), (5,i), rc),
                ('FONTNAME', (5,i), (5,i),
                 'Helvetica-Bold'),
                ('TEXTCOLOR', (6,i), (6,i), pc),
                ('FONTNAME', (6,i), (6,i),
                 'Helvetica-Bold'),
                ('TEXTCOLOR', (4,i), (4,i), sc),
                ('FONTNAME', (4,i), (4,i),
                 'Helvetica-Bold'),
            ]
        seq_t.setStyle(TableStyle(seq_style))
        content.append(seq_t)
    content.append(Spacer(1, 0.5*cm))

    # ========================
    # ATTACK BREAKDOWN
    # ========================
    breakdown = scan_results.get(
        'attack_breakdown', {})
    filtered_bd = {k: v for k, v in breakdown.items()
                   if k != 'BENIGN'}
    if filtered_bd:
        content.append(make_section_header(
            "ATTACK TYPE BREAKDOWN"))
        content.append(Spacer(1, 0.2*cm))

        bd_headers = ['ATTACK TYPE', 'SEQUENCES',
                      'DETECTED', 'DETECTION RATE',
                      'AVG SCORE']
        bd_data = [bd_headers]
        for atype, stats in sorted(
                filtered_bd.items(),
                key=lambda x: x[1].get(
                    'detection_rate', 0),
                reverse=True):
            dr = stats.get('detection_rate', 0)
            bd_data.append([
                atype[:28],
                str(stats.get('count', 0)),
                str(stats.get('detected', 0)),
                f"{dr}%",
                str(stats.get('avg_score', 0))
            ])

        bd_t = Table(bd_data,
                     colWidths=[6*cm, 2.5*cm,
                                2.5*cm, 3.5*cm,
                                3.5*cm])
        bd_style = [
            ('BACKGROUND', (0,0), (-1,0), C_CARD),
            ('BACKGROUND', (0,1), (-1,-1), C_CARD2),
            ('TEXTCOLOR', (0,0), (-1,0), C_TEXT3),
            ('TEXTCOLOR', (0,1), (-1,-1), C_TEXT),
            ('TEXTCOLOR', (3,1), (3,-1), C_CYAN),
            ('TEXTCOLOR', (4,1), (4,-1), C_ORANGE),
            ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
            ('FONTNAME', (3,1), (3,-1), 'Helvetica-Bold'),
            ('FONTSIZE', (0,0), (-1,0), 6.5),
            ('FONTSIZE', (0,1), (-1,-1), 8),
            ('LEFTPADDING', (0,0), (-1,-1), 10),
            ('RIGHTPADDING', (0,0), (-1,-1), 10),
            ('TOPPADDING', (0,0), (-1,-1), 8),
            ('BOTTOMPADDING', (0,0), (-1,-1), 8),
            ('GRID', (0,0), (-1,-1), 0.5, C_BORDER),
            ('ROWBACKGROUNDS', (0,1), (-1,-1),
             [C_CARD2, C_CARD]),
            ('LINEABOVE', (0,0), (-1,0), 2,
             C_ORANGE),
        ]
        bd_t.setStyle(TableStyle(bd_style))
        content.append(bd_t)
        content.append(Spacer(1, 0.5*cm))

    # ========================
    # DNA ALERTS
    # ========================
    dna_alerts = scan_results.get('dna_alerts', [])
    if dna_alerts:
        content.append(make_section_header(
            "THREAT DNA MATCH ALERTS"))
        content.append(Spacer(1, 0.2*cm))

        for alert in dna_alerts:
            alert_data = [
                [f"⚠  Sequence #{alert.get('sequence')} — Known Threat Signature Detected",
                 ''],
                ['DNA Hash:', alert.get('dna_hash','—')],
                ['Match:', alert.get('alert','—')],
            ]
            alert_t = Table(
                alert_data,
                colWidths=[3*cm, 15*cm])
            alert_t.setStyle(TableStyle([
                ('SPAN', (0,0), (1,0)),
                ('BACKGROUND', (0,0), (-1,0),
                 HexColor('#1a0a0a')),
                ('BACKGROUND', (0,1), (-1,-1),
                 HexColor('#120808')),
                ('TEXTCOLOR', (0,0), (-1,0), C_RED),
                ('TEXTCOLOR', (0,1), (0,-1), C_TEXT3),
                ('TEXTCOLOR', (1,1), (1,-1), C_TEXT2),
                ('TEXTCOLOR', (1,2), (1,2), C_CYAN),
                ('FONTNAME', (0,0), (-1,0),
                 'Helvetica-Bold'),
                ('FONTNAME', (0,1), (0,-1),
                 'Helvetica-Bold'),
                ('FONTNAME', (1,1), (1,1), 'Courier'),
                ('FONTSIZE', (0,0), (-1,0), 8.5),
                ('FONTSIZE', (0,1), (-1,-1), 7.5),
                ('LEFTPADDING', (0,0), (-1,-1), 10),
                ('RIGHTPADDING', (0,0), (-1,-1), 10),
                ('TOPPADDING', (0,0), (-1,-1), 7),
                ('BOTTOMPADDING', (0,0), (-1,-1), 7),
                ('BOX', (0,0), (-1,-1), 1,
                 HexColor('#3a0000')),
                ('LINEBELOW', (0,0), (-1,0), 0.5,
                 HexColor('#3a0000')),
            ]))
            content.append(alert_t)
            content.append(Spacer(1, 0.15*cm))

        content.append(Spacer(1, 0.3*cm))

    # ========================
    # RECOMMENDATIONS
    # ========================
    content.append(make_section_header(
        "RECOMMENDATIONS"))
    content.append(Spacer(1, 0.2*cm))

    recs = []
    if n_attacks > 0:
        recs.append((
            "Immediate Investigation Required",
            f"Review the {n_attacks} flagged sequence(s) "
            f"above. Prioritize CRITICAL and HIGH risk "
            f"detections — cross-reference with firewall "
            f"logs and access records."))
    if float(max_score) > 0.9:
        recs.append((
            "High-Confidence Attack Detected",
            f"Max ensemble score of {max_score} indicates "
            f"near-certain attack activity. Consider "
            f"isolating affected network segments and "
            f"initiating incident response protocol."))
    if dna_alerts:
        recs.append((
            "Known Threat Actor Identified",
            f"{len(dna_alerts)} DNA signature match(es) "
            f"found — these attack patterns were "
            f"previously observed. Correlate with "
            f"historical scans to identify campaign scope."))
    if float(attack_rate) > 1:
        recs.append((
            "Elevated Attack Rate",
            f"Attack rate of {attack_rate}% exceeds "
            f"normal baseline. Review firewall rules, "
            f"check for lateral movement, and monitor "
            f"high-risk hosts closely."))
    recs.append((
        "Model Improvement",
        "Continue providing feedback on detections "
        "via the STRIDE interface to improve model "
        "accuracy through the self-evolving system."))
    recs.append((
        "Federated Intelligence Sharing",
        "Run a federation round to share threat "
        "signatures with partner organizations — "
        "raw data never leaves your network."))

    for i, (title, desc) in enumerate(recs, 1):
        rec_data = [
            [f"{i}", f"{title}", ''],
            ['', desc, '']
        ]
        rec_t = Table(
            rec_data,
            colWidths=[0.8*cm, 16.2*cm, 1*cm])
        rec_t.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,-1), C_CARD),
            ('TEXTCOLOR', (0,0), (0,-1), C_ACCENT),
            ('TEXTCOLOR', (1,0), (1,0), C_TEXT),
            ('TEXTCOLOR', (1,1), (1,1), C_TEXT2),
            ('FONTNAME', (0,0), (0,-1),
             'Helvetica-Bold'),
            ('FONTNAME', (1,0), (1,0),
             'Helvetica-Bold'),
            ('FONTNAME', (1,1), (1,1), 'Helvetica'),
            ('FONTSIZE', (0,0), (-1,-1), 8.5),
            ('FONTSIZE', (1,1), (1,1), 8),
            ('LEFTPADDING', (0,0), (-1,-1), 8),
            ('RIGHTPADDING', (0,0), (-1,-1), 8),
            ('TOPPADDING', (0,0), (-1,0), 8),
            ('BOTTOMPADDING', (0,0), (-1,0), 2),
            ('TOPPADDING', (0,1), (-1,1), 2),
            ('BOTTOMPADDING', (0,1), (-1,1), 8),
            ('SPAN', (0,0), (0,1)),
            ('VALIGN', (0,0), (0,-1), 'MIDDLE'),
            ('ALIGN', (0,0), (0,-1), 'CENTER'),
            ('BOX', (0,0), (-1,-1), 0.5, C_BORDER),
            ('LINEBEFORE', (0,0), (0,-1), 3,
             C_ACCENT),
        ]))
        content.append(rec_t)
        content.append(Spacer(1, 0.15*cm))

    content.append(Spacer(1, 0.5*cm))

    # ========================
    # BUILD
    # ========================
    doc.build(content, canvasmaker=NumberedCanvas)
    buffer.seek(0)
    return buffer.getvalue()