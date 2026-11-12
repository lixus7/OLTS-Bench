#!/usr/bin/env python3
"""Draw the figures of docs/CONCEPTS.md and docs/CONCEPTS.zh-CN.md as SVG files.

    python docs/figures/make_figures.py          # writes docs/figures/<name>.en.svg and <name>.zh.svg

Figure 2 draws the real ETTh1 series, so data/ETTh1.csv must be present. Needs numpy and pandas.
Every figure is one function that returns a `Fig`; labels come from L(chinese, english).
"""
import argparse
import html
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LANG = 'en'


def L(zh, en):
    return zh if LANG == 'zh' else en


# Colours. Blue = online model and its updates, orange = frozen model, green = true values (targets),
# crimson = values used before they were observed. Checked for colour-vision deficiencies.
LIGHT = {
    'surface': '#ffffff', 'ink': '#14202b', 'ink-2': '#44515d', 'muted': '#66727d',
    'rule': '#dde2e7', 'rule-strong': '#c3cbd3', 'cell': '#eef1f4',
    'seg-train': '#c6ced6', 'seg-val': '#dbe1e7', 'seg-test': '#eef1f4', 'x-t': '#cfd6de',
    'ol': '#2a78d6', 'ol-t': '#dbe8f9', 'fz': '#eb6834', 'fz-t': '#fbe3d9',
    'lab': '#1a9c7a', 'lab-t': '#d8f0e8', 'bad': '#c2255c', 'bad-t': '#f9dde7',
}
DARK = {
    'surface': '#151a1f', 'ink': '#e6eaee', 'ink-2': '#b6bfc7', 'muted': '#8b959e',
    'rule': '#283037', 'rule-strong': '#3a444d', 'cell': '#20272d',
    'seg-train': '#46525c', 'seg-val': '#343e47', 'seg-test': '#232a30', 'x-t': '#3d4852',
    'ol': '#3987e5', 'ol-t': '#1b3350', 'fz': '#d95926', 'fz-t': '#4a2617',
    'lab': '#24a884', 'lab-t': '#143b31', 'bad': '#e0527e', 'bad-t': '#4a1c2b',
}
SANS = '"IBM Plex Sans", "Noto Sans SC", "PingFang SC", "Hiragino Sans GB", "Microsoft YaHei", "Noto Sans CJK SC", system-ui, sans-serif'
MONO = '"IBM Plex Mono", ui-monospace, "SFMono-Regular", Menlo, Consolas, "Noto Sans SC", "PingFang SC", "Microsoft YaHei", monospace'
CSS = '''
svg.g text { font-family: %(sans)s; font-size: 12px; fill: var(--ink-2); }
svg.g .g-b { font-size: 13px; font-weight: 600; fill: var(--ink); }
svg.g .g-k { fill: var(--ink); }
svg.g .g-s { font-size: 11px; fill: var(--muted); }
svg.g .g-m { font-family: %(mono)s; font-size: 11px; fill: var(--ink-2); }
svg.g .g-n { font-family: %(mono)s; font-size: 11px; font-weight: 500; fill: var(--surface); }
svg.g .g-cell { fill: var(--cell); }
svg.g .g-fut { fill: var(--surface); stroke: var(--rule-strong); stroke-width: 1; stroke-dasharray: 2 3; }
svg.g .g-train { fill: var(--seg-train); }
svg.g .g-val { fill: var(--seg-val); }
svg.g .g-test { fill: var(--seg-test); }
svg.g .g-xt { fill: var(--x-t); stroke: var(--ink-2); stroke-width: 1.25; }
svg.g .g-yt { fill: var(--lab-t); stroke: var(--lab); stroke-width: 1.5; }
svg.g .g-p { fill: var(--surface); stroke: var(--lab); stroke-width: 1.5; stroke-dasharray: 5 3; }
svg.g .g-olt { fill: var(--ol-t); stroke: var(--ol); stroke-width: 1.5; }
svg.g .g-fzt { fill: var(--fz-t); stroke: var(--fz); stroke-width: 1.5; }
svg.g .g-badt { fill: var(--bad-t); stroke: var(--bad); stroke-width: 1.5; stroke-dasharray: 5 3; }
svg.g .g-card { fill: var(--surface); stroke: var(--rule-strong); stroke-width: 1; }
svg.g .g-soft { fill: var(--cell); }
svg.g .g-dot { fill: var(--ink-2); }
svg.g .g-ax { stroke: var(--rule-strong); stroke-width: 1; fill: none; }
svg.g .g-grid { stroke: var(--rule); stroke-width: 1; fill: none; }
svg.g .g-guide { stroke: var(--muted); stroke-width: 1; stroke-dasharray: 2 4; fill: none; }
svg.g .g-now { stroke: var(--ink); stroke-width: 1.5; fill: none; }
svg.g .g-ln { stroke: var(--ink-2); stroke-width: 1.25; fill: none; }
svg.g .g-lnh { fill: var(--ink-2); }
svg.g .g-ln.ol, svg.g .g-tick.ol { stroke: var(--ol); } svg.g .g-lnh.ol { fill: var(--ol); }
svg.g .g-ln.fz { stroke: var(--fz); } svg.g .g-lnh.fz { fill: var(--fz); }
svg.g .g-ln.bad { stroke: var(--bad); } svg.g .g-lnh.bad { fill: var(--bad); }
svg.g .g-thick { stroke-width: 2.5; stroke-linecap: round; }
svg.g .g-tick { stroke-width: 1.5; fill: none; }
svg.g .g-spark { fill: none; stroke: var(--ink-2); stroke-width: 1.25; stroke-linejoin: round; }
''' % dict(sans=SANS, mono=MONO)


def css_literal(palette):
    """The stylesheet with colours written out, for a stand-alone SVG file."""
    return re.sub(r'var\(--([a-z0-9-]+)\)', lambda m: palette[m.group(1)], CSS)


def tw(s, size=12.0, mono=False):
    """Rough text width in px (CJK glyphs are one em wide)."""
    w = 0.0
    for ch in s:
        if ord(ch) > 0x2E80:
            w += size
        elif mono:
            w += size * 0.6
        elif ch in " il.,:;|!'·":
            w += size * 0.3
        elif ch in 'mwMW':
            w += size * 0.86
        elif ch.isupper() or ch.isdigit() or ch in '–−+=→':
            w += size * 0.62
        else:
            w += size * 0.53
    return w


class Fig:
    def __init__(self, name, w, h, label):
        self.name, self.w, self.h, self.label, self.p = name, w, h, label, []

    def rect(self, x, y, w, h, cls, rx=0):
        r = f' rx="{rx:g}"' if rx else ''
        self.p.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}"{r} class="{cls}"/>')

    def text(self, x, y, s, cls='', anchor='start'):
        c = f' class="{cls}"' if cls else ''
        a = '' if anchor == 'start' else f' text-anchor="{anchor}"'
        self.p.append(f'<text x="{x:.1f}" y="{y:.1f}"{c}{a}>{html.escape(s)}</text>')

    def line(self, x1, y1, x2, y2, cls='g-ax'):
        self.p.append(f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" class="{cls}"/>')

    def path(self, d, cls):
        self.p.append(f'<path d="{d}" class="{cls}"/>')

    def circle(self, x, y, r, cls):
        self.p.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{r:g}" class="{cls}"/>')

    def arrow(self, pts, tone='', head=7):
        """A polyline through pts with an arrowhead at the last point."""
        (x1, y1), (x2, y2) = pts[-2], pts[-1]
        d = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5 or 1
        ux, uy = (x2 - x1) / d, (y2 - y1) / d
        bx, by = x2 - ux * head, y2 - uy * head
        body = pts[:-1] + [(bx, by)]
        self.path('M' + ' L'.join(f'{x:.1f},{y:.1f}' for x, y in body), f'g-ln {tone}'.strip())
        w = head * 0.45
        tri = [(x2, y2), (bx - uy * w, by + ux * w), (bx + uy * w, by - ux * w)]
        self.p.append('<polygon points="' + ' '.join(f'{x:.1f},{y:.1f}' for x, y in tri) + f'" class="{("g-lnh " + tone).strip()}"/>')

    def brace(self, x1, x2, y, up=False, tone=''):
        t = -5 if up else 5
        self.path(f'M{x1:.1f},{y - t:.1f} L{x1:.1f},{y:.1f} L{x2:.1f},{y:.1f} L{x2:.1f},{y - t:.1f}', f'g-ln {tone}'.strip())

    def box(self, x, y, w, h, title, sub=None, cls='g-card'):
        self.rect(x, y, w, h, cls, 6)
        if sub is None:
            self.text(x + w / 2, y + h / 2 + 4.5, title, 'g-k', 'middle')
        else:
            self.text(x + w / 2, y + h / 2 - 3, title, 'g-k', 'middle')
            self.text(x + w / 2, y + h / 2 + 13, sub, 'g-s', 'middle')

    def svg(self, standalone=False, palette=LIGHT):
        label = html.escape(self.label)
        if not standalone:
            return (f'<svg class="g" viewBox="0 0 {self.w} {self.h}" role="img" aria-label="{label}">'
                    + ''.join(self.p) + '</svg>')
        return (f'<svg xmlns="http://www.w3.org/2000/svg" class="g" viewBox="0 0 {self.w} {self.h}" width="{self.w}" '
                f'height="{self.h}" role="img" aria-label="{label}">\n<title>{label}</title>\n<style>{css_literal(palette)}</style>\n'
                f'<rect width="{self.w}" height="{self.h}" fill="{palette["surface"]}"/>\n' + '\n'.join(self.p) + '\n</svg>\n')


def num(n):
    return f'{n:,}'


# ------------------------------------------------------------------ figure 1: one sample

def fig_sample():
    f = Fig('sample', 760, 232, L('一个样本由预测起点 t 定义：输入是最近 W 行，目标是之后 H 行。',
                                  'A sample is defined by its forecast origin t: the input is the last W rows, the targets are the next H rows.'))
    x0, cw, n, h = 150, 19, 30, 26

    def strip(y, a, b, c):
        for i in range(n):
            cls = 'g-xt' if a <= i < b else ('g-yt' if b <= i < c else 'g-cell')
            f.rect(x0 + i * cw + 1.5, y, cw - 3, h, cls, 2)

    y1, y2 = 62, 150
    strip(y1, 9, 17, 22)
    xn = x0 + 17 * cw
    f.line(xn, y1 - 20, xn, y1 + h + 8, 'g-now')
    f.text(xn, y1 - 27, L('预测起点 t：第 t 行刚观测到', 'forecast origin t: row t has just been observed'), 'g-k', 'middle')
    f.brace(x0 + 9 * cw + 1, xn - 3, y1 + h + 10)
    f.brace(xn + 3, x0 + 22 * cw - 1, y1 + h + 10)
    f.text(xn - 8, y1 + h + 28, L('输入：最近 W 行（t−W+1 … t）', 'input: the last W rows (t−W+1 … t)'), '', 'end')
    f.text(xn + 8, y1 + h + 28, L('目标：之后 H 行（t+1 … t+H）', 'targets: the next H rows (t+1 … t+H)'))
    f.text(16, y1 + 17, L('起点 t', 'origin t'), 'g-b')
    strip(y2, 10, 18, 23)
    f.text(16, y2 + 17, L('起点 t+1', 'origin t+1'), 'g-b')
    f.text(x0, y2 + h + 20, L('下一个样本整体右移一行，和上一个几乎完全重叠', 'the next sample is shifted by one row and overlaps the previous one almost entirely'))
    f.arrow([(x0, 216), (x0 + n * cw, 216)])
    f.text(x0 - 8, 220, L('时间', 'time'), 'g-s', 'end')
    return f


# ------------------------------------------------------------------ figure 2: the two splits on the real series

def etth1_daily(path):
    """Daily mean of ETTh1's last column (oil temperature) over the 14,400 rows the study uses."""
    import pandas as pd
    v = pd.read_csv(path)['OT'].to_numpy(dtype='float64')[:14400]
    return v.reshape(600, 24).mean(axis=1)


def fig_splits(daily):
    f = Fig('splits', 760, 322, L('ETTh1 的同一条序列：传统划分 60/20/20，在线划分 20/5/75。',
                                  'The same ETTh1 series: traditional split 60/20/20, online split 20/5/75.'))
    x0, x1, N = 168, 740, 14400
    X = lambda row: x0 + (x1 - x0) * row / N
    month = L(' 个月', ' months')

    def bands(y, b, texts, name, sub, below=None):
        f.text(16, y + 16, name, 'g-b')
        f.text(16, y + 32, sub, 'g-s')
        for k, cls in enumerate(('g-train', 'g-val', 'g-test')):
            a, c = X(b[k]), X(b[k + 1])
            f.rect(a + 1, y, c - a - 2, 38, cls, 3)
            if texts[k]:
                f.text((a + c) / 2, y + 16, texts[k][0], 'g-k', 'middle')
                f.text((a + c) / 2, y + 31, texts[k][1], 'g-s', 'middle')
        if below:
            a, c = X(b[1]), X(b[2])
            f.line((a + c) / 2, y + 38, (a + c) / 2, y + 46, 'g-ln')
            f.text((a + c) / 2, y + 60, below, '', 'middle')

    T, O = [0, 8640, 11520, 14400], [0, 2880, 3600, 14400]
    yT, top, bot, yO = 30, 94, 184, 226
    bands(yT, T, [(L('训练 8,640 行', 'TRAIN 8,640 rows'), '60% · 12' + month),
                  (L('验证 2,880 行', 'VAL 2,880'), '20% · 4' + month),
                  (L('测试 2,880 行', 'TEST 2,880'), '20% · 4' + month)],
          L('T 划分', 'T split'), L('传统 · 60 / 20 / 20', 'traditional · 60/20/20'))
    lo, hi = -10, 50
    Y = lambda v: bot - (bot - top) * (v - lo) / (hi - lo)
    for tick in (0, 20, 40):
        f.line(x0, Y(tick), x1, Y(tick), 'g-grid')
        f.text(x0 - 8, Y(tick) + 4, str(tick), 'g-m', 'end')
    for b in T[1:3]:
        f.line(X(b), yT + 38, X(b), bot, 'g-guide')
    for b in O[1:3]:
        f.line(X(b), top, X(b), yO, 'g-guide')
    f.path('M' + ' L'.join(f'{X((d + 0.5) * 24):.1f},{Y(v):.1f}' for d, v in enumerate(daily)), 'g-spark')
    f.text(16, top + 14, L('油温', 'oil temperature'), 'g-b')
    f.text(16, top + 30, L('OT 列，日均', 'column OT, daily mean'), 'g-s')
    f.line(x0, bot, x1, bot, 'g-ax')
    for label, day in (('2016-07', 0), ('2016-10', 92), ('2017-01', 184), ('2017-04', 274), ('2017-07', 365),
                       ('2017-10', 457), ('2018-01', 549)):
        f.line(X(day * 24), bot, X(day * 24), bot + 4, 'g-ax')
        f.text(X(day * 24), bot + 16, label, 'g-m', 'start' if day == 0 else 'middle')
    bands(yO, O, [(L('训练 2,880 行', 'TRAIN 2,880'), '20% · 4' + month), None,
                  (L('测试 10,800 行', 'TEST 10,800 rows'), '75% · 15' + month)],
          L('O 划分', 'O split'), L('在线 · 20 / 5 / 75', 'online · 20/5/75'),
          below=L('验证 720 行 · 5% · 1 个月', 'VALIDATION 720 rows · 5% · 1 month'))
    return f


# ------------------------------------------------------------------ figure 3: forecasts at a segment boundary

def fig_boundary():
    f = Fig('boundary', 760, 262, L('测试期的第一个和最后一个预测：目标都在测试段内，输入可以伸进前面的段。',
                                    'The first and the last TEST forecast: all targets lie in TEST, the input may reach back into earlier segments.'))
    s = 0.275                                   # px per row
    xb = 432                                    # row 3,600
    X = lambda row: xb + (row - 3600) * s
    f.rect(150, 36, 81, 24, 'g-train', 3)
    f.rect(233, 36, 198, 24, 'g-val', 3)
    f.rect(433, 36, 131, 24, 'g-test', 3)
    f.rect(590, 36, 150, 24, 'g-test', 3)
    f.text(190, 52, L('训练段', 'TRAIN'), 'g-k', 'middle')
    f.text(332, 52, L('验证段 2,880–3,599', 'VALIDATION 2,880–3,599'), 'g-k', 'middle')
    f.text(498, 52, L('测试段 3,600–', 'TEST 3,600–'), 'g-k', 'middle')
    f.text(665, 52, '–14,399', 'g-k', 'middle')
    for x in (572, 580):
        f.line(x - 3, 64, x + 3, 32, 'g-ln')
    f.line(xb, 24, xb, 214, 'g-now')
    f.text(xb, 18, L('测试从第 3,600 行开始', 'TEST starts at row 3,600'), 'g-k', 'middle')
    # row a: W = 96
    ya = 84
    f.text(16, ya + 9, L('第一个测试预测', 'first TEST forecast'), 'g-b')
    f.text(16, ya + 25, L('起点 3,599 · W = 96', 'origin 3,599 · W = 96'), 'g-s')
    f.rect(X(3504), ya, 96 * s - 1.5, 22, 'g-xt', 2)
    f.rect(xb + 1.5, ya, 96 * s - 1.5, 22, 'g-yt', 2)
    f.text(X(3504) - 6, ya + 15, L('输入 3,504–3,599', 'input 3,504–3,599'), '', 'end')
    f.text(X(3696) + 6, ya + 15, L('目标 3,600–3,695', 'targets 3,600–3,695'))
    # row b: W = 1024
    yb = 132
    f.text(16, yb + 9, L('同一个预测', 'the same forecast'), 'g-b')
    f.text(16, yb + 25, L('冻结模型 · W = 1024', 'frozen, W = 1024'), 'g-s')
    f.rect(X(2576), yb, 1024 * s - 1.5, 22, 'g-xt', 2)
    f.rect(xb + 1.5, yb, 96 * s - 1.5, 22, 'g-yt', 2)
    f.text((X(2576) + xb) / 2, yb + 15, L('输入 2,576–3,599，伸进训练段', 'input 2,576–3,599, reaches into TRAIN'), 'g-k', 'middle')
    f.text(X(3696) + 6, yb + 15, L('目标相同', 'same targets'))
    # row c: the last forecast
    yc = 180
    X2 = lambda row: 740 + (row - 14400) * s
    f.text(16, yc + 9, L('最后一个测试预测', 'last TEST forecast'), 'g-b')
    f.text(16, yc + 25, L('起点 14,303 · W = 96', 'origin 14,303 · W = 96'), 'g-s')
    f.rect(X2(14208), yc, 96 * s - 1.5, 22, 'g-xt', 2)
    f.rect(X2(14304) + 1.5, yc, 96 * s - 2, 22, 'g-yt', 2)
    f.text(X2(14208) - 6, yc + 15, L('输入 14,208–14,303，目标 14,304–14,399', 'input 14,208–14,303, targets 14,304–14,399'), '', 'end')
    f.text(16, 246, L('测试预测的个数 = 10,800 − 96 + 1 = 10,705，和窗口长度无关：所有模型在同一批预测上评分。',
                      'TEST forecasts: 10,800 − 96 + 1 = 10,705 for every window, so all models are scored on the same forecasts.'), 'g-k')
    return f


# ------------------------------------------------------------------ figure 4: offline training

def fig_offline():
    f = Fig('offline', 760, 214, L('离线训练：训练段的样本打乱后分批，重复 50 轮，保留最后一轮。',
                                   'Offline training: TRAIN samples are shuffled into batches, for 50 epochs; the last epoch is kept.'))
    y, h = 28, 52
    f.box(16, y, 184, h, L('训练样本', 'training samples'), L('各窗口共用的 1,761 个', '1,761, shared by all windows'))
    f.text(16, 172, L('图里的数字：', 'Numbers in this figure:'), 'g-s')
    f.text(16, 188, L('ETTh1 · O 划分 · H = 96', 'ETTh1, O split, H = 96'), 'g-s')
    f.box(224, y, 130, h, L('每轮重新打乱', 'reshuffle every epoch'), L('不保留时间顺序', 'time order is dropped'))
    f.box(378, y, 114, h, L('每批 32 个', 'batches of 32'), L('每轮 56 批', '56 per epoch'))
    f.box(516, y, 228, h, L('前向 → MSE → Adam 一步', 'forward → MSE → one Adam step'), L('只对观测到的目标算损失', 'loss on observed targets only'))
    for a, b in ((200, 224), (354, 378), (492, 516)):
        f.arrow([(a, y + h / 2), (b, y + h / 2)])
    f.arrow([(630, y + h), (630, 106), (289, 106), (289, y + h)])
    f.text(372, 124, L('重复 50 轮 = 2,800 步，不早停', 'repeat for 50 epochs = 2,800 steps, no early stopping'), 'g-k')
    y2, h2 = 150, 50
    f.box(224, y2, 268, h2, L('每轮结束：在验证段算一次 MSE', 'after each epoch: MSE on VALIDATION'),
          L('只用来挑学习率，不参与训练', 'used only to choose the learning rate'))
    f.box(516, y2, 228, h2, L('保留第 50 轮的参数', 'keep the parameters of epoch 50'), L('冻结模型到此定型', 'the frozen model is final here'), 'g-fzt')
    f.arrow([(358, 106), (358, y2)])
    f.arrow([(492, y2 + h2 / 2), (516, y2 + h2 / 2)])
    return f


# ------------------------------------------------------------------ figure 5: when the parameters change

def fig_lifecycle():
    f = Fig('lifecycle', 760, 334, L('三种模型的参数在什么时候改变：冻结模型只在训练段上训练；在线模型从验证期开始每步更新；同起点副本在测试开始时冻结。',
                                     'When the parameters change: the frozen model is only trained on TRAIN; the online model is updated at every step from VALIDATION on; the matched copy is frozen when TEST starts.'))
    a, b, c, d = 176, 338, 448, 744
    for x1, x2, cls, name, rows in ((a, b, 'g-train', L('训练段', 'TRAIN'), '0 – 2,879'),
                                     (b, c, 'g-val', L('验证段', 'VALIDATION'), '2,880 – 3,599'),
                                     (c, d, 'g-test', L('测试段', 'TEST'), '3,600 – 14,399')):
        f.rect(x1 + 1, 24, x2 - x1 - 2, 24, cls, 3)
        f.text((x1 + x2) / 2, 40, name, 'g-k', 'middle')
        f.text((x1 + x2) / 2, 64, rows, 'g-m', 'middle')
    f.text(16, 40, L('ETTh1 · O 划分', 'ETTh1 · O split'), 'g-k')
    f.text(16, 64, L('行号（未按比例）', 'rows (not to scale)'), 'g-s')
    for x in (b, c):
        f.line(x, 72, x, 300, 'g-guide')

    def ticks(x1, x2, y, tone):
        d_ = ''.join(f'M{x:.1f},{y + 6:.1f} L{x:.1f},{y + 26:.1f} ' for x in [x1 + 6 + 6.5 * k for k in range(int((x2 - x1 - 8) / 6.5))])
        f.path(d_.strip(), f'g-tick {tone}')

    y = 86
    f.text(16, y + 14, L('冻结模型', 'Frozen model'), 'g-b')
    f.text(16, y + 30, 'W = 96 … 1024', 'g-s')
    f.rect(a + 3, y, b - a - 6, 32, 'g-fzt', 4)
    f.text((a + b) / 2, y + 20, L('离线训练 50 轮', 'offline, 50 epochs'), 'g-k', 'middle')
    f.rect(b + 3, y, c - b - 6, 32, 'g-fut', 4)
    f.text((b + c) / 2, y + 20, L('只挑学习率', 'LR choice only'), 'g-s', 'middle')
    f.line(c + 4, y + 16, d - 2, y + 16, 'g-ln fz g-thick')
    f.text(c + 8, y + 9, L('参数不变，只预测', 'parameters fixed: forecasts only'), 'g-k')
    y = 156
    f.text(16, y + 14, L('在线模型', 'Online model'), 'g-b')
    f.text(16, y + 30, L('W = 96 · Naive、ER', 'W = 96 · Naive, ER'), 'g-s')
    f.rect(a + 3, y, b - a - 6, 32, 'g-olt', 4)
    f.text((a + b) / 2, y + 20, L('离线训练 50 轮', 'offline, 50 epochs'), 'g-k', 'middle')
    ticks(b, c, y, 'ol')
    ticks(c, d, y, 'ol')
    f.text(b + 6, y + 47, L('预热', 'warm-up'), 'g-k')
    f.text(b + 6, y + 62, L('720 次更新', '720 updates'), 'g-k')
    f.text(c + 8, y + 47, L('每步先更新、再预测', 'at every step: update, then forecast'), 'g-k')
    f.text(c + 8, y + 62, L('10,705 次更新', '10,705 updates'), 'g-k')
    y = 240
    f.text(16, y + 14, L('同起点副本', 'Matched copy'), 'g-b')
    f.text(16, y + 30, L('Naive-frozen 等', 'e.g. Naive-frozen'), 'g-s')
    f.rect(a + 3, y, b - a - 6, 32, 'g-olt', 4)
    f.text((a + b) / 2, y + 20, L('离线训练 50 轮', 'offline, 50 epochs'), 'g-k', 'middle')
    ticks(b, c, y, 'ol')
    f.circle(c, y + 16, 4.5, 'g-lnh ol')
    f.line(c + 6, y + 16, d - 2, y + 16, 'g-ln ol g-thick')
    f.text(c + 8, y + 9, L('从这一刻起不再更新', 'no update from here on'), 'g-k')
    f.text(b + 6, y + 47, L('同样的预热', 'same warm-up'), 'g-k')
    f.text(16, 322, L('PROCEED 只有预热不同：从 t = 2,975 起每步成对更新，625 步共 1,250 次。测试期也是每步一次。',
                      'PROCEED differs in the warm-up only: pairs of updates from t = 2,975 on, 625 steps and 1,250 updates. In TEST, one update per step.'), 'g-s')
    return f


# ------------------------------------------------------------------ figure 6: the three actions at time t

def _strip(f, x0, cw, r0, r1, y, t, h=24):
    for r in range(r0, r1 + 1):
        f.rect(x0 + (r - r0) * cw + 1.5, y, cw - 3, h, 'g-cell' if r <= t else 'g-fut', 2)
        f.text(x0 + (r - r0) * cw + cw / 2, y + h / 2 + 4, str(r), 'g-m', 'middle')


def fig_schedule():
    f = Fig('schedule', 760, 398, L('每个时刻按顺序做三件事：给到齐的预测评分，用刚到齐的样本更新，发出新预测。玩具例子：W = 4，H = 3。',
                                    'Three actions at every step, in this order: score the forecast that is now complete, update with the sample that is now complete, issue a new forecast. Toy example: W = 4, H = 3.'))
    x0, cw, r0, r1 = 190, 34, 9, 24
    cx = lambda r: x0 + (r - r0) * cw

    def bar(a, b, y, cls, label, tcls='g-k'):
        f.rect(cx(a) + 1.5, y, cx(b + 1) - cx(a) - 3, 22, cls, 3)
        f.text((cx(a) + cx(b + 1)) / 2, y + 15, label, tcls, 'middle')

    def panel(t, top):
        u = t - 3
        f.text(16, top + 6, L(f'时刻 t = {t}：第 {t} 行刚观测到', f'time t = {t}: row {t} has just been observed'), 'g-b')
        f.text(cx(t + 1) + 8, top + 6, L('以后的行还没发生', 'later rows do not exist yet'), 'g-s')
        _strip(f, x0, cw, r0, r1, top + 16, t)
        f.line(cx(t + 1), top - 4, cx(t + 1), top + 152, 'g-now')
        y = top + 54
        f.text(16, y + 9, L('① 评分', '1  score'), 'g-b')
        f.text(16, y + 25, L(f'{u} 时发出、预测第 {u + 1}–{t} 行', f'issued at {u} for rows {u + 1}–{t}'), 'g-s')
        bar(u + 1, t, y, 'g-p', L('到齐的预测', 'now complete'))
        y = top + 90
        f.text(16, y + 9, L('② 更新', '2  update'), 'g-b')
        f.text(16, y + 25, L(f'用样本 {u} 走一步梯度', f'one gradient step on sample {u}'), 'g-s')
        bar(u - 3, u, y, 'g-xt', L('输入', 'input'))
        bar(u + 1, t, y, 'g-yt', L('真实值', 'true values'))
        y = top + 126
        f.text(16, y + 9, L('③ 预测', '3  forecast'), 'g-b')
        f.text(16, y + 25, L(f'发出对第 {t + 1}–{t + 3} 行的预测', f'issue the forecast of rows {t + 1}–{t + 3}'), 'g-s')
        bar(t - 3, t, y, 'g-xt', L('输入', 'input'))
        bar(t + 1, t + 3, y, 'g-p', L('新预测', 'new forecast'))

    panel(18, 22)
    panel(19, 220)
    return f


# ------------------------------------------------------------------ figure 7: the leak

def fig_leak():
    f = Fig('leak', 760, 372, L('合法的更新只用已经观测到的行。泄漏的做法在预测后立刻用这个预测自己的真实值更新，这些值当时还不存在。',
                                'An allowed update uses observed rows only. The leaky schedule updates right after forecasting with that forecast\'s own targets, which do not exist yet.'))
    x0, cw, r0, r1, t = 226, 36, 11, 24, 18
    cx = lambda r: x0 + (r - r0) * cw

    def bar(a, b, y, cls, label, tcls='g-k'):
        f.rect(cx(a) + 1.5, y, cx(b + 1) - cx(a) - 3, 22, cls, 3)
        f.text((cx(a) + cx(b + 1)) / 2, y + 15, label, tcls, 'middle')

    def head(top, title):
        f.text(16, top + 6, title, 'g-b')
        _strip(f, x0, cw, r0, r1, top + 16, t)

    top = 22
    head(top, L('合法：真实值到齐了才更新（t = 18）', 'Allowed: update after the targets have arrived (t = 18)'))
    f.line(cx(t + 1), top + 8, cx(t + 1), top + 118, 'g-now')
    y = top + 54
    f.text(16, y + 9, L('先更新', 'update first'), 'g-b')
    f.text(16, y + 25, L('样本 15：目标最晚是第 18 行', 'sample 15: last target is row 18'), 'g-s')
    bar(12, 15, y, 'g-xt', L('输入', 'input'))
    bar(16, 18, y, 'g-yt', L('已观测到', 'observed'))
    y = top + 90
    f.text(16, y + 9, L('再预测', 'then forecast'), 'g-b')
    f.text(16, y + 25, L('第 19–21 行', 'rows 19–21'), 'g-s')
    bar(15, 18, y, 'g-xt', L('输入', 'input'))
    bar(19, 21, y, 'g-p', L('预测', 'forecast'))

    top = 178
    head(top, L('泄漏：预测完立刻用它自己的真实值更新（t = 18）', 'Leaky: update at once with the forecast\'s own targets (t = 18)'))
    f.line(cx(t + 1), top + 8, cx(t + 1), top + 118, 'g-now')
    y = top + 54
    f.text(16, y + 9, L('先预测', 'forecast first'), 'g-b')
    f.text(16, y + 25, L('第 19–21 行', 'rows 19–21'), 'g-s')
    bar(15, 18, y, 'g-xt', L('输入', 'input'))
    bar(19, 21, y, 'g-p', L('预测', 'forecast'))
    y = top + 90
    f.text(16, y + 9, L('马上更新', 'update at once'), 'g-b')
    f.text(16, y + 25, L('样本 18：目标是第 19–21 行', 'sample 18: targets are rows 19–21'), 'g-s')
    bar(15, 18, y, 'g-xt', L('输入', 'input'))
    bar(19, 21, y, 'g-badt', L('还没发生', 'not observed'))
    y = top + 134
    f.text(16, y + 9, L('下一步 t = 19 的预测', 'next forecast, t = 19'), 'g-b')
    f.text(16, y + 25, L('要预测第 20–22 行', 'rows 20–22'), 'g-s')
    bar(16, 19, y, 'g-xt', L('输入', 'input'))
    bar(20, 22, y, 'g-p', L('预测', 'forecast'))
    f.line(cx(20), y - 6, cx(20), y + 28, 'g-now')
    f.brace(cx(20) + 3, cx(22) - 3, y + 30, tone='bad')
    f.text(cx(21), y + 48, L('这 2 行刚当过训练标签（H − 1 行）', 'these 2 rows were just used as labels (H − 1 rows)'), 'g-k', 'middle')
    return f


# ------------------------------------------------------------------ figure 8: one experiment from start to end

def fig_pipeline():
    f = Fig('pipeline', 760, 258, L('一次实验的五个步骤和收尾，下面一行是 etth1_quick 的作业数。',
                                    'The five steps of one experiment and the wrap-up; the bottom line of each box is the job count of etth1_quick.'))
    f.text(16, 18, 'python -m olts.run configs/etth1_quick.json', 'g-m')
    steps = [
        (L('学习率候选', 'LR candidates'), L('种子 2024', 'seed 2024'), L('窗口 × 3 个学习率', 'windows × 3 LRs'), L('30 个训练作业', '30 fits')),
        (L('选学习率', 'LR choice'), L('验证段 MSE', 'VALIDATION MSE'), L('跨窗口的平均名次', 'best mean rank'), L('两个划分都是 1e-4', '1e-4 in both splits')),
        (L('其余训练', 'Other fits'), L('其余种子', 'the other seeds'), L('PROCEED 初始化', 'PROCEED init'), L('新增 2 个作业', '2 new jobs')),
        (L('选在线倍率', 'Multiplier'), L('可选步骤', 'optional'), L('验证段上的在线流', 'on VALIDATION'), L('跳过', 'skipped')),
        (L('测试流', 'TEST streams'), L('冻结：每档窗口', 'frozen: every window'), L('在线 + 同起点副本', 'online + copies'), L('22 条流', '22 streams')),
    ]
    w, gap, y, h = 136, 12, 32, 104
    for k, (title, l1, l2, count) in enumerate(steps):
        x = 16 + k * (w + gap)
        f.rect(x, y, w, h, 'g-card', 6)
        f.circle(x + 17, y + 19, 10, 'g-dot')
        f.text(x + 17, y + 23, str(k + 1), 'g-n', 'middle')
        f.text(x + 34, y + 23.5, title, 'g-b')
        f.text(x + 10, y + 48, l1, '')
        f.text(x + 10, y + 65, l2, '')
        f.line(x + 10, y + 76, x + w - 10, y + 76, 'g-grid')
        f.text(x + 10, y + 93, count, 'g-k')
        if k:
            f.arrow([(x - gap, y + h / 2), (x, y + h / 2)], head=6)
    x5 = 16 + 4 * (w + gap) + w / 2
    f.arrow([(x5, y + h), (x5, 152), (130, 152), (130, 170)])
    tail = [(L('汇总', 'collect'), 'streams.csv, fits.csv'),
            (L('分析：配对、取中位数', 'analyse: pairs and medians'), 'pairs.csv, summary.json'),
            (L('报告初稿', 'draft report'), 'report_zh.html, report_en.html')]
    for k, (title, files) in enumerate(tail):
        x = 16 + k * 250
        f.rect(x, 170, 228, 56, 'g-soft', 6)
        f.text(x + 114, 192, title, 'g-b', 'middle')
        f.text(x + 114, 211, files, 'g-m', 'middle')
        if k:
            f.arrow([(x - 22, 198), (x, 198)], head=6)
    f.text(16, 246, L('这些文件都写在 runs/etth1_quick/results/ 里', 'these files are written to runs/etth1_quick/results/'), 'g-s')
    return f


FIGURES = ('sample', 'splits', 'boundary', 'offline', 'lifecycle', 'schedule', 'leak', 'pipeline')


def build(name, daily=None):
    if name == 'splits':
        return fig_splits(daily)
    return globals()['fig_' + name]()


def main():
    global LANG
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--data', default=str(ROOT / 'data' / 'ETTh1.csv'))
    ap.add_argument('--out', default=str(Path(__file__).resolve().parent))
    a = ap.parse_args()
    if not Path(a.data).is_file():
        sys.exit(f'{a.data} not found: run bash scripts/download_data.sh ETTh1 first')
    daily = etth1_daily(a.data)
    Path(a.out).mkdir(parents=True, exist_ok=True)
    for LANG in ('en', 'zh'):
        for name in FIGURES:
            p = Path(a.out) / f'{name}.{LANG}.svg'
            p.write_text(build(name, daily).svg(standalone=True), encoding='utf-8')
            print('wrote', p)


if __name__ == '__main__':
    main()
