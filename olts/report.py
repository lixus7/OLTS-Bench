"""Draft report (HTML) from results/summary.json, in Chinese or English.

    python -m olts.report runs/etth1_quick --lang zh
    python -m olts.report --summary results/full_study/check/summary.json --lang en --out full_study_en.html

The draft fills in every number, table caption and "who is compared with whom" sentence.
The verdict, the explanations and the "can say / cannot say yet" lists are yours to write:
see docs/REPORTING.md. Print to PDF from a browser (A4) or with scripts/html_to_pdf.sh.
"""
import argparse
import html
from datetime import date
from pathlib import Path
from .common import read

T = {
    'zh': dict(
        title='短窗口在线学习 vs 长窗口冻结模型', draft='自动生成的初稿', run='实验',
        verdict_h='一句话结论（草稿：请核对后用自己的话重写）',
        verdict_cell='{sp} 划分：{n} 个比较（{m} 种在线方法 × {w} 档冻结窗口）里，冻结更好的 {neg} 个，在线更好的 {pos} 个；中位数从在线误差{lo}到{hi}。',
        sec='第 {n} 节 · ', setup_h='实验设置', read_h='怎么读这些数',
        read_gain='<b>收益</b> = (冻结误差 − 在线误差) ÷ 冻结误差。正数表示在线更好：+20% 是在线误差比冻结低 20%；负数表示冻结更好：−20% 是在线误差比冻结高 20%；−250% 是在线误差是冻结的 3.5 倍。收益最高 +100%，往下没有下限，所以表里都用中位数。',
        read_pair='<b>一组配对</b> = 两次测试运行，数据集、骨干、划分、预测长度 H、种子都相同，在同一批预测时刻上各算一个 MSE，得到一个收益值。',
        read_example='例：{dataset} · {model} · {split} 划分 · H = {H} · 种子 {seed} · {method}：在线 W{W} 的 MSE 是 {online_MSE}，冻结 W{ref_W} 是 {ref_MSE}，收益 = ({ref_MSE} − {online_MSE}) ÷ {ref_MSE} = {gain_s}%。',
        read_cells='表里每格写中位数，小字“k/n”是 n 组配对里在线更好的组数。蓝色：在线更好；橙色：冻结更好；灰色：相差 ±5% 以内。',
        main_h='在线 W{ow} 对各档冻结窗口', main_cap='表 {k}　{sp} 划分：在线 W{ow} 对冻结 W 的收益中位数',
        sent_h='逐格读法（自动生成，可直接引用或改写）',
        sent_neg='{sp} 划分 · {m} 对冻结 W{W}：在线误差比冻结高 {v:.1f}%（{n} 组里 {pos} 组在线更好）。',
        sent_pos='{sp} 划分 · {m} 对冻结 W{W}：在线误差比冻结低 {v:.1f}%（{n} 组里 {pos} 组在线更好）。',
        matched_h='继续更新本身有没有用（同一起点）',
        matched_how='测试开始时把在线模型复制一份冻结起来（“同起点冻结副本”）。两份起点完全一样，差别只在测试期是否继续更新。正数表示继续更新那份误差更低。',
        matched_cap='表 {k}　继续更新那份对同起点冻结副本的收益', col_method='在线方法', col_split='划分',
        col_med='收益中位数', col_pos='继续更新更好', col_pos_online='在线更好', all_x='全部 ×{x:g}', col_over='误差超过副本 10 倍',
        w96_h='输入一样长：在线 W{ow} 对冻结 W{ow}', w96_cap='表 {k}　在线 W{ow} 对冻结 W{ow}（只差部署后是否更新）',
        ds_h='分数据集', ds_cap='表 {k}　{m} 对冻结 W{W}，分数据集（收益中位数）',
        blk_h='测试期前段和后段', blk_cap='表 {k}　测试期按时间切成 10 段：前 3 段和后 3 段的收益中位数',
        blk_how='O 划分测试期占序列后 75%，后 3 段大约是整条序列的最后 22.5%，和 T 划分的测试期差不多是同一段时间。',
        col_first='前 30%', col_last='后 30%', col_whole='整段',
        cost_h='成本', cost_cap='表 {k}　每次预测的用时（毫秒）：在线 W{ow} 对冻结 W{ow}',
        col_online_ms='在线', col_frozen_ms='冻结', col_ratio='倍数', train_cap='表 {k}　离线训练用时中位数（秒）',
        claims_h='能说什么，还不能说什么（请自己写）',
        claims_yes='能说（每条都指向表号，写清跟谁比、正负是好是坏）', claims_no='还不能说（数据撑不住的、范围外的、只是猜测的）',
        claims_hint='写法见 docs/REPORTING.zh-CN.md。不要把猜测写成结论。',
        notes_h='说明', notes='中位数只作描述，不是统计检验；同一数据集的配对共用一段测试数据，彼此相关。不跨划分比较绝对误差（O、T 的测试期和训练量都不同）。',
        none='（无数据）', groups='组', lr_chosen='选中的离线学习率', mult_chosen='选中的在线倍率',
        set_data='数据与划分', set_models='骨干', set_h='预测长度 H', set_seeds='种子', set_windows='冻结窗口',
        set_methods='在线方法（输入 {ow} 步）', set_budget='训练预算', set_rule='因果规则',
        budget_C='FP32、batch 32、Adam、MSE、训练 50 轮、不早停、取最后一轮；训练起点按场景最长窗口对齐',
        budget_T='FP32、batch 32、Adam、MSE、最多 50 轮、验证误差连续 5 轮不降就停、取最好一轮',
        budget_smoke='只训练 {n} 轮（冒烟测试：数字没有意义）',
        rule='在 u 时刻发出的预测，等它的 H 个真实值全部到齐（t ≥ u + H）才评分、才拿来更新。',
        rows='行：TRAIN 到 {nt}，VALIDATION 到 {nv}，TEST 到 {end}',
    ),
    'en': dict(
        title='Short-window online learning vs long-window frozen models', draft='auto-generated draft', run='Run',
        verdict_h='One-line answer (draft: check it, then rewrite it in your own words)',
        verdict_cell='{sp} split: of {n} comparisons ({m} online methods x {w} frozen windows), the frozen model is better in {neg} and the online model in {pos}; medians range from {lo} to {hi} for the online error.',
        sec='Section {n} · ', setup_h='Setup', read_h='How to read the numbers',
        read_gain='<b>Gain</b> = (frozen error - online error) / frozen error. Positive means the online model is better: +20% means 20% lower error than the frozen model; -20% means 20% higher error; -250% means 3.5 times the error. Gains are at most +100% and unbounded below, so every table uses medians.',
        read_pair='<b>A pair</b> = two TEST runs with the same dataset, backbone, split, horizon H and seed, each scored by MSE on the same forecast origins; it gives one gain.',
        read_example='Example: {dataset} · {model} · {split} split · H = {H} · seed {seed} · {method}: online W{W} has MSE {online_MSE}, frozen W{ref_W} has {ref_MSE}, so gain = ({ref_MSE} − {online_MSE}) / {ref_MSE} = {gain_s}%.',
        read_cells='Each cell shows the median; the small "k/n" says in how many of the n pairs the online model was better. Blue: online better; orange: frozen better; grey: within +-5%.',
        main_h='Online W{ow} against each frozen window', main_cap='Table {k}　{sp} split: median gain of online W{ow} over frozen W',
        sent_h='Cell by cell (auto-generated; quote or rewrite)',
        sent_neg='{sp} split · {m} vs frozen W{W}: the online error is {v:.1f}% higher than the frozen error (online better in {pos} of {n} pairs).',
        sent_pos='{sp} split · {m} vs frozen W{W}: the online error is {v:.1f}% lower than the frozen error (online better in {pos} of {n} pairs).',
        matched_h='Does updating itself help? (same starting point)',
        matched_how='At the start of TEST the online model is copied and the copy is frozen. Both start from the same state; the only difference is whether updating continues during TEST. Positive means the updating copy has lower error.',
        matched_cap='Table {k}　Updating copy vs its frozen twin', col_method='Online method', col_split='Split',
        col_med='Median gain', col_pos='Updating better', col_pos_online='Online better', all_x='all ×{x:g}', col_over='Error > 10x the twin',
        w96_h='Same input length: online W{ow} vs frozen W{ow}', w96_cap='Table {k}　Online W{ow} vs frozen W{ow} (differ only in updating after deployment)',
        ds_h='By dataset', ds_cap='Table {k}　{m} vs frozen W{W} by dataset (median gain)',
        blk_h='Early and late in the TEST period', blk_cap='Table {k}　TEST split into 10 time blocks: median gain on the first 3 and the last 3',
        blk_how='The O split tests on the last 75% of the series, so its last 3 blocks are roughly the last 22.5% of the series, about the same stretch of time as the T split\'s TEST period.',
        col_first='first 30%', col_last='last 30%', col_whole='whole',
        cost_h='Cost', cost_cap='Table {k}　Time per forecast (ms): online W{ow} vs frozen W{ow}',
        col_online_ms='online', col_frozen_ms='frozen', col_ratio='ratio', train_cap='Table {k}　Median offline training time (s)',
        claims_h='What we can say, and what we cannot say yet (write this yourself)',
        claims_yes='Can say (each point names its table, who is compared with whom, and whether + is better or worse)',
        claims_no='Cannot say yet (not supported by the data, out of scope, or only a guess)',
        claims_hint='See docs/REPORTING.md. Never present a guess as a finding.',
        notes_h='Notes', notes='Medians are descriptive, not significance tests; pairs from the same dataset share TEST data and are correlated. Absolute errors are not compared across splits (O and T differ in TEST period and training size).',
        none='(no data)', groups='pairs', lr_chosen='chosen offline learning rate', mult_chosen='chosen online multiplier',
        set_data='Data and splits', set_models='Backbones', set_h='Horizon H', set_seeds='Seeds', set_windows='Frozen windows',
        set_methods='Online methods (input {ow} steps)', set_budget='Training budget', set_rule='Causality',
        budget_C='FP32, batch 32, Adam, MSE loss, 50 epochs, no early stopping, last epoch kept; training origins aligned to the longest window of the scenario',
        budget_T='FP32, batch 32, Adam, MSE loss, up to 50 epochs, stop after 5 epochs without validation improvement, best epoch kept',
        budget_smoke='only {n} epochs (smoke test: the numbers mean nothing)',
        rule='A forecast issued at time u is scored, and used for updating, only when all H of its targets have arrived (t >= u + H).',
        rows='rows: TRAIN to {nt}, VALIDATION to {nv}, TEST to {end}',
    ),
}

CSS = """
:root { --ink:#1d2433; --muted:#5b6475; --rule:#d9dde5; --b3:#2f6fbf; --b2:#6f9fd8; --b1:#b9d0ee; --n0:#eceef2;
        --o1:#f4cfb2; --o2:#e99a64; --o3:#c8602a; }
body { background:#fff; color:var(--ink); font:15px/1.7 "Noto Sans SC","IBM Plex Sans",system-ui,sans-serif; margin:0; }
main { max-width:980px; margin:0 auto; padding:28px 18px 60px; }
h1 { font-size:1.6rem; line-height:1.3; margin:.2rem 0 .4rem; } h2 { font-size:1.15rem; margin:2rem 0 .6rem; border-top:1px solid var(--rule); padding-top:1rem; }
h3 { font-size:1rem; margin:1.2rem 0 .4rem; } .eyebrow { color:var(--muted); font-size:.8rem; letter-spacing:.04em; }
.answer { background:#f4f6fa; border-radius:10px; padding:12px 16px; } .note { color:var(--muted); font-size:.85rem; }
table { border-collapse:collapse; margin:.4rem 0 1rem; font-variant-numeric:tabular-nums; }
caption { text-align:left; font-weight:600; padding-bottom:.3rem; } th, td { padding:.35rem .6rem; border-bottom:1px solid var(--rule); text-align:right; }
th:first-child, td:first-child { text-align:left; } td small { display:block; font-size:.72rem; opacity:.8; }
.heat td.v { border-radius:4px; border:2px solid #fff; min-width:64px; }
.b3 { background:var(--b3); color:#fff; } .b2 { background:var(--b2); } .b1 { background:var(--b1); } .n0 { background:var(--n0); }
.o1 { background:var(--o1); } .o2 { background:var(--o2); } .o3 { background:var(--o3); color:#fff; }
.write { border:1px dashed var(--muted); border-radius:8px; padding:10px 14px; color:var(--muted); }
.tw { overflow-x:auto; } dl { display:grid; grid-template-columns:11rem 1fr; gap:.3rem 1rem; } dt { color:var(--muted); }
@media (max-width:640px) { dl { grid-template-columns:1fr; } }
@page { size:A4; margin:14mm 12mm; }
@media print { * { -webkit-print-color-adjust:exact; print-color-adjust:exact; } table, .answer { break-inside:avoid; } h2, h3 { break-after:avoid; } }
"""


def band(v):
    if v is None:
        return ''
    return 'b3' if v >= 25 else 'b2' if v >= 10 else 'b1' if v >= 5 else 'n0' if v > -5 else 'o1' if v > -10 else 'o2' if v > -25 else 'o3'


def sgn(v):
    return '—' if v is None else (f'+{v:.1f}' if v > 0 else f'−{abs(v):.1f}' if v < 0 else '0.0')


def build(run=None, lang='zh', summary=None, out=None):
    L = T[lang]
    sp_path = Path(summary) if summary else Path(run) / 'results' / 'summary.json'
    S = read(sp_path)
    M = S['meta']
    ow = M['online_window']
    name = (M.get('spec') or {}).get('name') or sp_path.parent.name
    e = html.escape
    k = [0]

    def nxt():
        k[0] += 1
        return k[0]

    nsec = [0]

    def h2(title):
        nsec[0] += 1
        return f'<h2>{L["sec"].format(n=nsec[0])}{title}</h2>'

    def main_ws(sp):
        # the main tables compare with every other frozen window; a run with only the online
        # window (e.g. online W1024 vs frozen W1024) shows that comparison instead
        ws = M['frozen_windows'].get(sp, [])
        return [W for W in ws if W != ow] or [W for W in ws if W == ow]

    def heat(sp, k_):
        Ws = main_ws(sp)
        rows = ''
        for m in M['methods']:
            tds = ''
            for W in Ws:
                c = S['fixed'].get(f'{sp}|{m}|{W}')
                tds += (f'<td class="v {band(c["median"])}">{sgn(c["median"])}%<small>{c["pos"]}/{c["n"]}</small></td>'
                        if c else '<td class="v">—</td>')
            rows += f'<tr><th scope="row">{e(m)}</th>{tds}</tr>'
        head = ''.join(f'<th>W{W}</th>' for W in Ws)
        return (f'<div class="tw"><table class="heat"><caption>{L["main_cap"].format(k=k_, sp=sp, ow=ow)}</caption>'
                f'<thead><tr><th></th>{head}</tr></thead><tbody>{rows}</tbody></table></div>'), Ws

    parts = []
    # verdict draft
    vs = []
    for sp in M['splits']:
        cells = [S['fixed'][f'{sp}|{m}|{W}'] for m in M['methods'] for W in main_ws(sp)
                 if f'{sp}|{m}|{W}' in S['fixed']]
        if cells:
            meds = [c['median'] for c in cells]
            fmt = (lambda v: f'低 {v:.1f}%' if v > 0 else f'高 {-v:.1f}%') if lang == 'zh' else (lambda v: f'{v:.1f}% lower' if v > 0 else f'{-v:.1f}% higher')
            vs.append(L['verdict_cell'].format(sp=sp, n=len(cells), m=len(M['methods']),
                                               w=len(main_ws(sp)),
                                               neg=sum(v < 0 for v in meds), pos=sum(v > 0 for v in meds),
                                               lo=fmt(min(meds)), hi=fmt(max(meds))))
    parts.append(f'<div class="answer"><p class="eyebrow">{L["verdict_h"]}</p>' + ''.join(f'<p>{x}</p>' for x in vs) + '</div>')
    # setup
    spec = M.get('spec') or {}
    bounds = M.get('boundaries') or {}
    data_lines = '<br>'.join(f'{e(dn)}: {L["rows"].format(nt=b[1], nv=b[2], end=b[3])}' for dn, b in bounds.items()) or ', '.join(M['datasets'])
    lr = M.get('lr') or {}
    lr_line = '; '.join(f'{e(kk)}: {v["lr"]:g}' for kk, v in lr.items()) or '—'
    mu = M.get('multipliers') or {}
    xs = {v['multiplier'] for v in mu.values()}
    mu_line = (L['all_x'].format(x=xs.pop()) if len(xs) == 1 else
               '; '.join(f'{e(kk)}: ×{v["multiplier"]:g}' for kk, v in mu.items())) if mu else '—'
    me = int(spec.get('max_epochs', 50))
    budget = L['budget_smoke'].format(n=me) if me < 50 else (L['budget_T'] if spec.get('track') == 'T' else L['budget_C'])
    parts.append(h2(L['setup_h']) + '<dl>'
                 f'<dt>{L["set_data"]}</dt><dd>{data_lines}</dd>'
                 f'<dt>{L["set_models"]}</dt><dd>{e(", ".join(M["models"]))}</dd>'
                 f'<dt>{L["set_h"]}</dt><dd>{", ".join(map(str, M["horizons"]))}</dd>'
                 f'<dt>{L["set_seeds"]}</dt><dd>{", ".join(map(str, M["seeds"]))}</dd>'
                 f'<dt>{L["set_windows"]}</dt><dd>{"; ".join(f"{sp}: " + ", ".join(map(str, ws)) for sp, ws in M["frozen_windows"].items())}</dd>'
                 f'<dt>{L["set_methods"].format(ow=ow)}</dt><dd>{e(", ".join(M["methods"]))}</dd>'
                 f'<dt>{L["set_budget"]}</dt><dd>{budget}</dd>'
                 f'<dt>{L["lr_chosen"]}</dt><dd>{lr_line}</dd><dt>{L["mult_chosen"]}</dt><dd>{mu_line}</dd>'
                 f'<dt>{L["set_rule"]}</dt><dd>{L["rule"]}</dd></dl>')
    # how to read
    ex = M.get('example_pair')
    parts.append(h2(L['read_h']) + f'<p>{L["read_gain"]}</p><p>{L["read_pair"]}</p>'
                 + (f'<p>{L["read_example"].format(gain_s=sgn(ex["gain"]), **ex)}</p>' if ex else '') + f'<p class="note">{L["read_cells"]}</p>')
    # main tables + sentences
    parts.append(h2(L['main_h'].format(ow=ow)))
    sentences = []
    for sp in M['splits']:
        t, Ws = heat(sp, nxt())
        parts.append(t)
        for m in M['methods']:
            for W in Ws:
                c = S['fixed'].get(f'{sp}|{m}|{W}')
                if c:
                    sentences.append((L['sent_pos'] if c['median'] > 0 else L['sent_neg']).format(
                        sp=sp, m=m, W=W, v=abs(c['median']), n=c['n'], pos=c['pos']))
    parts.append(f'<h3>{L["sent_h"]}</h3><ul>' + ''.join(f'<li>{x}</li>' for x in sentences) + '</ul>')
    # matched copies
    rows = ''.join(f'<tr><td>{sp}</td><td>{e(m)}</td><td>{sgn(c["median"])}%</td><td>{c["pos"]}/{c["n"]}</td><td>{c["over10x"]}</td></tr>'
                   for sp in M['splits'] for m in M['methods'] for c in [S.get('matched', {}).get(f'{sp}|{m}')] if c)
    if rows:
        parts.append(h2(L['matched_h']) + f'<p>{L["matched_how"]}</p><div class="tw"><table><caption>{L["matched_cap"].format(k=nxt())}</caption>'
                     f'<thead><tr><th>{L["col_split"]}</th><th>{L["col_method"]}</th><th>{L["col_med"]}</th><th>{L["col_pos"]}</th><th>{L["col_over"]}</th></tr></thead>'
                     f'<tbody>{rows}</tbody></table></div>')
    # same input length
    rows = ''.join(f'<tr><td>{sp}</td><td>{e(m)}</td><td>{sgn(c["median"])}%</td><td>{c["pos"]}/{c["n"]}</td></tr>'
                   for sp in M['splits'] for m in M['methods'] for c in [S['fixed'].get(f'{sp}|{m}|{ow}')] if c)
    if rows and not any(ow in main_ws(sp) for sp in M['splits']):
        parts.append(h2(L['w96_h'].format(ow=ow)) + f'<div class="tw"><table><caption>{L["w96_cap"].format(k=nxt(), ow=ow)}</caption>'
                     f'<thead><tr><th>{L["col_split"]}</th><th>{L["col_method"]}</th><th>{L["col_med"]}</th><th>{L["col_pos_online"]}</th></tr></thead><tbody>{rows}</tbody></table></div>')
    # by dataset (one slice: the first method against W1024, or the longest common window)
    if len(M['datasets']) > 1:
        m0 = 'ER' if 'ER' in M['methods'] else M['methods'][0]
        for sp in M['splits']:
            ws = [W for W in M['frozen_windows'].get(sp, []) if W != ow]
            if not ws:
                continue
            W = 1024 if 1024 in ws else ws[-1]
            rows = ''.join(f'<tr><td>{e(d)}</td><td class="v {band(c["median"])}">{sgn(c["median"])}%<small>{c["pos"]}/{c["n"]}</small></td></tr>'
                           for d in M['datasets'] for c in [S.get('fixed_by_dataset', {}).get(f'{sp}|{m0}|{W}|{d}')] if c)
            parts.append(h2(f'{L["ds_h"]} · {sp}') + f'<div class="tw"><table class="heat"><caption>{L["ds_cap"].format(k=nxt(), m=m0, W=W)}</caption><tbody>{rows}</tbody></table></div>')
    # time blocks
    rows = ''
    for sp in M['splits']:
        for m in M['methods']:
            for W in M['frozen_windows'].get(sp, []):
                c = S['fixed'].get(f'{sp}|{m}|{W}')
                if c and 'first30' in c:
                    rows += f'<tr><td>{sp}</td><td>{e(m)}</td><td>W{W}</td><td>{sgn(c["first30"])}%</td><td>{sgn(c["median"])}%</td><td>{sgn(c["last30"])}%</td></tr>'
    if rows:
        parts.append(h2(L['blk_h']) + f'<p>{L["blk_how"]}</p><div class="tw"><table><caption>{L["blk_cap"].format(k=nxt())}</caption>'
                     f'<thead><tr><th>{L["col_split"]}</th><th>{L["col_method"]}</th><th></th><th>{L["col_first"]}</th><th>{L["col_whole"]}</th><th>{L["col_last"]}</th></tr></thead><tbody>{rows}</tbody></table></div>')
    # cost
    rows = ''.join(f'<tr><td>{kk.split("|")[0]}</td><td>{e(kk.split("|")[1])}</td><td>{v["online_ms"]:.2f}</td><td>{v["frozen_ms"]:.2f}</td><td>{v["ratio_median"]:.1f}×</td></tr>'
                   for kk, v in sorted(S.get('cost', {}).items()))
    if rows:
        parts.append(h2(L['cost_h']) + f'<div class="tw"><table><caption>{L["cost_cap"].format(k=nxt(), ow=ow)}</caption>'
                     f'<thead><tr><th>{L["col_split"]}</th><th>{L["col_method"]}</th><th>{L["col_online_ms"]}</th><th>{L["col_frozen_ms"]}</th><th>{L["col_ratio"]}</th></tr></thead><tbody>{rows}</tbody></table></div>')
    tr = S.get('train_seconds') or {}
    if tr:
        rows = ''.join(f'<tr><td>{kk.split("|")[0]}</td><td>{kk.split("|")[1]}</td><td>{v:.1f}</td></tr>'
                       for kk, v in sorted(tr.items(), key=lambda kv: (kv[0].split('|')[0], int(kv[0].split('|W')[1]))))
        parts.append(f'<div class="tw"><table><caption>{L["train_cap"].format(k=nxt())}</caption><tbody>{rows}</tbody></table></div>')
    # claims (to be written)
    parts.append(h2(L['claims_h']) + f'<p class="note">{L["claims_hint"]}</p>'
                 f'<h3>{L["claims_yes"]}</h3><div class="write">…</div><h3>{L["claims_no"]}</h3><div class="write">…</div>'
                 f'<h2>{L["notes_h"]}</h2><p class="note">{L["notes"]}</p>')
    doc = (f'<!doctype html><html lang="{"zh-CN" if lang == "zh" else "en"}"><head><meta charset="utf-8">'
           f'<meta name="viewport" content="width=device-width,initial-scale=1"><title>{e(L["title"])} · {e(name)}</title>'
           f'<style>{CSS}</style></head><body><main><p class="eyebrow">OLTS-Bench · {L["run"]} {e(name)} · {L["draft"]} · {date.today().isoformat()}</p>'
           f'<h1>{L["title"]}</h1>' + ''.join(parts) + '</main></body></html>')
    out = Path(out) if out else sp_path.parent / f'report_{lang}.html'
    out.write_text(doc, encoding='utf-8')
    return out


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('run', nargs='?')
    ap.add_argument('--summary')
    ap.add_argument('--lang', choices=('zh', 'en'), default='zh')
    ap.add_argument('--out')
    a = ap.parse_args()
    if not a.run and not a.summary:
        ap.error('give a run folder or --summary')
    print(build(a.run, a.lang, a.summary, a.out))
