# -*- coding: utf-8 -*-
"""
make_recommended_report.py
- 입력: 권장 호라이즌 및 트레이드 레벨 CSV (질문에서 제시한 스키마)
- 출력: HTML 리포트(요약 테이블 + 종목별 카드)

필수 컬럼(예시):
종목명,종목코드,권장호라이즌,요약점수(h1),요약점수(h2),요약점수(h3),
h2_MAPE(%),h2_DirAcc(%),h2_Bias,h2_MAE,h2_count,
last_close,매수가(entry),익절가(tp),손절가(sl),RR,
h1_MAPE(%),h1_DirAcc(%),h1_Bias,h1_MAE,h1_count, ... (h3_* 있을 수 있음)
"""

import os
import argparse
from datetime import datetime
import pandas as pd
import numpy as np
from pathlib import Path
from DSConfig_3 import DSConfig, config
from dataset_functions import last_trading_day
from make_trade_price import make_trade_price

cfg = DSConfig
cfg.end_date = last_trading_day()


STYLE = """
<style>
:root {
  --bg:#0b1020; --panel:#121936; --text:#e6eaf3; --muted:#98a2b3; --border:#253055; --accent:#6ea8fe;
}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--text);font-family:-apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif}
.header{background:linear-gradient(135deg,#1a2248,#0f1533);border-bottom:1px solid var(--border)}
.container{max-width:1100px;margin:0 auto;padding:24px 16px}
.title{margin:0;font-weight:800;font-size:28px}
.subtitle{color:var(--muted);font-size:13px;margin-top:6px}
.card{background:var(--panel);border:1px solid var(--border);border-radius:16px;padding:16px;margin:16px 0}
.card h2{margin:0 0 8px;font-size:18px}
.table{width:100%;border-collapse:collapse}
.table thead th{position:sticky;top:0;background:#172047;border-bottom:1px solid var(--border);padding:10px;text-align:left;color:#c6d0e5;font-size:12px}
.table tbody td{border-bottom:1px solid var(--border);padding:10px;font-size:13px}
.row{border:1px solid var(--border);border-radius:16px;padding:12px;margin:10px 0;background:#0f1633}
.row h3{margin:0 0 8px;font-size:16px}
.row .meta{color:var(--muted);font-size:12px;margin-bottom:6px}
.metrics{display:grid;grid-template-columns:repeat(5,1fr);gap:8px}
.levels{display:grid;grid-template-columns:repeat(4,1fr);gap:8px;margin-top:8px}
.badge{display:inline-block;padding:2px 8px;border-radius:999px;border:1px solid var(--border);font-size:12px;margin-left:6px}
.badge.h1{background:rgba(110,168,254,.12);border-color:rgba(110,168,254,.35);color:#cfe2ff}
.badge.h2{background:rgba(46,204,113,.12);border-color:rgba(46,204,113,.35);color:#b6f0c8}
.badge.h3{background:rgba(241,196,15,.12);border-color:rgba(241,196,15,.35);color:#ffeaa7}
.note{color:var(--muted);font-size:12px}
footer{color:var(--muted);font-size:12px;margin-top:16px}
</style>
"""

def _to_float_cols(df: pd.DataFrame) -> pd.DataFrame:
    for c in df.columns:
        if c not in ("종목명", "종목코드", "권장호라이즌"):
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

def _kpis_for_row(row: pd.Series):
    """권장 호라이즌 prefix(h1/h2/h3)에 맞는 KPI 추출"""
    h = row.get("권장호라이즌", "h2")
    prefix = h if h in ("h1","h2","h3") else "h2"
    mape = row.get(f"{prefix}_MAPE(%)", np.nan)
    da   = row.get(f"{prefix}_DirAcc(%)", np.nan)
    bias = row.get(f"{prefix}_Bias", np.nan)
    mae  = row.get(f"{prefix}_MAE", np.nan)
    cnt  = row.get(f"{prefix}_count", np.nan)
    return prefix, mape, da, bias, mae, cnt

def _summary_table_html(df: pd.DataFrame) -> str:
    base_cols = ["종목명","종목코드","권장호라이즌","last_close","매수가(entry)","익절가(tp)","손절가(sl)","RR"]
    extras = ["요약점수(h1)","요약점수(h2)","요약점수(h3)","h1_MAPE(%)","h2_MAPE(%)","h3_MAPE(%)","h1_DirAcc(%)","h2_DirAcc(%)","h3_DirAcc(%)"]
    cols = base_cols + [c for c in extras if c in df.columns]
    return df[cols].to_html(index=False, classes="table", border=0,
                            float_format=lambda x: f"{x:,.2f}" if isinstance(x, (float, np.floating)) else x)

def _cards_html(df: pd.DataFrame) -> str:
    cards = []
    for _, row in df.iterrows():
        prefix, mape, da, bias, mae, cnt = _kpis_for_row(row)
        badge = f'<span class="badge {prefix}">{prefix.upper()}</span>'
        def _fmt(x, n=2, comma=True):
            if pd.isna(x): return ""
            return f"{x:,.{n}f}" if comma else f"{x:.{n}f}"

        card = f"""
        <div class="row">
          <h3>{row['종목명']} ({row['종목코드']}) {badge}</h3>
          <div class="meta">요약점수: h1={_fmt(row.get('요약점수(h1)'))}, h2={_fmt(row.get('요약점수(h2)'))}, h3={_fmt(row.get('요약점수(h3)'))}</div>
          <div class="metrics">
            <div class="kv"><div class="label">{prefix} MAPE(%)</div><div class="value">{_fmt(mape)}</div></div>
            <div class="kv"><div class="label">{prefix} DirAcc(%)</div><div class="value">{_fmt(da)}</div></div>
            <div class="kv"><div class="label">{prefix} Bias</div><div class="value">{_fmt(bias)}</div></div>
            <div class="kv"><div class="label">{prefix} MAE</div><div class="value">{_fmt(mae)}</div></div>
            <div class="kv"><div class="label">{prefix} count</div><div class="value">{'' if pd.isna(cnt) else int(cnt)}</div></div>
          </div>
          <div class="levels">
            <div class="kv"><div class="label">last_close</div><div class="value">{_fmt(row['last_close'])}</div></div>
            <div class="kv"><div class="label">매수가(entry)</div><div class="value">{_fmt(row['매수가(entry)'])}</div></div>
            <div class="kv"><div class="label">익절가(tp)</div><div class="value">{_fmt(row['익절가(tp)'])}</div></div>
            <div class="kv"><div class="label">손절가(sl)</div><div class="value">{_fmt(row['손절가(sl)'])}</div></div>
          </div>
        </div>
        """
        cards.append(card)
    return "\n".join(cards)

def build_html(df: pd.DataFrame) -> str:
    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M")
    table_html = _summary_table_html(df)
    cards_html = _cards_html(df)

    html = f"""<!doctype html>
<html lang="ko">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>권장 호라이즌 & 트레이드 레벨 리포트</title>
{STYLE}
</head>
<body>
  <div class="header">
    <div class="container">
      <h1 class="title">권장 호라이즌 & 트레이드 레벨 리포트</h1>
      <div class="subtitle">생성시각: {generated_at}</div>
    </div>
  </div>

  <div class="container">
    <div class="card">
      <h2>요약 테이블</h2>
      {table_html}
      <div class="note">RR은 (익절-매수)/(매수-손절)로 계산된 기대 보상/위험 비율입니다.</div>
    </div>

    <div class="card">
      <h2>종목별 카드</h2>
      {cards_html}
    </div>

    <footer>자동 생성 리포트 • 전략은 참고용이며 투자 책임은 투자자 본인에게 있습니다.</footer>
  </div>
</body>
</html>"""
    return html

def report_trade_price(cfg):
    
    make_trade_price(cfg)
    
    input_dir = Path(cfg.report_dir) / f"Report_{cfg.end_date}"
    input_file = input_dir / f"Trading_price_{cfg.end_date}.csv"

    output_dir = Path(cfg.report_dir)
    output_file = output_dir / f"Report_{cfg.end_date}" / f"Trading_price_{cfg.end_date}.html"

    df = pd.read_csv(input_file, dtype={"종목코드": str})
    # 코드 6자리 보정
    if "종목코드" in df.columns:
        df["종목코드"] = df["종목코드"].astype(str).str.zfill(6)

    # 수치 컬럼 변환
    df = _to_float_cols(df)

    html = build_html(df)
    
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"[OK] HTML saved -> {output_file}")

def main():
    print("→ report_trade_price.py 시작")
    report_trade_price()  
    
if __name__ == "__main__":
    main()
