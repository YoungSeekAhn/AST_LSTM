# -*- coding: utf-8 -*-
"""
report_builder.py
- 멀티타깃(h1/h2/h3 × high/low/close) 예측결과 CSV를 평가하여
  1) 종목별 1행 요약 CSV (wide)
  2) 상세 long CSV
  3) 종목별 HTML 리포트(표 + 시각화)
를 생성합니다.

의존: pandas, numpy, matplotlib
"""

import os, re, glob, argparse
from typing import List, Dict, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---- 고정 설정 ----
HORIZONS = [1, 2, 3]
TARGETS  = ["high", "low", "close"]
EPS = 1e-12

# ---------------------------
# 유틸: 종목명/코드 추출
# ---------------------------
def parse_symbol_from_filename(path: str) -> Tuple[str, str]:
    """
    파일명에서 (종목명, 종목코드)를 추출.
    허용 예:
      - '005930_삼성전자.csv'
      - '삼성전자_005930.csv'
      - '005930.csv' or '삼성전자.csv'
    우선순위: 숫자 6자리를 종목코드로 간주.
    """
    base = os.path.splitext(os.path.basename(path))[0]
    parts = re.split(r'[_\- ]+', base)

    code = None
    name = None
    for p in parts:
        if re.fullmatch(r'\d{6}', p):
            code = p
        else:
            name = p if name is None else name

    if code is None and name is None:
        name = base
    return (name or ""), (code or "")

# ---------------------------
# 메트릭/방향 적중률
# ---------------------------
def _safe_num(s):
    return pd.to_numeric(s, errors="coerce")

def _metrics(y_true: pd.Series, y_pred: pd.Series) -> Dict[str, float]:
    y_true = _safe_num(y_true)
    y_pred = _safe_num(y_pred)
    mask = y_true.notna() & y_pred.notna()
    if mask.sum() == 0:
        return {"count": 0, "RMSE": np.nan, "MAE": np.nan, "MAPE(%)": np.nan, "Bias": np.nan}
    e = (y_pred[mask] - y_true[mask]).values
    yt = y_true[mask].values
    rmse = float(np.sqrt(np.mean(e**2)))
    mae  = float(np.mean(np.abs(e)))
    mape = float(np.mean(np.abs(e / np.clip(yt, EPS, None))) * 100.0)
    bias = float(np.mean(e))
    return {"count": int(mask.sum()), "RMSE": rmse, "MAE": mae, "MAPE(%)": mape, "Bias": bias}

def _direction_accuracy_close(df: pd.DataFrame, h: int) -> Dict[str, float]:
    """
    방향 적중률(%) - close 기준
    sign(pred_h{h}_close - prev_true_close) == sign(true_close - prev_true_close)
    """
    col_pred = f"pred_h{h}_close"
    if "true_close" not in df.columns or col_pred not in df.columns:
        return {"DirAcc_close(%)": np.nan, "Dir_count": 0}

    df = df.sort_values("date")
    y_true = _safe_num(df["true_close"])
    y_pred = _safe_num(df[col_pred])
    prev_true = y_true.shift(1)

    mask = y_true.notna() & y_pred.notna() & prev_true.notna()
    if mask.sum() == 0:
        return {"DirAcc_close(%)": np.nan, "Dir_count": 0}

    true_dir = np.sign(y_true[mask] - prev_true[mask]).values
    pred_dir = np.sign(y_pred[mask] - prev_true[mask]).values
    acc = float((true_dir == pred_dir).mean() * 100.0)
    return {"DirAcc_close(%)": acc, "Dir_count": int(mask.sum())}

# ---------------------------
# 단일 종목 평가 (wide/long)
# ---------------------------
def evaluate_symbol_to_wide_and_long(df: pd.DataFrame,
                                     stock_name: str,
                                     stock_code: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    # 날짜 정리
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.sort_values("date")

    # 숫자 변환
    for t in ["high", "low", "close"]:
        col = f"true_{t}"
        if col in df.columns:
            df[col] = _safe_num(df[col])
    for h in HORIZONS:
        for t in TARGETS:
            col = f"pred_h{h}_{t}"
            if col in df.columns:
                df[col] = _safe_num(df[col])

    # wide 1행
    wide_row = {"종목명": stock_name, "종목코드": stock_code}
    # long 행들
    long_rows: List[Dict] = []

    for h in HORIZONS:
        dir_res = _direction_accuracy_close(df, h)

        for t in TARGETS:
            true_col = f"true_{t}"
            pred_col = f"pred_h{h}_{t}"
            if true_col in df.columns and pred_col in df.columns:
                m = _metrics(df[true_col], df[pred_col])
            else:
                m = {"count": 0, "RMSE": np.nan, "MAE": np.nan, "MAPE(%)": np.nan, "Bias": np.nan}

            # wide 채우기
            wide_row[f"h{h}_{t}_RMSE"]     = m["RMSE"]
            wide_row[f"h{h}_{t}_MAE"]      = m["MAE"]
            wide_row[f"h{h}_{t}_MAPE(%)"]  = m["MAPE(%)"]
            wide_row[f"h{h}_{t}_Bias"]     = m["Bias"]
            wide_row[f"h{h}_{t}_count"]    = m["count"]

            # long 행 추가
            long_rows.append({
                "종목명": stock_name,
                "종목코드": stock_code,
                "horizon": f"h{h}",
                "target": t,
                **m,
                "DirAcc_close(%)": dir_res["DirAcc_close(%)"] if t == "close" else np.nan,
                "Dir_count": dir_res["Dir_count"] if t == "close" else 0
            })

        # 방향성(종가) wide 필드
        wide_row[f"h{h}_close_DirAcc(%)"] = dir_res["DirAcc_close(%)"]
        wide_row[f"h{h}_close_Dir_count"] = dir_res["Dir_count"]

    wide_df = pd.DataFrame([wide_row])
    long_df = pd.DataFrame(long_rows)
    return wide_df, long_df

# ---------------------------
# 리포트(HTML) 생성
# ---------------------------
def plot_close_series(df: pd.DataFrame, out_png: str, h: int):
    """
    실제 close vs pred_h{h}_close 라인 차트 저장
    (요구사항: matplotlib, 단일 plot, 색상 지정 금지)
    """
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.sort_values("date")
    plt.figure(figsize=(9,4.5))
    plt.plot(df["date"], df["true_close"], label="true_close")
    col = f"pred_h{h}_close"
    if col in df.columns:
        plt.plot(df["date"], df[col], label=col)
    plt.title(f"Close vs {col}")
    plt.xlabel("date"); plt.ylabel("price")
    plt.legend(); plt.tight_layout()
    plt.grid(True)
    plt.savefig(out_png, dpi=140)
    plt.close()

def plot_error_hist(df: pd.DataFrame, out_png: str, h: int, target: str):
    """
    오차 히스토그램 저장: pred_h{h}_{target} - true_{target}
    """
    true_col = f"true_{target}"
    pred_col = f"pred_h{h}_{target}"
    if true_col not in df.columns or pred_col not in df.columns:
        return
    e = (_safe_num(df[pred_col]) - _safe_num(df[true_col])).dropna()
    if len(e) == 0:
        return

    plt.figure(figsize=(6,4))
    plt.hist(e.values, bins=20)
    plt.title(f"Error Histogram: {pred_col} - {true_col}")
    plt.xlabel("error"); plt.ylabel("freq")
    plt.tight_layout()
    plt.grid(True)
    plt.savefig(out_png, dpi=140)
    plt.close()

def write_symbol_report_html(df: pd.DataFrame,
                             wide_df: pd.DataFrame,
                             long_df: pd.DataFrame,
                             stock_name: str,
                             stock_code: str,
                             out_dir: str):
    os.makedirs(out_dir, exist_ok=True)

    # 그래프들 생성
    close_imgs = []
    for h in HORIZONS:
        img = os.path.join(out_dir, f"chart_close_h{h}.png")
        plot_close_series(df, img, h)
        close_imgs.append(os.path.basename(img))

    err_imgs = []
    for h in HORIZONS:
        for t in TARGETS:
            img = os.path.join(out_dir, f"errors_hist_{t}_h{h}.png")
            plot_error_hist(df, img, h, t)
            if os.path.exists(img):
                err_imgs.append(os.path.basename(img))

    # 표: wide 상단, long 하단
    wide_html = wide_df.to_html(index=False, float_format=lambda x: f"{x:.6g}" if isinstance(x, float) else x)
    long_html = long_df.to_html(index=False, float_format=lambda x: f"{x:.6g}" if isinstance(x, float) else x)

    # HTML
    title = f"예측 리포트 - {stock_name} ({stock_code})"
    html = f"""<!doctype html>
<html lang="ko">
<head>
<meta charset="utf-8">
<title>{title}</title>
<style>
body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Noto Sans KR", Arial, sans-serif; margin: 24px; }}
h1 {{ font-size: 22px; margin-bottom: 8px; }}
h2 {{ font-size: 18px; margin-top: 24px; }}
.card {{ border: 1px solid #ddd; border-radius: 8px; padding: 16px; margin-bottom: 16px; }}
img {{ max-width: 100%; height: auto; }}
table {{ border-collapse: collapse; width: 100%; font-size: 13px; }}
th, td {{ border: 1px solid #ddd; padding: 6px 8px; text-align: right; }}
th {{ background: #f7f7f7; }}
td:first-child, th:first-child {{ text-align: left; }}
.small {{ color: #666; font-size: 12px; }}
</style>
</head>
<body>
  <h1>{title}</h1>
  <div class="small">자동 생성: report_builder.py</div>

  <div class="card">
    <h2>요약(종목당 1행, wide)</h2>
    {wide_html}
  </div>

  <div class="card">
    <h2>시계열: 실제 vs 예측 (Close)</h2>
    {"".join([f'<div><img src="{img}" alt="{img}"/></div>' for img in close_imgs])}
  </div>

  <div class="card">
    <h2>오차 분포(히스토그램)</h2>
    {"".join([f'<div><img src="{img}" alt="{img}"/></div>' for img in err_imgs])}
  </div>

  <div class="card">
    <h2>상세(long)</h2>
    {long_html}
  </div>
</body>
</html>
"""
    out_html = os.path.join(out_dir, f"report_{stock_name}_{stock_code or 'NA'}.html")
    with open(out_html, "w", encoding="utf-8") as f:
        f.write(html)
    return out_html

# ---------------------------
# 폴더 일괄 처리 + 인덱스
# ---------------------------
def run_build(input_dir: str, out_dir: str,
              pattern: str = "*.csv",
              wide_csv: str = "symbol_metrics_wide.csv",
              long_csv: str = "symbol_metrics_long.csv"):
    os.makedirs(out_dir, exist_ok=True)

    wide_rows = []
    long_rows = []
    links = []

    paths = sorted(glob.glob(os.path.join(input_dir, pattern)))
    if not paths:
        raise FileNotFoundError(f"No CSV files found in {input_dir}/{pattern}")

    for path in paths:
        df = pd.read_csv(path)

        # 파일 내 Name/Code 우선 사용
        name_col = next((c for c in df.columns if c.lower() in ["name","종목명"]), None)
        code_col = next((c for c in df.columns if c.lower() in ["code","종목코드"]), None)
        stock_name = df[name_col].iloc[0] if name_col else ""
        stock_code = str(df[code_col].iloc[0]) if code_col else ""

        if not stock_name or not stock_code:
            n, c = parse_symbol_from_filename(path)
            stock_name = stock_name or n
            stock_code = stock_code or c

        # 평가
        wide_df, long_df = evaluate_symbol_to_wide_and_long(df, stock_name, stock_code)
        wide_rows.append(wide_df)
        long_rows.append(long_df)

        # 리포트
        sym_dir = os.path.join(out_dir, stock_code or os.path.splitext(os.path.basename(path))[0])
        os.makedirs(sym_dir, exist_ok=True)
        html_path = write_symbol_report_html(df, wide_df, long_df, stock_name, stock_code, sym_dir)
        rel_path = os.path.relpath(html_path, out_dir)
        links.append((stock_name, stock_code, rel_path))

    # CSV 저장
    wide_all = pd.concat(wide_rows, ignore_index=True)
    long_all = pd.concat(long_rows, ignore_index=True)

    # 컬럼 정렬: 종목명/종목코드 먼저
    front = ["종목명", "종목코드"]
    others = [c for c in wide_all.columns if c not in front]
    wide_all = wide_all[front + others]

    wide_path = os.path.join(out_dir, wide_csv)
    long_path = os.path.join(out_dir, long_csv)
    wide_all.to_csv(wide_path, index=False, encoding="utf-8-sig")
    long_all.to_csv(long_path, index=False, encoding="utf-8-sig")

    # 인덱스 HTML
    idx_html = """<!doctype html>
<html lang="ko">
<head>
<meta charset="utf-8">
<title>예측 리포트 인덱스</title>
<style>
body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Noto Sans KR", Arial, sans-serif; margin: 24px; }
h1 { font-size: 22px; margin-bottom: 12px; }
table { border-collapse: collapse; width: 100%; font-size: 14px; }
th, td { border: 1px solid #ddd; padding: 6px 8px; text-align: left; }
th { background: #f7f7f7; }
.small { color: #666; font-size: 12px; }
</style>
</head>
<body>
  <h1>예측 리포트 인덱스</h1>
  <div class="small">자동 생성: report_builder.py</div>
  <table>
    <thead><tr><th>종목명</th><th>종목코드</th><th>리포트</th></tr></thead>
    <tbody>
      {}
    </tbody>
  </table>
  <p class="small">요약 CSV: <code>{}</code> / 상세 CSV: <code>{}</code></p>
</body>
</html>
"""
    rows_html = "\n".join(
        [f'<tr><td>{n}</td><td>{c}</td><td><a href="{p}">{os.path.basename(p)}</a></td></tr>'
         for (n, c, p) in links]
    )
    with open(os.path.join(out_dir, "index.html"), "w", encoding="utf-8") as f:
        f.write(idx_html.format(rows_html, wide_csv, long_csv))

    print(f"[DONE] wide: {wide_path}")
    print(f"[DONE] long: {long_path}")
    print(f"[DONE] index: {os.path.join(out_dir,'index.html')}")

# ---------------------------
# CLI
# ---------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dir", required=True, help="예측 CSV 폴더")
    ap.add_argument("--out_dir", required=True, help="리포트 출력 폴더")
    ap.add_argument("--pattern", default="*.csv", help="입력 파일 패턴 (기본: *.csv)")
    args = ap.parse_args()

    run_build(args.input_dir, args.out_dir, pattern=args.pattern)

if __name__ == "__main__":
    main()
