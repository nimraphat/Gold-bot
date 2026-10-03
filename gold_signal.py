"""
Gold (XAUUSD) M15 Signal Scanner  -  modular, free, runs locally
-----------------------------------------------------------------
ติดตั้ง:   pip install pandas numpy MetaTrader5 yfinance
           (MetaTrader5 ใช้ได้บน Windows เท่านั้น; Mac/Linux จะใช้ yfinance อัตโนมัติ)

วิธีใช้:
  python gold_signal.py             -> สแกนสัญญาณสดทุกครั้งที่แท่ง M15 ปิด
  python gold_signal.py backtest    -> ทดสอบย้อนหลังด้วยท่าที่เปิดอยู่

เพิ่ม/ลด "ท่า": แก้ ENABLED ด้านล่าง (True/False) หรือเขียนฟังก์ชันใหม่แล้วลงทะเบียนใน STRATEGIES
"""
import sys
import time
import json
import urllib.request
import numpy as np
import pandas as pd

# ======================= ตั้งค่า =======================
CONFIG = dict(
    SYMBOL="XAUUSDc",          # ชื่อสัญลักษณ์ของโบรกเกอร์ (cent มักลงท้าย c / m / .c) ดูใน Market Watch
    BARS=3000,                 # จำนวนแท่งที่ดึง
    SERVER_UTC_OFFSET=3,       # เวลาเซิร์ฟเวอร์โบรกเกอร์ - UTC (Exness/ส่วนใหญ่ = +2 หรือ +3 ตามฤดูกาล)
    MIN_VOTES=2,               # ต้องมีกี่ท่าเห็นตรงกันถึงจะส่งสัญญาณ (ยิ่งสูงยิ่งน้อยแต่คัดกว่า)
    ATR_SL_MULT=1.5,           # SL = ATR x ค่านี้
    RR_TP1=1.0,                # TP1 (ปิดครึ่ง + เลื่อน SL มาทุน)
    RR_TP2=2.0,                # TP2 (ปิดที่เหลือ)
    RISK_PCT=2.0,              # % เสี่ยงต่อไม้ (พร้อมเสี่ยง = 2-3, ไม่แนะนำเกิน 5)
    MANUAL_BALANCE=1000.0,     # ใช้เมื่อไม่มี MT5 (หน่วยเดียวกับบัญชี)
    SPREAD_PRICE=0.30,         # สเปรดโดยประมาณ หน่วยราคาทอง ใช้ใน backtest
    TELEGRAM_TOKEN="",         # (ไม่บังคับ) ใส่เพื่อให้แจ้งเตือนเข้า Telegram
    TELEGRAM_CHAT_ID="",
)

# เปิด/ปิดท่าตรงนี้
ENABLED = {
    "trend_pullback": True,    # เทรดตามเทรนด์ ย่อซื้อ/เด้งขาย
    "bb_rsi_reversion": True,  # สวนเทรนด์ตอนไซด์เวย์
    "donchian_breakout": True, # เบรกเอาต์ 20 แท่ง
    "session_breakout": True,  # เบรกกรอบเอเชียตอนลอนดอนเปิด
}

try:
    import MetaTrader5 as mt5
    HAS_MT5 = True
except Exception:
    mt5 = None
    HAS_MT5 = False


# ======================= ดึงข้อมูล =======================
def get_data():
    """คืน DataFrame index = เวลา UTC, คอลัมน์ open high low close"""
    if HAS_MT5 and mt5.initialize():
        rates = mt5.copy_rates_from_pos(CONFIG["SYMBOL"], mt5.TIMEFRAME_M15, 0, CONFIG["BARS"])
        if rates is not None and len(rates) > 250:
            df = pd.DataFrame(rates)
            df["time"] = pd.to_datetime(df["time"], unit="s") - pd.Timedelta(hours=CONFIG["SERVER_UTC_OFFSET"])
            return df.set_index("time")[["open", "high", "low", "close"]]
        print("MT5 ดึงข้อมูลไม่ได้ (เช็กชื่อ SYMBOL) -> ใช้ Yahoo แทน")
    import yfinance as yf
    d = yf.download("GC=F", interval="15m", period="30d", progress=False, auto_adjust=False)
    d.columns = [c[0].lower() if isinstance(c, tuple) else c.lower() for c in d.columns]
    d.index = pd.to_datetime(d.index).tz_convert("UTC").tz_localize(None)
    return d[["open", "high", "low", "close"]].dropna()


# ======================= อินดิเคเตอร์ =======================
def add_indicators(df):
    df = df.copy()
    c, h, l = df["close"], df["high"], df["low"]
    df["ema20"] = c.ewm(span=20, adjust=False).mean()
    df["ema50"] = c.ewm(span=50, adjust=False).mean()
    df["ema200"] = c.ewm(span=200, adjust=False).mean()

    delta = c.diff()
    up = delta.clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean()
    dn = (-delta.clip(upper=0)).ewm(alpha=1 / 14, adjust=False).mean()
    df["rsi"] = 100 - 100 / (1 + up / dn.replace(0, np.nan))

    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    df["atr"] = tr.ewm(alpha=1 / 14, adjust=False).mean()

    pdm = (h.diff()).where((h.diff() > -l.diff()) & (h.diff() > 0), 0.0)
    ndm = (-l.diff()).where((-l.diff() > h.diff()) & (-l.diff() > 0), 0.0)
    pdi = 100 * pdm.ewm(alpha=1 / 14, adjust=False).mean() / df["atr"]
    ndi = 100 * ndm.ewm(alpha=1 / 14, adjust=False).mean() / df["atr"]
    dx = 100 * (pdi - ndi).abs() / (pdi + ndi).replace(0, np.nan)
    df["adx"] = dx.ewm(alpha=1 / 14, adjust=False).mean()

    mid, sd = c.rolling(20).mean(), c.rolling(20).std()
    df["bb_up"], df["bb_lo"] = mid + 2 * sd, mid - 2 * sd
    df["dc_hi"] = h.rolling(20).max().shift(1)
    df["dc_lo"] = l.rolling(20).min().shift(1)
    return df


# ======================= ท่าเทรด (คืน +1 ซื้อ / -1 ขาย / 0 ไม่มี) =======================
def _sig(buy, sell):
    return pd.Series(np.where(buy, 1, np.where(sell, -1, 0)), index=buy.index)


def s_trend_pullback(d):
    up = (d.ema50 > d.ema200) & (d.close > d.ema200)
    dn = (d.ema50 < d.ema200) & (d.close < d.ema200)
    buy = up & (d.low <= d.ema20) & (d.close > d.ema20) & d.rsi.between(40, 65) & (d.adx > 20)
    sell = dn & (d.high >= d.ema20) & (d.close < d.ema20) & d.rsi.between(35, 60) & (d.adx > 20)
    return _sig(buy, sell)


def s_bb_rsi(d):
    rng = d.adx < 25
    buy = rng & (d.low < d.bb_lo) & (d.close > d.bb_lo) & (d.rsi < 35)
    sell = rng & (d.high > d.bb_up) & (d.close < d.bb_up) & (d.rsi > 65)
    return _sig(buy, sell)


def s_donchian(d):
    buy = (d.close > d.dc_hi) & (d.adx > 20)
    sell = (d.close < d.dc_lo) & (d.adx > 20)
    return _sig(buy, sell)


def s_session(d):
    hour = d.index.hour
    day = d.index.date
    asia = hour < 7  # 00:00-07:00 UTC = 07:00-14:00 เวลาไทย
    a_hi = d.high.where(asia).groupby(day).transform("max")
    a_lo = d.low.where(asia).groupby(day).transform("min")
    win = (hour >= 7) & (hour < 11)  # ลอนดอนเปิด
    buy = win & (d.close > a_hi) & (d.close.shift() <= a_hi)
    sell = win & (d.close < a_lo) & (d.close.shift() >= a_lo)
    return _sig(buy, sell)


STRATEGIES = {
    "trend_pullback": s_trend_pullback,
    "bb_rsi_reversion": s_bb_rsi,
    "donchian_breakout": s_donchian,
    "session_breakout": s_session,
}


def compute_signals(d):
    votes = pd.DataFrame({n: f(d) for n, f in STRATEGIES.items() if ENABLED.get(n)})
    buy_v, sell_v = (votes == 1).sum(axis=1), (votes == -1).sum(axis=1)
    final = np.where((buy_v >= CONFIG["MIN_VOTES"]) & (buy_v > sell_v), 1,
                     np.where((sell_v >= CONFIG["MIN_VOTES"]) & (sell_v > buy_v), -1, 0))
    return pd.Series(final, index=d.index), votes


# ======================= แผนเทรด / ขนาดไม้ =======================
def calc_lot(sl_dist):
    risk_pct = CONFIG["RISK_PCT"] / 100
    if HAS_MT5 and mt5.terminal_info() is not None:
        info, acc = mt5.symbol_info(CONFIG["SYMBOL"]), mt5.account_info()
        if info and acc and info.trade_tick_size > 0:
            loss_per_lot = sl_dist / info.trade_tick_size * info.trade_tick_value
            lot = acc.balance * risk_pct / loss_per_lot
            lot = max(info.volume_min, min(info.volume_max, lot))
            step = info.volume_step
            return round(np.floor(lot / step) * step, 2), acc.balance
    return None, CONFIG["MANUAL_BALANCE"]


def build_plan(d, direction):
    last = d.iloc[-1]
    entry, atr = last.close, last.atr
    sl_dist = atr * CONFIG["ATR_SL_MULT"]
    sl = entry - direction * sl_dist
    tp1 = entry + direction * sl_dist * CONFIG["RR_TP1"]
    tp2 = entry + direction * sl_dist * CONFIG["RR_TP2"]
    lot, bal = calc_lot(sl_dist)
    return dict(side="BUY" if direction == 1 else "SELL", entry=entry, sl=sl, tp1=tp1, tp2=tp2,
                atr=atr, lot=lot, balance=bal, add_level=entry + direction * atr)


def format_plan(p, t, names):
    lot_txt = f"{p['lot']}" if p["lot"] else f"เสี่ยง {CONFIG['RISK_PCT']}% ของพอร์ต (คำนวณ lot เองจาก SL)"
    return (
        f"\n{'=' * 46}\n สัญญาณ {p['side']} XAUUSD  |  แท่ง {t:%Y-%m-%d %H:%M} UTC\n{'=' * 46}\n"
        f" ท่าที่เห็นตรงกัน : {', '.join(names)}\n"
        f" Entry          : {p['entry']:.2f}\n"
        f" Stop Loss      : {p['sl']:.2f}\n"
        f" TP1 (ปิดครึ่ง)   : {p['tp1']:.2f}  -> แล้วเลื่อน SL มาที่ทุน\n"
        f" TP2 (ปิดที่เหลือ) : {p['tp2']:.2f}\n"
        f" Lot แนะนำ       : {lot_txt}\n"
        f" ถ้าจะเพิ่มไม้     : ราคาไปถึง {p['add_level']:.2f} และยังไม่โดน SL (ไม้ที่ 2 ใช้ SL เดียวกับไม้แรก, lot ไม่เกินครึ่งของไม้แรก)\n"
    )


def notify(msg):
    print(msg)
    if CONFIG["TELEGRAM_TOKEN"] and CONFIG["TELEGRAM_CHAT_ID"]:
        try:
            url = f"https://api.telegram.org/bot{CONFIG['TELEGRAM_TOKEN']}/sendMessage"
            data = json.dumps({"chat_id": CONFIG["TELEGRAM_CHAT_ID"], "text": msg}).encode()
            urllib.request.urlopen(urllib.request.Request(url, data, {"Content-Type": "application/json"}), timeout=10)
        except Exception as e:
            print("ส่ง Telegram ไม่สำเร็จ:", e)


# ======================= โหมดสด =======================
def live():
    print("เริ่มสแกน M15 ... (Ctrl+C เพื่อหยุด) ท่าที่เปิด:", [k for k, v in ENABLED.items() if v])
    last_bar = None
    while True:
        try:
            d = add_indicators(get_data()).iloc[:-1]  # ตัดแท่งที่ยังไม่ปิด
            t = d.index[-1]
            if t != last_bar:
                last_bar = t
                sig, votes = compute_signals(d)
                s = sig.iloc[-1]
                if s != 0:
                    names = [n for n in votes.columns if votes[n].iloc[-1] == s]
                    notify(format_plan(build_plan(d, s), t, names))
                else:
                    print(f"[{t:%H:%M} UTC] ยังไม่มีสัญญาณ")
        except Exception as e:
            print("error:", e)
        time.sleep(20)


# ======================= Backtest =======================
def backtest():
    d = add_indicators(get_data())
    sig, _ = compute_signals(d)
    o, h, l, atr = d.open.values, d.high.values, d.low.values, d.atr.values
    s = sig.values
    sp, k, rr = CONFIG["SPREAD_PRICE"], CONFIG["ATR_SL_MULT"], CONFIG["RR_TP2"]
    rs, i, n = [], 200, len(d)
    while i < n - 2:
        if s[i] == 0:
            i += 1
            continue
        dr = s[i]
        entry = o[i + 1] + dr * sp / 2
        risk = atr[i] * k
        sl, tp = entry - dr * risk, entry + dr * risk * rr
        res, j = None, i + 1
        while j < n:
            hit_sl = l[j] <= sl if dr == 1 else h[j] >= sl
            hit_tp = h[j] >= tp if dr == 1 else l[j] <= tp
            if hit_sl:                # ถ้าชนทั้งคู่ในแท่งเดียว ถือว่า SL ก่อน (เข้มงวด)
                res = -1 - sp / risk
                break
            if hit_tp:
                res = rr - sp / risk
                break
            j += 1
        if res is None:
            break
        rs.append(res)
        i = j + 1
    if not rs:
        print("ไม่มีเทรดเลย ลองลด MIN_VOTES หรือเพิ่มข้อมูล")
        return
    rs = np.array(rs)
    eq = rs.cumsum()
    gp, gl = rs[rs > 0].sum(), -rs[rs < 0].sum()
    print(f"จำนวนเทรด {len(rs)} | ชนะ {np.mean(rs > 0) * 100:.1f}% | Profit Factor {gp / gl if gl else float('inf'):.2f}")
    print(f"กำไรรวม {eq[-1]:.1f}R | Max Drawdown {np.max(np.maximum.accumulate(eq) - eq):.1f}R | เฉลี่ย {rs.mean():.2f}R/เทรด")
    print(f"ข้อมูล {d.index[0]:%Y-%m-%d} ถึง {d.index[-1]:%Y-%m-%d}  (ถ้า Yahoo จะได้แค่ ~30 วัน ควรใช้ MT5 เพื่อได้ข้อมูลยาวกว่า)")


if __name__ == "__main__":
    backtest() if len(sys.argv) > 1 and sys.argv[1] == "backtest" else live()
