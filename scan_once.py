"""
scan_once.py - สแกนครั้งเดียวแล้วจบ สำหรับรันบน GitHub Actions
ต้องวางไว้โฟลเดอร์เดียวกับ gold_signal.py
Token/Chat ID อ่านจาก GitHub Secrets (ไม่ต้องใส่ในโค้ด)
"""
import os
import datetime as dt
import pandas as pd
import gold_signal as g

g.CONFIG["TELEGRAM_TOKEN"] = os.environ.get("TELEGRAM_TOKEN", "")
g.CONFIG["TELEGRAM_CHAT_ID"] = os.environ.get("TELEGRAM_CHAT_ID", "")

MAX_AGE_MIN = 14  # แจ้งเฉพาะแท่งที่เพิ่งปิดไม่เกิน 14 นาที กันสัญญาณเก่าซ้ำ / ตอนตลาดปิด

now = dt.datetime.utcnow()
if now.hour == 0 and now.minute < 15:  # ราว 07:00 น. เวลาไทย วันละครั้ง
    g.notify("ระบบสแกนทองยังทำงานอยู่ ✅")
d = g.add_indicators(g.get_data())
d = d[d.index + pd.Timedelta(minutes=15) <= now]  # ตัดแท่งที่ยังไม่ปิด
t = d.index[-1]
age = (now - (t + pd.Timedelta(minutes=15))).total_seconds() / 60

if age > MAX_AGE_MIN:
    print(f"ข้อมูลล่าสุดเก่า {age:.0f} นาที (ตลาดอาจปิด) -> ข้าม")
    raise SystemExit

sig, votes = g.compute_signals(d)
s = sig.iloc[-1]
if s == 0:
    print(f"[{t:%H:%M} UTC] ยังไม่มีสัญญาณ")
    raise SystemExit

names = [n for n in votes.columns if votes[n].iloc[-1] == s]
p = g.build_plan(d, s)
sl_d = abs(p["entry"] - p["sl"])
msg = g.format_plan(p, t, names) + (
    f"\n ระยะห่าง: SL {sl_d:.1f} | TP1 {sl_d * g.CONFIG['RR_TP1']:.1f} | TP2 {sl_d * g.CONFIG['RR_TP2']:.1f} (ดอลลาร์ราคาทอง)\n"
    " หมายเหตุ: ราคาจาก Yahoo (ทองล่วงหน้า GC=F) อาจต่างจาก XAUUSD ใน MT5 ไม่กี่ดอลลาร์\n"
    " ให้ใช้ 'ระยะห่าง' นับจากราคาที่คุณเข้าจริงใน MT5 แทนตัวเลข SL/TP ด้านบน\n"
    " Lot ด้านบนคำนวณจากพอร์ตสมมติ ปรับตามพอร์ตจริงของคุณเอง\n"
)
g.notify(msg)
