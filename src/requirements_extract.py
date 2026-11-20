"""Extract buyer requirements from free text (Arabic, English and Franco-Arabic).

Used by the client chatbot (to learn what the client wants as they chat) and by
lead profiling (to turn years of messy sales notes into structured CRM fields).
"""
from __future__ import annotations

import re

from .inventory import normalize_digits, parse_amount

LOCATIONS = {
    "New Cairo": ["new cairo", "new caio", "tagamo3", "tagamoa", "el tagamo", "التجمع", "القاهرة الجديدة", "fifth settlement", "التجمع الخامس"],
    "New Capital": ["new capital", "nac", "العاصمة الإدارية", "العاصمة الادارية", "العاصمة", "el 3asema", "administrative capital"],
    "Sheikh Zayed": ["sheikh zayed", "zayed", "زايد", "الشيخ زايد"],
    "6th of October": ["6th of october", "october", "oct", "أكتوبر", "اكتوبر", "6 أكتوبر"],
    "Mostakbal City": ["mostakbal", "المستقبل", "future city"],
    "North Coast": ["north coast", "sahel", "el sahel", "الساحل", "الساحل الشمالي"],
    "Ain Sokhna": ["sokhna", "السخنة", "العين السخنة"],
    "Madinaty": ["madinaty", "مدينتي"],
    "Maadi": ["maadi", "المعادي"],
    "Nasr City": ["nasr city", "madinet nasr", "مدينة نصر"],
    "Heliopolis": ["heliopolis", "masr el gedida", "مصر الجديدة"],
    "Shorouk": ["shorouk", "الشروق"],
    "Assiut": ["assiut", "asyut", "أسيوط", "اسيوط"],
}

TYPES = {
    "Apartment": ["apartment", "apatrment", "appartment", "flat", "شقة", "شقه"],
    "Villa": ["villa", "فيلا", "standalone"],
    "Townhouse": ["townhouse", "town house", "تاون"],
    "Twin House": ["twin house", "twinhouse", "توين"],
    "Duplex": ["duplex", "دوبلكس"],
    "Penthouse": ["penthouse", "بنتهاوس"],
    "Chalet": ["chalet", "شاليه"],
    "Studio": ["studio", "tudio", "ستوديو", "استوديو"],
}

OBJECTIONS = {
    "price_high": ["غالي", "غالى", "expensive", "price high", "price is high", "too expensive", "السعر عالي"],
    "maintenance": ["maintenance", "الصيانة", "صيانة"],
    "delivery_late": ["الاستلام متأخر", "estlam metakhar", "delivery late", "late delivery", "delivery too far"],
    "short_installments": ["installment period too short", "سنين التقسيط قليلة", "تقسيط قصير", "years too short"],
    "high_down_payment": ["down payment too high", "dp too high", "المقدم كبير", "المقدم عالي", "المقدم ميزيدش"],
    "comparing": ["بيقارن", "comparing", "compare with", "another project", "مشروع تاني"],
    "developer_trust": ["not sure about the developer", "المطور", "developer reputation"],
    "location": ["location far", "بعيد", "far away"],
}

TIMING_LATER = ["مش مستعجل", "maybe next year", "next year", "السنة الجاية", "after ramadan", "بعد رمضان",
                "next month", "الشهر الجاي", "not in a hurry", "will think", "هيفكر"]
UNREACHABLE = ["voicemail", "مردش", "مبيردش", "مش بيرد", "switched off", "no answer", "didn't answer",
               "not reachable", "مقفول", "busy"]
LOST = ["bought from another broker", "not interested", "مش مهتم", "اشترى من", "bought elsewhere", "cancel"]
POSITIVE = ["site visit", "after site visit", "visit", "meeting", "met at office", "قابلته", "معاينة", "interested",
            "مهتم", "reserve", "حجز"]
CONTACT_REQUEST = ["call me", "contact me", "agent", "sales", "كلمني", "كلموني", "اتصل", "اتصلوا", "موظف",
                   "حد يكلمني", "عايز اكلم", "رقمي", "my number", "my phone"]

_AMOUNT = r"\d[\d,٬.]*\s*(?:m\b|mil\b|mn\b|million|مليون|k\b|الف|ألف)?"


def _has(text: str, words) -> bool:
    return any(re.search(rf"(?<![\w]){re.escape(w)}(?![\w])", text) for w in words)


def _first_match(text: str, table: dict) -> str | None:
    best = None
    for name, words in table.items():
        for w in words:
            m = re.search(rf"(?<![\w]){re.escape(w)}(?![\w])", text)
            if m and (best is None or m.start() < best[0]):
                best = (m.start(), name)
    return best[1] if best else None


def _budget(text: str) -> tuple[float | None, float | None]:
    """Return (min, max). 'budget 4.8 to 5.9 million' -> (4.8M, 5.9M); 'budget max 6.26M' -> (None, 6.26M)."""
    cue = r"(?:budget|الميزانية|ميزانية|ميزانيتي|في حدود|حوالي|7awaly|around|max|up to|لحد|تحت|under|مش أكتر من|مش اكتر من)"
    rng = re.search(rf"{cue}[^\d]{{0,25}}({_AMOUNT})\s*(?:to|-|الى|إلى|لـ|ل)\s*({_AMOUNT})", text)
    if rng:
        a, b = rng.group(1), rng.group(2)
        unit = re.search(r"(m\b|mil\b|million|مليون|k\b|الف|ألف)", b)
        if unit and not re.search(r"(m\b|mil\b|million|مليون|k\b|الف|ألف)", a):
            a = a + " " + unit.group(1)
        lo, hi = parse_amount(a), parse_amount(b)
        if lo and hi and hi >= 100_000:
            return (min(lo, hi), max(lo, hi))
    m = re.search(rf"{cue}[^\d]{{0,25}}({_AMOUNT})", text)
    if m:
        v = parse_amount(m.group(1))
        if v and v >= 100_000:
            return (None, v)
    # bare amounts with a money unit, e.g. "5 مليون" or "3,200,000 جنيه"
    for m in re.finditer(rf"({_AMOUNT})\s*(?:جنيه|egp|le\b|نيه)?", text):
        v = parse_amount(m.group(1))
        if v and v >= 300_000 and re.search(r"(m\b|mil\b|million|مليون|k\b|الف|ألف|جنيه|egp|\d{6,}|\d{1,3},\d{3},\d{3})", m.group(0)):
            return (None, v)
    return (None, None)


def extract(text: str) -> dict:
    """Return only the fields that the text actually states."""
    t = normalize_digits(text or "").lower().replace("٬", ",")
    out: dict = {}
    phone = re.search(r"(?:\+?20|0)1[0125]\d{8}", re.sub(r"[\s-]", "", t))
    # never read a phone number as a budget
    t_no_phone = re.sub(r"(?:\+?\s*20|0)\s*1[0125](?:[\s-]*\d){8}", " ", t)
    lo, hi = _budget(t_no_phone)
    if hi:
        out["budget"] = hi
    if lo:
        out["budget_min"] = lo
    beds = re.search(r"(\d+)\s*(?:br\b|r\b|bed|bedroom|beds|ghoraf|ghorfa|غرف|غرفة|غرفه|اوض|أوض|اوضة)", t)
    if beds and 0 < int(beds.group(1)) <= 8:
        out["bedrooms"] = int(beds.group(1))
    elif _has(t, ["studio", "tudio", "ستوديو", "استوديو"]):
        out["bedrooms"] = 0
    loc = _first_match(t, LOCATIONS)
    if loc:
        out["location"] = loc
    typ = _first_match(t, TYPES)
    if typ:
        out["unit_type"] = typ
    if _has(t, ["cash", "كاش"]):
        out["payment"] = "cash"
    elif _has(t, ["installment", "installments", "تقسيط", "أقساط", "اقساط", "mortgage", "قسط"]):
        out["payment"] = "installments"
    dp = re.search(r"(\d{1,2})\s*%\s*(?:dp|down|مقدم)", t) or re.search(r"(?:مقدم|down payment|dp)[^\d%]{0,20}(\d{1,2})\s*%", t)
    if dp:
        out["dp_pct"] = int(dp.group(1))
    years = re.search(r"(\d{1,2})\s*(?:y\s*rs|yrs|years|year|سنين|سنة|سنوات)", t)
    if years and out.get("payment") != "cash" and 2 <= int(years.group(1)) <= 15:
        out["installment_years"] = int(years.group(1))
    if _has(t, ["fully finished", "متشطب", "متشطبة", "تشطيب كامل", "finished"]) and "not fully finished" not in t:
        out["finishing"] = "finished"
    elif _has(t, ["core & shell", "core and shell", "نص تشطيب", "semi-finished", "semi finished"]):
        out["finishing"] = "core_shell"
    objections = [k for k, words in OBJECTIONS.items() if _has(t, words)]
    if objections:
        out["objections"] = objections
    if _has(t, TIMING_LATER):
        out["timing"] = "later"
    if phone:
        digits = phone.group(0).lstrip("+")
        out["phone"] = "+20" + digits[-10:] if not digits.startswith("20") else "+" + digits
    return out


def outcome_signals(text: str) -> dict:
    t = normalize_digits(text or "").lower()
    return {"unreachable": _has(t, UNREACHABLE), "lost": _has(t, LOST), "positive": _has(t, POSITIVE)}


def wants_contact(text: str) -> bool:
    return _has(normalize_digits(text or "").lower(), CONTACT_REQUEST)
