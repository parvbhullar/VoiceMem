"""Deterministic returning-caller workload for the cartridge benchmark.

One hospital (the organisation, cartridge A), N callers (cartridge B each) and
their accounts (A x B). Every question has an answer that exists in exactly one
place in the context, plus "cross" questions that need two cartridges at once
(the caller's doctor from B, that doctor's room from A), which is where naive
KV reuse would lose quality first.

Everything is generated from a seed, so every arm and every rerun reads the
same tokens and asks the same questions.
"""
from __future__ import annotations

import random
import re
from dataclasses import dataclass, field

from supermem.cartridge.compiler import Fact

ORG_ID = "sunrise-hospital"
TENANT = "unpod-demo"

DEPARTMENTS = [
    "Cardiology", "Orthopaedics", "Neurology", "Dermatology", "ENT", "Gastroenterology",
    "Nephrology", "Pulmonology", "Endocrinology", "Ophthalmology", "Paediatrics",
    "Gynaecology", "Urology", "Oncology", "Psychiatry", "Physiotherapy",
]
FIRST = ["Aarav", "Meera", "Rohan", "Kavya", "Vikram", "Ananya", "Arjun", "Isha", "Karan",
         "Nisha", "Rahul", "Priya", "Sameer", "Divya", "Aditya", "Sneha", "Manish", "Pooja",
         "Varun", "Ritu", "Neha", "Suresh", "Tanvi", "Harsh"]
LAST = ["Sharma", "Iyer", "Verma", "Reddy", "Nair", "Gupta", "Kapoor", "Menon", "Singh",
        "Bose", "Joshi", "Pillai", "Malhotra", "Rao", "Chatterjee", "Desai"]
DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"]
SLOTS = ["8:30 AM to 12:30 PM", "9:00 AM to 1:00 PM", "10:00 AM to 2:00 PM",
         "2:00 PM to 6:00 PM", "3:30 PM to 7:30 PM", "5:00 PM to 9:00 PM"]
MEDS = ["metformin 500 mg", "atorvastatin 10 mg", "levothyroxine 50 mcg", "amlodipine 5 mg",
        "pantoprazole 40 mg", "montelukast 10 mg", "sertraline 50 mg", "telmisartan 40 mg",
        "vitamin D3 60000 IU weekly", "insulin glargine 12 units"]
ALLERGIES = ["penicillin", "sulfa drugs", "ibuprofen", "latex", "iodine contrast", "aspirin",
             "peanuts", "shellfish", "codeine", "amoxicillin"]
LANGS = ["Hinglish", "Hindi", "English", "Tamil and English", "Kannada and English"]
PLANS = ["Star Health Comprehensive", "HDFC Ergo Optima", "ICICI Lombard Complete",
         "Niva Bupa ReAssure", "CGHS", "Self-pay"]
CITIES = ["Indiranagar", "Whitefield", "Koramangala", "HSR Layout", "Jayanagar", "Hebbal"]


@dataclass
class Question:
    text: str
    expect: list[str]              # every string must appear in the answer (normalised)
    source: str                    # org | user | rel | cross


@dataclass
class Caller:
    user_id: str
    name: str
    facts: list[Fact]
    traits: list[str]
    account: dict[str, str]
    questions: list[Question] = field(default_factory=list)


@dataclass
class Workload:
    org_sections: list[tuple[str, str]]
    callers: list[Caller]
    seed: int


def _norm(s: str) -> str:
    s = s.lower().replace(",", "")
    return re.sub(r"\s+", " ", s)


def score(answer: str, expect: list[str]) -> bool:
    a = _norm(answer)
    return all(_norm(e) in a for e in expect)


def _org(rng: random.Random, org_tokens: int):
    doctors: dict[str, list[dict]] = {}
    used_names: set[str] = set()
    sections: list[tuple[str, str]] = [
        ("About", "Sunrise Multispeciality Hospital, 14th Cross, Indiranagar, Bengaluru. "
                  "Front desk helpline 080-4719-2200. Emergency is open 24 hours on the ground floor."),
    ]
    for i, dept in enumerate(DEPARTMENTS):
        docs = []
        for _ in range(3):
            while True:
                name = f"Dr. {rng.choice(FIRST)} {rng.choice(LAST)}"
                if name not in used_names:
                    used_names.add(name)
                    break
            days = sorted(rng.sample(DAYS, 3), key=DAYS.index)
            docs.append({"name": name, "days": days, "slot": rng.choice(SLOTS),
                         "room": f"{i % 5 + 1}{rng.randint(0, 3)}{rng.randint(1, 9)}"})
        doctors[dept] = docs
        fee = rng.choice([600, 700, 800, 900, 1000, 1200, 1500])
        tele = rng.choice([400, 500, 600])
        lines = [f"Consultation fee: Rs {fee}. Tele-consult fee: Rs {tele}.",
                 f"Follow-up within {rng.choice([7, 10, 14])} days is free."]
        for d in docs:
            lines.append(f"{d['name']}: OPD on {', '.join(d['days'])}, {d['slot']}, room {d['room']}.")
        sections.append((f"Department: {dept}", "\n".join(lines)))

    sections += [
        ("Cancellation policy", "Appointments can be cancelled free of charge up to 6 hours before the "
                                "slot. Later cancellations are charged Rs 200. No-shows are charged the "
                                "full consultation fee."),
        ("Laboratory", "Sample collection runs 7:00 AM to 7:00 PM daily. Fasting tests must be booked "
                       "before 10:00 AM. Home collection costs Rs 150 within 8 km."),
        ("Pharmacy", "The pharmacy on the ground floor is open 24 hours. Prescriptions older than "
                     "30 days need a fresh doctor's note."),
        ("Insurance desk", "Cashless is supported for Star Health, HDFC Ergo, ICICI Lombard, Niva Bupa "
                           "and CGHS. Pre-authorisation takes up to 4 hours for planned admissions."),
    ]

    # Pad the organisation cartridge with authored-looking policy clauses until it
    # reaches the requested size, so the benchmark runs at the context length it claims.
    topics = ["visitor passes", "parking", "wheelchair assistance", "medical records", "billing",
              "discharge", "diet counselling", "vaccination", "health check packages", "refunds",
              "second opinions", "international patients", "blood bank", "ambulance", "feedback"]
    clause = 1
    body = "\n\n".join(t + t2 for t, t2 in sections)
    extra: list[str] = []
    while (len(body) + sum(len(x) for x in extra)) // 4 < org_tokens:
        topic = topics[clause % len(topics)]
        extra.append(
            f"Clause {clause:03d} ({topic}): requests about {topic} are handled at counter "
            f"{rng.randint(1, 12)} between {rng.choice(SLOTS)}; reference code "
            f"SR-{rng.randint(1000, 9999)}; turnaround {rng.randint(1, 5)} working days; "
            f"escalation to the duty manager on extension {rng.randint(200, 499)}.")
        clause += 1
    if extra:
        sections.append(("Service clauses", "\n".join(extra)))
    return sections, doctors


def build(n_callers: int = 10, org_tokens: int = 11000, user_tokens: int = 3000,
          seed: int = 7) -> Workload:
    rng = random.Random(seed)
    sections, doctors = _org(rng, org_tokens)
    callers: list[Caller] = []
    fees = {title.split(": ", 1)[1]: int(re.search(r"Rs (\d+)", text).group(1))
            for title, text in sections if title.startswith("Department: ")}

    for i in range(n_callers):
        name = f"{FIRST[(i * 5) % len(FIRST)]} {LAST[(i * 3) % len(LAST)]}"
        uid = f"caller-{i + 1:03d}"
        dept = DEPARTMENTS[(i * 7) % len(DEPARTMENTS)]
        doc = doctors[dept][i % 3]
        med = MEDS[i % len(MEDS)]
        allergy = ALLERGIES[(i * 3) % len(ALLERGIES)]
        lang = LANGS[i % len(LANGS)]
        pref_day = doc["days"][i % 3]
        city = CITIES[i % len(CITIES)]
        facts = [
            Fact("identity", f"{name} lives in {city}, Bengaluru.", "2026-03-02"),
            Fact("health", f"{name} is treated in {dept} by {doc['name']}.", "2026-03-02"),
            Fact("health", f"{name} takes {med} every day.", "2026-04-11"),
            Fact("health", f"{name} is allergic to {allergy}.", "2026-04-11"),
            Fact("preference", f"{name} prefers to speak {lang}.", "2026-03-02"),
            Fact("preference", f"{name} can only come on {pref_day}s.", "2026-05-19"),
        ]
        # Past-call summaries: the long tail a real memory accumulates.
        day = 3
        while sum(len(f.content) for f in facts) // 4 < user_tokens:
            facts.append(Fact(
                "history",
                f"Call on 2026-06-{day % 28 + 1:02d}: {name} asked about "
                f"{rng.choice(['a lab report', 'a refill', 'a bill', 'parking', 'a follow-up', 'a diet chart'])}; "
                f"agent {rng.choice(['resolved it', 'raised ticket SR-' + str(rng.randint(1000, 9999)), 'sent an SMS'])}; "
                f"caller sounded {rng.choice(['calm', 'rushed', 'worried', 'relieved'])}.",
                f"2026-06-{day % 28 + 1:02d}"))
            day += 1
        traits = [
            "Gets anxious about severity; give a concrete next step before reassurance.",
            f"Switches to {lang} mid-sentence; mirror it.",
        ]
        pid = f"SUN-{rng.randint(100000, 999999)}"
        balance = rng.choice([350, 800, 1250, 2400, 4150])
        hour, meridiem = (rng.choice(["9", "10", "11"]), "AM") if i % 2 == 0 else \
            (rng.choice(["3", "4", "5"]), "PM")
        appt = f"{pref_day} {hour}:{rng.choice(['00', '15', '30', '45'])} {meridiem}"
        plan = PLANS[i % len(PLANS)]
        account = {
            "patient id": pid,
            "insurance plan": plan,
            "outstanding balance": f"Rs {balance}",
            "next appointment": f"{appt} with {doc['name']} ({dept})",
            "last lab report": rng.choice(["HbA1c 7.2%", "LDL 132 mg/dL", "TSH 5.8", "Vitamin D 14 ng/mL",
                                           "Creatinine 1.4 mg/dL"]),
        }
        c = Caller(uid, name, facts, traits, account)
        c.questions = [
            Question("Mera next appointment kab hai?", [appt.split(" ", 1)[1]], "rel"),
            Question("What's my patient ID?", [pid], "rel"),
            Question("Which doctor do I usually see?", [doc["name"].replace("Dr. ", "")], "user"),
            Question("Mujhe kis cheez se allergy hai, pharmacist ko batana hai.", [allergy], "user"),
            Question("Which medicine am I on?", [med.split(" ")[0]], "user"),
            Question(f"What is the consultation fee in {dept}?", [str(fees[dept])], "org"),
            Question("Till how many hours before can I cancel for free?", ["6 hours"], "org"),
            Question("Kitna paisa baaki hai mera?", [str(balance)], "rel"),
            Question("Which room should I go to for my doctor?", [doc["room"]], "cross"),
            Question("What were the timings of my doctor's OPD?", [doc["slot"].split(" to ")[0]], "cross"),
        ]
        callers.append(c)
    return Workload(sections, callers, seed)


def turn_order(workload: Workload, turns: int) -> list[tuple[Caller, Question, int]]:
    """Round-robin across callers (caller 1 turn 1, caller 2 turn 1, ...), the
    way concurrent calls interleave on a real line. Returns (caller, question,
    turn index within that caller's call)."""
    order = []
    per = max(len(c.questions) for c in workload.callers)
    for t in range(per):
        for c in workload.callers:
            if t < len(c.questions):
                order.append((c, c.questions[t], t))
            if len(order) == turns:
                return order
    return order
