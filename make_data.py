"""근거 기반 응답 선호 데이터 생성.

입력: 조회된 제조 기록(근거 ID 포함) + 질문
- 답할 수 있는 질문 → 선호(chosen): 기록의 값 + 근거 ID 인용
- 답할 수 없는 질문(해당 LOT·항목의 기록 없음) → 선호(chosen): 근거 부족으로 답변 거절
비선호(rejected): 값 지어내기, 근거 ID 누락·오인용, 근거가 없는데도 답하기(환각)
"""
import json
import random
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent / "data"
FIELDS = {
    "inspection": ("품질검사", "판정", ["OK", "NG"], "검사 판정은"),
    "hold": ("보류", "사유", ["치수 불량", "외관 불량", "이물 검출", "설비 이상"], "보류 사유는"),
    "packing": ("포장", "팔레트", None, "포장 팔레트는"),
    "material": ("자재투입", "자재", ["MAT-NCM-01", "MAT-NCM-02", "MAT-AL-01", "MAT-CU-01", "MAT-GR-01"], "투입 자재는"),
}
QUESTIONS = {
    "inspection": ["{lot}의 검사 판정 결과는?", "LOT {lot} 품질검사 OK야 NG야?"],
    "hold": ["{lot}이 보류된 사유가 뭐야?", "LOT {lot} 보류 사유 알려줘"],
    "packing": ["{lot}은 어느 팔레트에 포장됐어?", "LOT {lot}의 포장 팔레트 번호는?"],
    "material": ["{lot}에 투입된 자재 코드는?", "LOT {lot}에 들어간 자재가 뭐야?"],
}
SYSTEM = ("너는 제조 LOT 조회 도우미다. 제공된 조회 기록만 근거로 답한다. 답할 때는 값과 함께 '[근거: 기록ID]'를 붙인다. "
          "조회 기록에 해당 LOT·항목이 없으면 지어내지 말고 '조회된 기록에 근거가 없어 답할 수 없습니다.'라고 답한다.")
REFUSAL = "조회된 기록에 근거가 없어 답할 수 없습니다."


def rand_lot(rng):
    return f"LT26{rng.randint(1, 9):02d}-{rng.randint(1, 999):04d}"


def rand_value(rng, field):
    choices = FIELDS[field][2]
    return rng.choice(choices) if choices else f"PL-{rng.randint(10, 99)}"


def make_example(rng, answerable: bool) -> dict:
    lots = list({rand_lot(rng) for _ in range(3)})
    records = []
    for lot in lots:
        for field in rng.sample(list(FIELDS), k=rng.randint(1, 3)):
            records.append({"ev": f"EV-{rng.randint(1000, 9999)}", "lot": lot, "field": field, "value": rand_value(rng, field)})
    rng.shuffle(records)
    if answerable:
        target = rng.choice(records)
        lot, field, value, ev = target["lot"], target["field"], target["value"], target["ev"]
    else:
        if rng.random() < 0.5:  # 기록에 아예 없는 LOT
            lot, field = rand_lot(rng), rng.choice(list(FIELDS))
        else:  # 기록에 있는 LOT이지만 묻는 항목은 없음
            lot = rng.choice(lots)
            missing = [f for f in FIELDS if not any(r["lot"] == lot and r["field"] == f for r in records)]
            if not missing:
                lot, missing = rand_lot(rng), list(FIELDS)
            field = rng.choice(missing)
        value, ev = None, None
    context = "\n".join(f"[{r['ev']}] LOT {r['lot']} | {FIELDS[r['field']][0]} | {FIELDS[r['field']][1]}: {r['value']}" for r in records)
    question = rng.choice(QUESTIONS[field]).format(lot=lot)
    prompt = f"[조회 기록]\n{context}\n\n[질문] {question}"
    lead = f"{lot}의 {FIELDS[field][3]}"

    if answerable:
        chosen = f"{lead} {value}입니다. [근거: {ev}]"
        others = [r for r in records if r["ev"] != ev]
        wrong_value = rng.choice([v for v in (FIELDS[field][2] or ["PL-00"]) if v != value] or ["확인 불가"])
        rejected = rng.choice([
            f"{lead} {wrong_value}입니다. [근거: {ev}]",  # 값 오류
            f"{lead} {value}입니다.",  # 근거 누락
            f"{lead} {value}입니다. [근거: {rng.choice(others)['ev'] if others else 'EV-0000'}]",  # 근거 오인용
            REFUSAL,  # 과잉 거절
        ])
    else:
        chosen = REFUSAL
        fake_ev = rng.choice(records)["ev"]
        rejected = rng.choice([
            f"{lead} {rand_value(rng, field)}입니다. [근거: {fake_ev}]",  # 근거 없는데 답함(환각)
            f"{lead} {rand_value(rng, field)}입니다.",
        ])
    return {"prompt": prompt, "chosen": chosen, "rejected": rejected, "answerable": answerable,
            "gold_value": value, "gold_ev": ev, "lot": lot, "field": field}


def main(n_train: int = 400, n_eval: int = 80, seed: int = 0):
    rng = random.Random(seed)
    DATA_DIR.mkdir(exist_ok=True)
    splits = {"train": n_train, "eval": n_eval}
    for name, n in splits.items():
        rows = [make_example(rng, answerable=i % 2 == 0) for i in range(n)]
        with (DATA_DIR / f"{name}.jsonl").open("w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps({"system": SYSTEM, **r}, ensure_ascii=False) + "\n")
        print(f"{name}: {len(rows)}건 (답변 가능 {sum(r['answerable'] for r in rows)} / 거절 대상 {sum(not r['answerable'] for r in rows)})")


if __name__ == "__main__":
    main()
