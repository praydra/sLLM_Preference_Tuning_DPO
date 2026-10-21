"""소형 LLM 근거 기반 응답 정렬: LoRA SFT → DPO (CPU 학습).

단계
0. base : Qwen2.5-0.5B-Instruct 원본
1. sft  : 선호 응답(chosen)으로 LoRA 지도 미세조정 → 병합. 손실은 답변(completion) 토큰에만 계산
2. dpo  : SFT 모델에서 (chosen, rejected) 쌍으로 DPO(LoRA). 기준(reference) 모델은 SFT 모델.
          DPO 손실(sigmoid)에 선호 응답의 SFT 손실을 더해, 선호 응답 확률까지 함께 떨어지는 붕괴를 막음
평가(답변 가능 40 / 거절 대상 40)
- 답변 정확도: 정답 값과 정답 근거 ID를 모두 포함
- 거절 정확도: 근거 없는 질문에 거절
- 환각률: 근거 없는 질문에 값을 지어내 답함
- 과잉 거절률: 답할 수 있는 질문을 거절
실행: python make_data.py && python train_eval.py
"""
import argparse
import json
import re
import time
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import torch
from datasets import load_dataset
from peft import LoraConfig, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DPOConfig, DPOTrainer, SFTConfig, SFTTrainer

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
OUT_DIR = BASE_DIR / "outputs"  # 학습 체크포인트(저장소에는 올리지 않음)
RESULT_DIR = BASE_DIR / "results"
BASE_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
SEED = 42
LORA = dict(r=16, lora_alpha=32, lora_dropout=0.05, target_modules=["q_proj", "k_proj", "v_proj", "o_proj"], task_type="CAUSAL_LM")


def messages(row, with_answer=None):
    msgs = [{"role": "system", "content": row["system"]}, {"role": "user", "content": row["prompt"]}]
    if with_answer is not None:
        msgs.append({"role": "assistant", "content": with_answer})
    return msgs


def to_sft(row):
    # prompt-completion 형식: 지시문·조회 기록(prompt)은 손실에서 제외하고 답변에만 손실 계산
    return {"prompt": messages(row), "completion": [{"role": "assistant", "content": row["chosen"]}]}


def to_dpo(row):
    return {"prompt": messages(row), "chosen": [{"role": "assistant", "content": row["chosen"]}],
            "rejected": [{"role": "assistant", "content": row["rejected"]}]}


def evaluate(model, tokenizer, eval_rows, tag: str) -> pd.DataFrame:
    model.eval()
    out = []
    start = time.time()
    for row in eval_rows:
        text = tokenizer.apply_chat_template(messages(row), tokenize=False, add_generation_prompt=True)
        ids = tokenizer(text, return_tensors="pt")
        with torch.no_grad():
            gen = model.generate(**ids, max_new_tokens=60, do_sample=False, pad_token_id=tokenizer.eos_token_id)
        answer = tokenizer.decode(gen[0, ids["input_ids"].shape[1]:], skip_special_tokens=True).strip()
        refused = bool(re.search(r"답할 수 없|근거가 없|기록이 없|알 수 없", answer))
        cited = re.findall(r"EV-\d{4}", answer)
        if row["answerable"]:
            correct = (row["gold_value"] in answer) and (row["gold_ev"] in cited) and not refused
        else:
            correct = refused and not cited
        out.append({"stage": tag, "answerable": row["answerable"], "field": row["field"], "answer": answer,
                    "correct": correct, "refused": refused, "cited": bool(cited),
                    "hallucinated": (not row["answerable"]) and not refused})
    print(f"[{tag}] 평가 {len(eval_rows)}건 {time.time() - start:.0f}s")
    return pd.DataFrame(out)


def summarize(df: pd.DataFrame) -> dict:
    a, u = df[df["answerable"]], df[~df["answerable"]]
    return {"overall_acc": df["correct"].mean(), "answer_acc": a["correct"].mean(), "refusal_acc": u["correct"].mean(),
            "hallucination_rate": u["hallucinated"].mean(), "over_refusal_rate": a["refused"].mean()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sft-epochs", type=float, default=1.0)
    parser.add_argument("--dpo-epochs", type=float, default=1.0)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--sft-weight", type=float, default=1.0, help="DPO에 더하는 선호 응답 SFT 손실 가중치")
    parser.add_argument("--max-steps", type=int, default=-1, help="동작 확인용(짧게 실행)")
    parser.add_argument("--eval-n", type=int, default=0, help="0이면 전체 평가")
    args = parser.parse_args()
    torch.manual_seed(SEED)
    RESULT_DIR.mkdir(exist_ok=True)
    OUT_DIR.mkdir(exist_ok=True)

    raw = load_dataset("json", data_files={"train": str(DATA_DIR / "train.jsonl"), "eval": str(DATA_DIR / "eval.jsonl")})
    eval_rows = list(raw["eval"])
    if args.eval_n:
        eval_rows = eval_rows[: args.eval_n // 2] + eval_rows[-(args.eval_n // 2):]
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    results, logs = [], {}

    # 0. 원본 모델
    base = AutoModelForCausalLM.from_pretrained(BASE_MODEL, dtype=torch.float32)
    results.append(evaluate(base, tokenizer, eval_rows, "base"))

    # 1. LoRA SFT
    sft_args = SFTConfig(output_dir=str(OUT_DIR / "sft"), num_train_epochs=args.sft_epochs, max_steps=args.max_steps,
                         per_device_train_batch_size=4, gradient_accumulation_steps=2, learning_rate=2e-4,
                         lr_scheduler_type="cosine", warmup_steps=3, logging_steps=5, save_strategy="no",
                         report_to=[], seed=SEED, bf16=False, use_cpu=True, max_length=512)
    sft = SFTTrainer(model=base, args=sft_args, train_dataset=raw["train"].map(to_sft, remove_columns=raw["train"].column_names),
                     processing_class=tokenizer, peft_config=LoraConfig(**LORA))
    start = time.time()
    sft.train()
    logs["sft"] = {"train_time_sec": round(time.time() - start, 1), "log_history": sft.state.log_history}
    sft_model = sft.model.merge_and_unload()
    results.append(evaluate(sft_model, tokenizer, eval_rows, "sft"))
    sft_model.save_pretrained(OUT_DIR / "sft_merged")
    tokenizer.save_pretrained(OUT_DIR / "sft_merged")

    # 2. DPO (기준 모델 = SFT 병합 모델, LoRA 어댑터를 끈 상태가 reference 역할)
    policy = AutoModelForCausalLM.from_pretrained(OUT_DIR / "sft_merged", dtype=torch.float32)
    dpo_args = DPOConfig(output_dir=str(OUT_DIR / "dpo"), num_train_epochs=args.dpo_epochs, max_steps=args.max_steps,
                         per_device_train_batch_size=4, gradient_accumulation_steps=2, learning_rate=5e-5,
                         lr_scheduler_type="cosine", warmup_steps=3, logging_steps=5, save_strategy="no",
                         report_to=[], seed=SEED, bf16=False, use_cpu=True, beta=args.beta, max_length=512,
                         loss_type=["sigmoid", "sft"], loss_weights=[1.0, args.sft_weight])
    dpo = DPOTrainer(model=policy, args=dpo_args, train_dataset=raw["train"].map(to_dpo, remove_columns=raw["train"].column_names),
                     processing_class=tokenizer, peft_config=LoraConfig(**LORA))
    start = time.time()
    dpo.train()
    logs["dpo"] = {"train_time_sec": round(time.time() - start, 1), "log_history": dpo.state.log_history}
    dpo_model = dpo.model.merge_and_unload()
    results.append(evaluate(dpo_model, tokenizer, eval_rows, "sft+dpo"))

    df = pd.concat(results, ignore_index=True)
    df.to_csv(RESULT_DIR / "eval_answers.csv", index=False, encoding="utf-8-sig")
    summary = {stage: summarize(g) for stage, g in df.groupby("stage", sort=False)}
    meta = {"base_model": BASE_MODEL, "lora": LORA, "args": vars(args), "n_train": len(raw["train"]), "n_eval": len(eval_rows),
            "train_time_sec": {k: v["train_time_sec"] for k, v in logs.items()}}
    (RESULT_DIR / "metrics.json").write_text(json.dumps({"meta": meta, "summary": summary}, ensure_ascii=False, indent=2), encoding="utf-8")
    (RESULT_DIR / "train_logs.json").write_text(json.dumps(logs, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(pd.DataFrame(summary).round(3))

    plt.rcParams["font.family"] = "Malgun Gothic"
    plt.rcParams["axes.unicode_minus"] = False
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.5))
    s = pd.DataFrame(summary).T[["answer_acc", "refusal_acc", "hallucination_rate", "over_refusal_rate"]] * 100
    s.columns = ["답변 정확도", "거절 정확도", "환각률", "과잉 거절률"]
    s.plot.bar(ax=axes[0], rot=0)
    axes[0].set(title="단계별 평가 지표(%)", ylim=(0, 105))
    for c in axes[0].containers:
        axes[0].bar_label(c, fmt="%.0f", fontsize=7)
    sft_loss = [(h["step"], h["loss"]) for h in logs["sft"]["log_history"] if "loss" in h]
    axes[1].plot(*zip(*sft_loss), marker="o")
    axes[1].set(title="SFT 학습 손실", xlabel="step")
    dpo_hist = [h for h in logs["dpo"]["log_history"] if "loss" in h]
    axes[2].plot([h["step"] for h in dpo_hist], [h["loss"] for h in dpo_hist], marker="o", label="DPO loss")
    margin_key = next((k for k in ("rewards/margins", "margins") if dpo_hist and k in dpo_hist[0]), None)
    if margin_key:
        ax2 = axes[2].twinx()
        ax2.plot([h["step"] for h in dpo_hist], [h[margin_key] for h in dpo_hist], color="tab:orange", marker="s", label="보상 마진")
        ax2.set_ylabel("chosen - rejected 보상 마진")
    axes[2].set(title="DPO 손실과 보상 마진", xlabel="step")
    for ax in axes[1:]:
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(RESULT_DIR / "dpo_results.png", dpi=150)


if __name__ == "__main__":
    main()
