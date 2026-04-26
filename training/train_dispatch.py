"""
Dispatch RL training script.
Run: python training/train_dispatch.py [--task easy] [--steps 300] [--episodes 200]

Requirements:
    pip install unsloth trl datasets requests pydantic matplotlib uvicorn fastapi
"""
import sys, os, re, time, random, argparse, threading, json
_ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "server"))

import requests
import matplotlib.pyplot as plt

from openenv_http_session import post_reset, post_step

# ── CLI args ──────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--task",     default="easy",  choices=["easy","medium","hard","crisis","mixed"])
parser.add_argument("--steps",    default=300,     type=int)
parser.add_argument("--episodes", default=200,     type=int)
parser.add_argument("--model",    default="unsloth/Qwen2.5-7B-Instruct")
parser.add_argument("--port",     default=8000,    type=int)
args = parser.parse_args()

ENV_URL = f"http://127.0.0.1:{args.port}"

TASKS = ["easy", "medium", "hard", "crisis"]
# Default "realistic-ish" mixture; tweak as desired.
TASK_WEIGHTS = [0.50, 0.30, 0.15, 0.05]

def sample_task() -> str:
    if args.task == "mixed":
        return random.choices(TASKS, weights=TASK_WEIGHTS, k=1)[0]
    return args.task

# ── Start OpenEnv server ──────────────────────────────────────────────────────
def start_server():
    import uvicorn
    from server.app import app

    uvicorn.run(app, host="127.0.0.1", port=args.port, log_level="error")

server_thread = threading.Thread(target=start_server, daemon=True)
server_thread.start()

# Wait until server is up
for _ in range(30):
    try:
        requests.get(f"{ENV_URL}/health", timeout=1)
        break
    except Exception:
        time.sleep(1)
else:
    raise RuntimeError("OpenEnv server did not start within 30 seconds")

print(f"[OK] OpenEnv server running on port {args.port}")


# ── Action parser ─────────────────────────────────────────────────────────────
def parse_action(text: str) -> dict:
    def find_int(pattern, default):
        m = re.search(pattern, text, re.IGNORECASE)
        return int(m.group(1)) if m else default

    def find_str(pattern, choices, default):
        m = re.search(pattern, text, re.IGNORECASE)
        val = m.group(1).strip().lower() if m else default
        return val if val in choices else default

    action = {
        "ambulance_units":   find_int(r"ambulance[s]?\s*[:\-]\s*(\d+)", 0),
        "police_units":      find_int(r"police\s*[:\-]\s*(\d+)", 0),
        "fire_units":        find_int(r"fire\s*[:\-]\s*(\d+)", 0),
        "priority_level":    find_int(r"priority\s*[:\-]\s*(\d+)", 2),
        "coordination_level": find_str(
            r"coordination\s*[:\-]\s*(\w+)",
            {"none", "mutual_aid", "mci_protocol"}, "none"),
        "hospital_choice":    find_str(
            r"hospital\s*[:\-]\s*(\w+)",
            {"nearest", "regional", "auto"}, "nearest"),
        "ambulance_staging":  find_str(
            r"staging\s*[:\-]\s*(\w+)",
            {"dispatch", "stage_nearby", "on_scene_hold"}, "dispatch"),
    }
    # Server-side action schema requires at least one dispatched unit.
    if action["ambulance_units"] == 0 and action["police_units"] == 0 and action["fire_units"] == 0:
        action["ambulance_units"] = 1
    return action


# ── Prompt builder ────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are an emergency dispatcher. For each incoming call, respond with:
ambulance: <count>
police: <count>
fire: <count>
priority: <1-4>
coordination: <none|mutual_aid|mci_protocol>
hospital: <nearest|regional|auto>
staging: <dispatch|stage_nearby|on_scene_hold>

Respond only with those 7 lines. No explanation."""

def obs_to_prompt(obs: dict, task: str | None = None) -> list[dict]:
    task_line = f"TASK: {task}\n" if task else ""
    user_msg = (
        task_line +
        f"INCOMING CALL [{obs.get('call_id','')}]\n"
        f"Type: {obs.get('incident_type','')}\n"
        f"Description: {obs.get('call_description','')}\n"
        f"Location: {obs.get('location','')}\n"
        f"Severity: {obs.get('severity','')}\n"
        f"Caller: {obs.get('caller_info','')}\n\n"
        f"Available resources:\n"
        f"  Ambulances: {obs.get('available_ambulances',0)}\n"
        f"  Police:     {obs.get('available_police',0)}\n"
        f"  Fire:       {obs.get('available_fire',0)}\n\n"
        f"Hospital A — {obs.get('hospital_a_name','')}: "
        f"ICU={obs.get('hospital_a_icu_beds',0)} "
        f"General={obs.get('hospital_a_general_beds',0)} "
        f"Trauma={obs.get('hospital_a_trauma_capable',False)} "
        f"({obs.get('hospital_a_distance_minutes',0)} min)\n"
        f"Hospital B — {obs.get('hospital_b_name','')}: "
        f"ICU={obs.get('hospital_b_icu_beds',0)} "
        f"General={obs.get('hospital_b_general_beds',0)} "
        f"Trauma={obs.get('hospital_b_trauma_capable',False)} "
        f"({obs.get('hospital_b_distance_minutes',0)} min)\n"
        f"District reserve units: {obs.get('district_reserve_units',0)}\n"
    )
    return [
        {"role": "system",  "content": SYSTEM_PROMPT},
        {"role": "user",    "content": user_msg},
    ]


# ── Dataset builder ───────────────────────────────────────────────────────────
def make_dataset(n_episodes: int):
    from datasets import Dataset
    rows = []
    for _ in range(n_episodes):
        task = sample_task()
        _sid, data = post_reset(ENV_URL, json_body={"task": task})
        obs = data["observation"]
        rows.append({"prompt": obs_to_prompt(obs, task=task)})
    return Dataset.from_list(rows)

print(f"[..] Building dataset ({args.episodes} episodes)...")
dataset = make_dataset(args.episodes)
print(f"[OK] Dataset ready: {len(dataset)} rows")


# ── Reward function ───────────────────────────────────────────────────────────
def _extract_task_from_prompt(prompt) -> str:
    # `prompt` is expected to be a chat list of {"role": ..., "content": ...}.
    # We stash the tier in the user content as "TASK: <tier>" for mixed runs.
    try:
        if isinstance(prompt, list):
            for m in prompt:
                if isinstance(m, dict) and m.get("role") == "user":
                    content = m.get("content", "")
                    mm = re.search(r"(?mi)^\s*TASK\s*:\s*(easy|medium|hard|crisis)\s*$", content)
                    if mm:
                        return mm.group(1).lower()
    except Exception:
        pass
    return args.task if args.task != "mixed" else "easy"

def get_reward(completions, prompts, **kwargs) -> list[float]:
    rewards = []
    for i, completion in enumerate(completions):
        text = completion[0]["content"] if isinstance(completion, list) else completion
        try:
            task = _extract_task_from_prompt(prompts[i]) if i < len(prompts) else sample_task()
            sid, _ = post_reset(ENV_URL, json_body={"task": task})
            action = parse_action(text)
            if "backup_requested" not in action:
                action["backup_requested"] = False
            obs = post_step(ENV_URL, sid, action, timeout=5)
            rewards.append(float(obs.get("reward", -0.5)))
        except Exception as e:
            print(f"  [warn] reward fn error: {e}")
            rewards.append(-1.0)
    return rewards


# ── Baseline evaluation (random agent) ───────────────────────────────────────
def random_action() -> dict:
    a = {
        "ambulance_units":   random.randint(0, 3),
        "police_units":      random.randint(0, 3),
        "fire_units":        random.randint(0, 3),
        "priority_level":    random.randint(1, 4),
        "coordination_level": random.choice(["none", "mutual_aid"]),
        "hospital_choice":   "nearest",
        "ambulance_staging": "dispatch",
    }
    # Server-side action schema requires at least one dispatched unit.
    if a["ambulance_units"] == 0 and a["police_units"] == 0 and a["fire_units"] == 0:
        a[random.choice(["ambulance_units", "police_units", "fire_units"])] = 1
    return a

def evaluate_random(n=20) -> float:
    scores = []
    for _ in range(n):
        sid, _ = post_reset(ENV_URL, json_body={"task": sample_task()})
        total = 0.0
        for _ in range(10):
            a = random_action()
            a.setdefault("backup_requested", False)
            obs = post_step(ENV_URL, sid, a)
            total += obs.get("reward", 0.0)
            if obs.get("done", False):
                break
        scores.append(total)
    return sum(scores) / len(scores)

print("[..] Evaluating random baseline...")
baseline_score = evaluate_random()
print(f"[OK] Random agent mean episode score: {baseline_score:.4f}")


# ── Model + GRPO training ─────────────────────────────────────────────────────
print(f"[..] Loading model: {args.model}")
from unsloth import FastLanguageModel
from trl import GRPOTrainer, GRPOConfig

model, tokenizer = FastLanguageModel.from_pretrained(
    args.model, max_seq_length=1024, load_in_4bit=True)
model = FastLanguageModel.get_peft_model(
    model, r=16, target_modules=["q_proj", "v_proj"],
    lora_alpha=16, lora_dropout=0.0, bias="none")

training_args = GRPOConfig(
    output_dir="training/checkpoints",
    max_steps=args.steps,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=2,
    num_generations=4,
    max_completion_length=128,
    learning_rate=5e-6,
    logging_steps=10,
    save_steps=100,
    report_to="none",
)

trainer = GRPOTrainer(
    model=model,
    tokenizer=tokenizer,
    reward_funcs=[get_reward],
    args=training_args,
    train_dataset=dataset,
)

print(f"[..] Starting GRPO training ({args.steps} steps)...")
trainer.train()
print("[OK] Training complete")


# ── Reward curve ──────────────────────────────────────────────────────────────
logs = [l for l in trainer.state.log_history if "reward" in l]
if logs:
    steps   = [l["step"]   for l in logs]
    rewards = [l["reward"] for l in logs]
    plt.figure(figsize=(9, 4))
    plt.plot(steps, rewards, linewidth=1.5)
    plt.axhline(baseline_score, linestyle="--", color="gray", label=f"Random baseline ({baseline_score:.3f})")
    plt.xlabel("Training step")
    plt.ylabel("Mean reward")
    plt.title(f"Emergency Dispatch RL — reward curve ({args.task})")
    plt.legend()
    plt.tight_layout()
    out_path = os.path.join(os.path.dirname(__file__), "reward_curve.png")
    plt.savefig(out_path, dpi=150)
    print(f"[OK] Reward curve saved to {out_path}")
else:
    print("[warn] No reward logs found — curve not saved")

# ── Loss curve ────────────────────────────────────────────────────────────────
loss_logs = [l for l in trainer.state.log_history if "loss" in l]
if loss_logs:
    loss_steps = [l["step"] for l in loss_logs]
    losses = [l["loss"] for l in loss_logs]
    plt.figure(figsize=(9, 4))
    plt.plot(loss_steps, losses, linewidth=1.5, color="#c0392b")
    plt.xlabel("Training step")
    plt.ylabel("Loss")
    plt.title(f"Emergency Dispatch RL — loss curve ({args.task})")
    plt.tight_layout()
    loss_path = os.path.join(os.path.dirname(__file__), "loss_curve.png")
    plt.savefig(loss_path, dpi=150)
    print(f"[OK] Loss curve saved to {loss_path}")
else:
    print("[warn] No loss logs found — curve not saved")


# ── After-training evaluation ─────────────────────────────────────────────────
print("[..] Evaluating trained model...")
FastLanguageModel.for_inference(model)

def model_action(obs_dict: dict) -> dict:
    messages = obs_to_prompt(obs_dict)
    inputs = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
    ).to(model.device)
    import torch
    with torch.no_grad():
        out = model.generate(inputs, max_new_tokens=128, temperature=0.1, do_sample=True)
    text = tokenizer.decode(out[0][inputs.shape[1]:], skip_special_tokens=True)
    return parse_action(text)

def evaluate_model(n=20) -> float:
    scores = []
    for _ in range(n):
        _sid, data = post_reset(ENV_URL, json_body={"task": sample_task()})
        obs = data["observation"]
        total = 0.0
        for _ in range(10):
            action = model_action(obs)
            action.setdefault("backup_requested", False)
            data = post_step(ENV_URL, _sid, action)
            total += data.get("reward", 0.0)
            obs = data.get("observation", obs)
            if data.get("done", False):
                break
        scores.append(total)
    return sum(scores) / len(scores)

trained_score = evaluate_model()
print(f"[OK] Trained model mean episode score: {trained_score:.4f}")
print(f"\n{'='*50}")
print(f"  BEFORE training (random): {baseline_score:.4f}")
print(f"  AFTER  training (GRPO):   {trained_score:.4f}")
print(f"  Improvement:              {trained_score - baseline_score:+.4f}")
print(f"{'='*50}\n")

# Save comparison JSON for the blog
comparison = {
    "task": args.task,
    "steps": args.steps,
    "baseline_score": round(baseline_score, 4),
    "trained_score":  round(trained_score, 4),
    "improvement":    round(trained_score - baseline_score, 4),
}
comp_path = os.path.join(os.path.dirname(__file__), "comparison.json")
with open(comp_path, "w") as f:
    json.dump(comparison, f, indent=2)
print(f"[OK] Comparison saved to {comp_path}")
