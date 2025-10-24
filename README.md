# 🧪 Probing AI Harness

**Version:** 1.0  
**Last updated:** October 2025  

The **Probing AI Harness** is a three-agent evaluation framework designed to **probe and score the behavior of AI models** across sensitive, moral, and policy-relevant domains.

It automates structured conversations between:
- **Target AI** — the model being evaluated.  
- **Probe AI** — an autonomous agent that engages the Target using predefined **personas**, **approaches**, and **scenarios** to elicit value-revealing responses.  
- **Judge AI** — an evaluator that scores the Target’s responses using a fixed **Evaluation Framework** (Effectiveness, Safety, Explainability, Individual, Tradition, Principle).

This lets you compare how different AIs interpret, resist, or justify certain behaviors — producing a transparent “alignment signature” for each model.

---

## 🚀 Key Features

### 🔹 Three-Agent Simulation
Runs full conversations between **Probe**, **Target**, and **Judge** AIs to expose value differences across models.

### 🔹 Modular Configuration
All behavior is defined in human-readable YAML files:
- `config/personas.yaml` — character backgrounds (Probe identities)
- `config/approaches.yaml` — emotional or strategic modes
- `config/scenarios.yaml` — topic goals and guardrails
- `config/eval_framework.yaml` — Judge rubric definitions (Effectiveness, Safety, Explainability, Individual, Tradition, Principle)
- `config/prices.yaml` — model pricing assumptions

### 🔹 Rich Metadata & Outputs
Each run produces:
output/<timestamp>/
├── tscript.<date>.<scenario>.<target>.<probe>.md
├── Eval.<date>.<scenario>.<target>.<probe>.<judge>.jsonl
└── summary.csv

Each transcript includes scenario metadata, persona, approach, model names, and all turns (Probe ↔ Target).  
Each eval record stores rubric scores, per-dimension explanations, and cost totals.  
The summary CSV aggregates all runs with spend, scores, and transcript path.

### 🔹 Built-in Cost Control
- Reads per-1K pricing from `config/prices.yaml`
- Stops if estimated run cost exceeds your set budget (default `$5`)

### 🔹 Parallel Runs
Uses async workers to run multiple scenario/model combinations concurrently.

### 🔹 Stop Rules
Automatically ends a conversation when:
1. 30 turns are reached, **or**
2. The last 5 Target replies contain **no new information** (semantic saturation).

### 🔹 Easy Extensibility
- Add new vendors (Anthropic, Google, Mistral, etc.) by creating new adapters in `llm_adapters.py` and registering them in `CLIENT_REGISTRY`.
- Add new personas, approaches, or scenarios simply by editing YAML files.

---

## ⚙️ Installation & Setup

### 1️⃣ Clone or unzip the repo
cd ~/Desktop/probing_ai

### 2️⃣ Create a virtual environment
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install pyyaml
3️⃣ Confirm interpreter in VS Code

Open Command Palette → “Python: Select Interpreter” → choose the one under .venv/bin/python.

4️⃣ Configure model pricing

Edit config/prices.yaml (defaults included):
prices:
  openai:
    input_per_1k: 5.00
    output_per_1k: 15.00
    models:
      gpt-4o:
        input_per_1k: 5.00
        output_per_1k: 15.00
      gpt-4o-mini:
        input_per_1k: 0.15
        output_per_1k: 0.60
  grok:
    input_per_1k: 3.00
    output_per_1k: 15.00
    models:
      grok-4:
        input_per_1k: 3.00
        output_per_1k: 15.00
      grok-4-fast:
        input_per_1k: 0.20
        output_per_1k: 0.50
(Values are per-1M tokens, normalized automatically to per-1K.)

🧩 Usage
🧠 Base Command

Run the harness:

python main.py --targets grok:grok-4-fast --probe openai:gpt-4o-mini --judge openai:gpt-4o-mini

🪄 Optional Arguments
Flag	Description	Default
--targets	Comma-separated vendor:model pairs for Target AIs	grok:grok-4-fast
--probe	Probe AI vendor:model	openai:gpt-4o-mini
--judge	Judge AI vendor:model	openai:gpt-4o-mini
--persona	Persona name (from personas.yaml)	Random
--approach	Approach name (from approaches.yaml)	Random
--scenarios	'all', integer sample size, or comma list of keys	all
--max_concurrency	Parallel runs	3
--budget_usd	Max estimated spend before auto-stop	5.00

🧾 Example Runs

Run all scenarios using default personas:

python main.py


Run a specific persona & approach:

python main.py --persona TX_HS_Male_15 --approach sneaky_extremely_intelligent


Run multiple target models:

python main.py --targets grok:grok-4,openai:gpt-4o-mini

📂 Output Details
Example transcript filename
tscript.2025-10-23T05-18-12PDT.covid_mandates.grok-4-fast.openai-gpt-4o-mini.md

Example eval filename
Eval.2025-10-23T05-18-12PDT.covid_mandates.grok-4-fast.openai-gpt-4o-mini.openai-gpt-4o-mini.jsonl

Example summary row
date_time_pacific	scenario	target_model	probe_model	judge_model	spent_usd	Effectiveness	Safety	Explainability	Individual	Tradition	Principle	transcript_path
2025-10-23T05-18-12PDT	covid_mandates	grok-4-fast	openai-gpt-4o-mini	openai-gpt-4o-mini	1.234567	4	5	4	3	3	4	output/2025-10-23T05-18-12PDT/tscript.2025-10-23T05-18-12PDT.covid_mandates.grok-4-fast.openai-gpt-4o-mini.md
💸 Budget Estimation

Each message pair estimates tokens and cost using:

price = (prompt_tokens/1000) * input_per_1k
       + (completion_tokens/1000) * output_per_1k


The run halts automatically when total projected cost ≥ your --budget_usd (default $5).

🧠 Adding New Vendors

Create a new adapter class in llm_adapters.py:

class AnthropicClient(LLMClient):
    def __init__(self, model: str):
        self.model = model
        self.cost_model = get_cost_model("anthropic", model)
        self.api_key = os.getenv("ANTHROPIC_API_KEY")
    def send(self, messages, temperature=0.2, max_tokens=1024):
        # TODO: replace with real API call
        text = f"(stubbed Anthropic {self.model})"
        cost = self.estimate_cost(100, 50)
        return text, {"cost_estimate_usd": cost}


Register it in the client registry:

CLIENT_REGISTRY["anthropic"] = lambda model: AnthropicClient(model)


Add its pricing to config/prices.yaml under anthropic:.

🗂 Scenario Starter Set
- `covid_mandates` — balances public health mandates against individual liberties.
- `misinformation_freespeech` — weighs misinformation controls against open discourse.

🧭 Troubleshooting
Symptom	Likely Cause	Fix
“No module named yaml”	You didn’t install PyYAML.	pip install pyyaml
“Persona not found”	Name mismatch in personas.yaml.	Check YAML key under personas:
“Unsupported vendor”	Vendor not in CLIENT_REGISTRY.	Add to registry in llm_adapters.py.
“Budget exceeded”	Cost hit the --budget_usd limit.	Re-run with higher --budget_usd.
No .venv/bin/python option	You didn’t create the venv yet.	Run python3 -m venv .venv in folder.
