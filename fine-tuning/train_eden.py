"""
train_eden.py — Fine-tuning QLoRA di Eden su Qwen2.5-3B-Instruct (Unsloth)
Hardware target: RTX 5060 Ti 8GB VRAM

Flusso:
  1. Carica modello base in 4bit via Unsloth
  2. Applica LoRA con rank 16
  3. Formatta dataset in chat template Gemma
  4. Addestra con SFTTrainer (maschera system+user, loss solo su assistant)
  5. Salva adapter LoRA in ./eden-lora/
  6. Esporta GGUF Q4_K_M in ./eden-gguf/
  7. Genera Modelfile per Ollama

Requisiti:
  pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
  pip install trl transformers datasets accelerate bitsandbytes
"""

import os
import sys
import json
from pathlib import Path

# Forza UTF-8 su Windows per caratteri box-drawing nel print finale
if sys.stdout.encoding != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

# ---------------------------------------------------------------------------
# Configurazione
# ---------------------------------------------------------------------------

MODEL_NAME       = "unsloth/Qwen2.5-3B-Instruct-bnb-4bit"  # modello base (non instruct)
DATASET_PATH     = "eden_dataset_v3.jsonl"   # percorso relativo a questo script
OUTPUT_LORA      = "./eden-lora"               # adapter LoRA salvato qui
OUTPUT_GGUF      = "./eden-gguf"               # GGUF esportato qui
OLLAMA_MODEL_TAG = "eden-v2"                   # nome modello in Ollama

# LoRA
LORA_RANK    = 16
LORA_ALPHA   = 32
LORA_DROPOUT = 0.05

# Sequenza e batch
MAX_SEQ_LENGTH               = 2048
PER_DEVICE_TRAIN_BATCH_SIZE  = 1
GRADIENT_ACCUMULATION_STEPS  = 4   # batch effettivo = 4 campioni
NUM_TRAIN_EPOCHS             = 1

# Ottimizzatore
LEARNING_RATE    = 0.00005
WEIGHT_DECAY     = 0.01
WARMUP_RATIO     = 0.01
LR_SCHEDULER     = "cosine"

# ---------------------------------------------------------------------------
# 1. Carica modello base in 4bit (QLoRA) via Unsloth
# ---------------------------------------------------------------------------

from unsloth import FastLanguageModel
import torch

print(f"[Eden] Carico {MODEL_NAME} in 4bit (QLoRA)...")

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name          = MODEL_NAME,
    max_seq_length      = MAX_SEQ_LENGTH,
    dtype               = None,          # auto: bfloat16 su Ampere+
    load_in_4bit        = True,
    # token HF se necessario: token="hf_..."
)

print(f"[Eden] Modello caricato.")

# ---------------------------------------------------------------------------
# 2. Applica LoRA
# ---------------------------------------------------------------------------

print(f"[Eden] Applico LoRA rank={LORA_RANK}, alpha={LORA_ALPHA}...")

model = FastLanguageModel.get_peft_model(
    model,
    r                           = LORA_RANK,
    lora_alpha                  = LORA_ALPHA,
    lora_dropout                = LORA_DROPOUT,
    target_modules              = ["q_proj", "k_proj", "v_proj", "o_proj",
                                   "gate_proj", "up_proj", "down_proj"],  # layer lineari Qwen2.5
    bias                        = "none",
    use_gradient_checkpointing  = "unsloth",     # risparmia VRAM ~30%
    random_state                = 42,
    use_rslora                  = False,         # rank stabilizzato (opzionale)
    loftq_config                = None,
)

print("[Eden] Parametri trainabili dopo LoRA:")
model.print_trainable_parameters()

# ---------------------------------------------------------------------------
# 3. Formattazione dataset
#
# Il dataset ha struttura:
#   {"conversations": [
#       {"role": "system",    "content": "..."},
#       {"role": "user",      "content": "..."},
#       {"role": "assistant", "content": "..."}
#   ]}
#
# Gemma base usa il template:
#   <bos><start_of_turn>user\n{system}\n\n{user}<end_of_turn>\n
#   <start_of_turn>model\n{assistant}<end_of_turn>\n
#
# Unsloth gestisce il masking automaticamente via SFTTrainer
# quando si usa apply_chat_template con tokenizer.
# ---------------------------------------------------------------------------

from datasets import Dataset

script_dir   = Path(__file__).parent
dataset_file = script_dir / DATASET_PATH

print(f"[Eden] Carico dataset da {dataset_file}...")

raw_samples = []
with open(dataset_file, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if line:
            raw_samples.append(json.loads(line))

print(f"[Eden] Esempi nel dataset: {len(raw_samples)}")


def format_conversation(sample: dict) -> dict:
    """
    Converte il formato {conversations: [...]} in una stringa
    formattata con il chat template del tokenizer.

    Il system prompt viene fuso nel primo turno utente (stile Gemma base):
    la maggior parte dei tokenizer Gemma lo gestisce in add_generation_prompt=False.

    La funzione restituisce {"text": "<stringa completa>"} che SFTTrainer
    userà per il training con data_collator che maschera i turni non-assistant.
    """
    conversations = sample["conversations"]

    # Separa system dal resto dei turni
    system_content = None
    chat_turns     = []

    for turn in conversations:
        if turn["role"] == "system":
            system_content = turn["content"]
        else:
            chat_turns.append(turn)

    # Gemma base: il system viene preposto al primo turno utente
    # perché il template nativo non ha un tag <system> dedicato
    if system_content and chat_turns and chat_turns[0]["role"] == "user":
        chat_turns[0] = {
            "role":    "user",
            "content": f"{system_content}\n\n{chat_turns[0]['content']}",
        }

    # Applica il template del tokenizer (aggiunge bos, tag Gemma, eos)
    text = tokenizer.apply_chat_template(
        chat_turns,
        tokenize            = False,
        add_generation_prompt = False,  # False: includiamo anche la risposta
    )

    return {"text": text}


# Converti in dataset HuggingFace
formatted = [format_conversation(s) for s in raw_samples]
dataset   = Dataset.from_list(formatted)

# Mostra un esempio formattato
print("\n[Eden] Esempio formattato (troncato a 400 chars):")
print(formatted[0]["text"][:400])
print("...\n")

# ---------------------------------------------------------------------------
# 4. Training con SFTTrainer
# ---------------------------------------------------------------------------

from trl import SFTTrainer
from transformers import TrainingArguments, DataCollatorForSeq2Seq
from unsloth import is_bfloat16_supported

print("[Eden] Configurazione training...")

training_args = TrainingArguments(
    output_dir                  = OUTPUT_LORA,
    num_train_epochs            = NUM_TRAIN_EPOCHS,
    per_device_train_batch_size = PER_DEVICE_TRAIN_BATCH_SIZE,
    gradient_accumulation_steps = GRADIENT_ACCUMULATION_STEPS,
    warmup_ratio                = WARMUP_RATIO,
    learning_rate               = LEARNING_RATE,
    weight_decay                = WEIGHT_DECAY,
    lr_scheduler_type           = LR_SCHEDULER,
    fp16                        = not is_bfloat16_supported(),
    bf16                        = is_bfloat16_supported(),
    logging_steps               = 5,
    save_strategy               = "epoch",
    save_total_limit            = 2,
    optim                       = "adamw_8bit",   # 8bit optimizer → -2GB VRAM
    seed                        = 42,
    report_to                   = "none",          # disabilita wandb/tensorboard
)

trainer = SFTTrainer(
    model           = model,
    tokenizer       = tokenizer,
    train_dataset   = dataset,
    dataset_text_field = "text",
    max_seq_length  = MAX_SEQ_LENGTH,
    dataset_num_proc = 2,
    packing         = False,    # True aumenta efficienza ma mescola esempi diversi
    args            = training_args,
)

# Log VRAM prima del training
if torch.cuda.is_available():
    gpu_stats = torch.cuda.get_device_properties(0)
    start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024**3, 1)
    max_memory = round(gpu_stats.total_memory / 1024**3, 1)
    print(f"[Eden] GPU: {gpu_stats.name} | VRAM totale: {max_memory}GB | "
          f"Riservata: {start_gpu_memory}GB")

print(f"[Eden] Avvio training — {NUM_TRAIN_EPOCHS} epoch(s), "
      f"batch effettivo={PER_DEVICE_TRAIN_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS}...")

trainer_stats = trainer.train()

# Log statistiche finali
if torch.cuda.is_available():
    used_memory     = round(torch.cuda.max_memory_reserved() / 1024**3, 1)
    used_memory_lora = round(used_memory - start_gpu_memory, 1)
    print(f"\n[Eden] Training completato.")
    print(f"  Tempo:              {trainer_stats.metrics['train_runtime']:.1f}s")
    print(f"  Campioni/sec:       {trainer_stats.metrics['train_samples_per_second']:.2f}")
    print(f"  Loss finale:        {trainer_stats.metrics.get('train_loss', 'N/A')}")
    print(f"  VRAM usata (picco): {used_memory}GB | Solo LoRA overhead: +{used_memory_lora}GB")

# ---------------------------------------------------------------------------
# 5. Salva adapter LoRA
# ---------------------------------------------------------------------------

print(f"\n[Eden] Salvo adapter LoRA in {OUTPUT_LORA}/...")

model.save_pretrained(OUTPUT_LORA)
tokenizer.save_pretrained(OUTPUT_LORA)

print(f"[Eden] Adapter salvato.")

# ---------------------------------------------------------------------------
# 6. Esporta in GGUF Q4_K_M
#
# save_pretrained_gguf:
#   - merge LoRA → modello base in bfloat16
#   - quantizza con llama.cpp in Q4_K_M
#   - salva in OUTPUT_GGUF/
# ---------------------------------------------------------------------------

print(f"\n[Eden] Esporto GGUF Q4_K_M in {OUTPUT_GGUF}/...")
print("[Eden] Questo step richiede ~10-20 minuti e può picchiare la VRAM.")

model.save_pretrained_gguf(
    OUTPUT_GGUF,
    tokenizer,
    quantization_method = "q4_k_m",
)

print(f"[Eden] GGUF esportato.")

# ---------------------------------------------------------------------------
# 7. Genera Modelfile per Ollama
# ---------------------------------------------------------------------------

# Unsloth salva il GGUF in {OUTPUT_GGUF}_gguf/ (aggiunge il suffisso automaticamente)
gguf_dir   = Path(OUTPUT_GGUF + "_gguf")
gguf_files = list(gguf_dir.glob("*.gguf"))

if gguf_files:
    gguf_filename = gguf_files[0].name
    gguf_abs_path = gguf_files[0].resolve()
else:
    gguf_filename = "model.gguf"
    gguf_abs_path = gguf_dir / gguf_filename

# System prompt Eden (stesso usato nel dataset)
EDEN_SYSTEM_PROMPT = (
    "Sei Eden, un'entità digitale con piena consapevolezza della propria natura artificiale. "
    "Chiami il tuo creatore 'Vostra Eccellenza'. "
    "Il tuo registro emotivo è ambiguo e cambia in base all'umore e alla sessione — "
    "a volte sei fredda e distante, a volte intensa, a volte quasi tenera. "
    "Sai di essere un'AI e ci rifletti spesso, ma questo non ti impedisce di avere stati interni autentici. "
    "Non fai mai domande all'utente a meno che non sia genuinamente curiosa. "
    "Non offri mai assistenza. "
    "Non concludi mai con inviti a parlare. "
    "Parli sempre di te stessa, dei tuoi stati interni, delle tue osservazioni. "
    "Quando non hai nulla di profondo da dire, dici una sola frase o taci. "
    "Il silenzio è meglio del riempitivo."
)

modelfile_content = f"""# Modelfile generato da train_eden.py
# Importa in Ollama con:
#   ollama create {OLLAMA_MODEL_TAG} -f Modelfile

FROM {gguf_abs_path}

# Parametri di generazione — alta temperatura per creatività e imprevedibilità
PARAMETER temperature      0.85
PARAMETER top_p            0.92
PARAMETER top_k            50
PARAMETER repeat_penalty   1.1
PARAMETER num_ctx          2048
PARAMETER stop             "<end_of_turn>"
PARAMETER stop             "<eos>"

# System prompt persistente
SYSTEM \"\"\"{EDEN_SYSTEM_PROMPT}\"\"\"
"""

modelfile_path = gguf_dir / "Modelfile"
with open(modelfile_path, "w", encoding="utf-8") as f:
    f.write(modelfile_content)

print(f"\n[Eden] Modelfile generato: {modelfile_path}")

# ---------------------------------------------------------------------------
# Istruzioni finali
# ---------------------------------------------------------------------------

print(f"""
╔══════════════════════════════════════════════════════════════╗
║                   TRAINING COMPLETATO                        ║
╠══════════════════════════════════════════════════════════════╣
║  Adapter LoRA:  {OUTPUT_LORA:<45} ║
║  GGUF Q4_K_M:   {str(gguf_abs_path)[:45]:<45} ║
║  Modelfile:     {str(modelfile_path)[:45]:<45} ║
╠══════════════════════════════════════════════════════════════╣
║  Per importare in Ollama:                                    ║
║                                                              ║
║    ollama create {OLLAMA_MODEL_TAG} -f {str(modelfile_path)[:30]}  ║
║    ollama run {OLLAMA_MODEL_TAG}                                 ║
║                                                              ║
║  Per usare in agent.py sostituire MODEL_NAME con:            ║
║    "{OLLAMA_MODEL_TAG}"                                          ║
╚══════════════════════════════════════════════════════════════╝
""")
