# Historia zmian i optymalizacji modelu oraz treningu (`changes`)

W tym katalogu znajduje się pełna historia poprawek architektonicznych, optymalizacyjnych i funkcjonalnych wprowadzonych do projektu **Mały GPT po polsku**:

- **[Wersja 0.1 (`v0.1.md`)](v0.1.md)** — Naprawa tokenizera BPE (dołączanie spacji jako prefiksu słów, 23× szybsze `encode()`), dodanie RoPE (w wariancie hybrydowym z `position_embedding`), dodanie Attention Dropout, wyłączenie `weight_decay` dla embeddingów, obsługa tokenu `<end>` i wielowątkowość CPU.
- **[Wersja 0.2 (`v0.2.md`)](v0.2.md)** — Przejście na **czyste RoPE** (całkowite usunięcie `position_embedding`), natywne `F.scaled_dot_product_attention` w `forward()`, walidacja ze skokiem `stride = context_size // 2`, wektorowa walidacja tensorów i kompaktowy zapis checkpointu JSON.
- **[Wersja 0.3 (`v0.3.md`)](v0.3.md)** — Pełny zestaw mechanizmów współczesnego LLM: **SwiGLU**, **GQA (*Grouped-Query Attention*)**, **KV-Cache**, **Top-p (*Nucleus Sampling*) + Repetition Penalty**, **UTF-8 Byte Fallback**, obsługa **Apple Silicon (`mps`) / CUDA** oraz drugi etap uczenia **SFT (*Supervised Fine-Tuning*)** z trybem Q&A (`sft_chat.py`).
- **[Wersja 0.4 (`v0.4.md`)](v0.4.md)** — **Czterokrotne rozszerzenie okna kontekstu (`context_size = 256`)** bez zwiększania liczby parametrów (`459 392`), automatyczna akceleracja treningu na **Apple Silicon GPU (`mps`) / `cuda`** (2× krótszy czas treningu mimo 4× dłuższego okna) oraz niższy błąd walidacji (`3,4196`).
- **[Wersja 0.5 (`v0.5.md`)](v0.5.md)** — **Próg 1 koherencji językowej i format `.safetensors`:** korpus **22,09 MB polskiej prozy (36 tomów z Wolnych Lektur)** ([`download_corpus.py`](../download_corpus.py)), słownik BPE **`vocab_size = 4 096`**, model **`4 196 608` (~4,20 mln) parametrów** (`4` warstwy, `d_model = 256`, `d_ff = 768`, `8` głów uwagi) oraz binarny zapis **`.safetensors`** (`16,15 MB` zamiast `86 MB` w JSON, 30× szybszy odczyt).

---

## Zbiorcze porównanie wersji (`Bazowa` → `v0.1` → `v0.2` → `v0.3` → `v0.4` → `v0.5`)

| Metryka / Mechanizm | Wersja bazowa | Wersja 0.2 | Wersja 0.3 | Wersja 0.4 | Wersja 0.5 (aktualna) |
|---|---:|---:|---:|---:|---:|
| Korpus treningowy | `0,45 MB` (1 poemat) | `0,45 MB` | `0,45 MB` | `0,45 MB` | **`22,09 MB` (36 tomów prozy)** |
| Słownik BPE (`vocab_size`) | `512` | `512` | `512` | `512` | **`4 096` (całe polskie słowa)** |
| Okno kontekstu (`context_size`) | `64` | `64` | `64` | **`256`** | **`256`** |
| Warstwy × Głowy × `d_model` | `2 × 4 × 128` | `2 × 4 × 128` | `2 × 4 × 128` (GQA) | `2 × 4 × 128` (GQA) | **`4 × 8 × 256` (GQA `8Q/4KV`)** |
| Blok Feed-Forward | `GELU` (`512`) | `GELU` (`512`) | **SwiGLU (`384`)** | **SwiGLU (`384`)** | **SwiGLU (`768`)** |
| Liczba parametrów modelu | 467 584 | 459 392 | 459 392 | 459 392 | **`4 196 608` (~4,20 mln)** |
| Format i rozmiar checkpointu | `.json` (14,1 MB) | `.json` (9,2 MB) | `.json` (9,2 MB) | `.json` (9,2 MB) | **`.safetensors` (`16,15 MB` vs `86 MB` JSON)** |
| KV-Cache + Top-p + Rep. Penalty | Brak | Brak | **Tak** | **Tak** | **Tak** |
| Czas treningu | 997,1 s (CPU) | 470,5 s (CPU) | 496,2 s (CPU) | 246,2 s (MPS) | **1 464,5 s (~24,4 min, MPS, 49M tok.)** |
| Końcowy `full_validation_loss` | 2,8873 *(stary BPE)* | 3,4187 *(V=512)* | 3,4375 *(V=512)* | 3,4196 *(V=512)* | **`4,3907` *(V=4096, top 1,9% słownika!)*** |
| Testy jednostkowe (`unittest`) | 22/22 | 22/22 | 25/25 | 25/25 | **26/26 OK** |

---

## Jak przetestować Wersję 0.5 (`.safetensors`)

### 1. Generowanie spójnej polskiej prozy (`tiny_gpt.safetensors`):
```sh
.venv/bin/python predict_gpt.py --model tiny_gpt.safetensors --device mps --prompt 'Wokulski wszedł do salonu i spojrzał na' --seed 1 --max-new-tokens 80 --threads 8 --temperature 0.7 --top-k 40 --top-p 0.9 --repetition-penalty 1.15 | tail -n 3
```

### 2. Tryb rozmowy Q&A po etapie SFT (`tiny_gpt_chat.safetensors`):
```sh
.venv/bin/python sft_chat.py ask --model tiny_gpt_chat.safetensors --question "Kim jest Stanisław Wokulski?"
.venv/bin/python sft_chat.py ask --model tiny_gpt_chat.safetensors --question "Kim jest profesor Rafał Wilczur?"
.venv/bin/python sft_chat.py ask --model tiny_gpt_chat.safetensors --question "Jak zaczyna się inwokacja?"
```

### 3. Pełny zestaw 26 testów jednostkowych:
```sh
.venv/bin/python -m unittest -v
```
