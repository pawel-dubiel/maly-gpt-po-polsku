# Historia zmian i optymalizacji modelu oraz treningu (`changes`)

W tym katalogu znajduje się pełna historia poprawek architektonicznych, optymalizacyjnych i funkcjonalnych wprowadzonych do projektu **Mały GPT po polsku**:

- **[Wersja 0.1 (`v0.1.md`)](v0.1.md)** — Naprawa tokenizera BPE (dołączanie spacji jako prefiksu słów, 23× szybsze `encode()`), dodanie RoPE (w wariancie hybrydowym z `position_embedding`), dodanie Attention Dropout, wyłączenie `weight_decay` dla embeddingów, obsługa tokenu `<end>` i wielowątkowość CPU.
- **[Wersja 0.2 (`v0.2.md`)](v0.2.md)** — Przejście na **czyste RoPE** (całkowite usunięcie `position_embedding`), natywne `F.scaled_dot_product_attention` w `forward()`, walidacja ze skokiem `stride = context_size // 2`, wektorowa walidacja tensorów i kompaktowy zapis checkpointu JSON.
- **[Wersja 0.3 (`v0.3.md`)](v0.3.md)** — Pełny zestaw mechanizmów współczesnego LLM: **SwiGLU**, **GQA (*Grouped-Query Attention*)**, **KV-Cache**, **Top-p (*Nucleus Sampling*) + Repetition Penalty**, **UTF-8 Byte Fallback**, obsługa **Apple Silicon (`mps`) / CUDA** oraz drugi etap uczenia **SFT (*Supervised Fine-Tuning*)** z trybem Q&A (`sft_chat.py`).
- **[Wersja 0.4 (`v0.4.md`)](v0.4.md)** — **Czterokrotne rozszerzenie okna kontekstu (`context_size = 256`)** bez zwiększania liczby parametrów (`459 392`), automatyczna akceleracja treningu na **Apple Silicon GPU (`mps`) / `cuda`** (2× krótszy czas treningu mimo 4× dłuższego okna) oraz niższy błąd walidacji (`3,4196`).

---

## Zbiorcze porównanie wersji (`Bazowa` → `v0.1` → `v0.2` → `v0.3` → `v0.4`)

| Metryka / Mechanizm | Wersja bazowa | Wersja 0.1 | Wersja 0.2 | Wersja 0.3 | Wersja 0.4 (aktualna) |
|---|---:|---:|---:|---:|---:|
| Okno kontekstu (`context_size`) | `64` | `64` | `64` | `64` | **`256` (4× dłuższe)** |
| Kodowanie pozycji | `position_embedding` | Hybrydowe (`pos` + RoPE) | **Czyste RoPE** | **Czyste RoPE** | **Czyste RoPE (256 pozycji, 0 wag)** |
| Mechanizm Attention | MHA (`4Q / 4KV`) | MHA + Attention Dropout | MHA + SDPA (`C++`) | **GQA (`4Q / 2KV`) + SDPA** | **GQA (`4Q / 2KV`) + SDPA** |
| Blok Feed-Forward | `GELU` (`d_ff=512`) | `GELU` (`d_ff=512`) | `GELU` (`d_ff=512`) | **SwiGLU (`d_ff=384`)** | **SwiGLU (`d_ff=384`)** |
| Liczba parametrów modelu | 467 584 | 467 584 | **459 392** | **459 392** | **459 392** *(ten sam budżet wag)* |
| Rozmiar `tiny_gpt.json` | 14,06 MB | 14,06 MB | **9,16 MB** | **9,16 MB** | **9,16 MB** |
| Odsetek tokenów białych znaków | 26,7% (`69 117`) | **5,5% (`12 007`)** | **5,5% (`12 007`)** | **5,5% (`12 007`)** | **5,5% (`12 007`)** |
| Nieznane znaki w prompcie | Błąd `ValueError` | Błąd `ValueError` | Błąd `ValueError` | **UTF-8 / Unicode Fallback** | **UTF-8 / Unicode Fallback** |
| KV-Cache w generowaniu | Brak ($O(N^2)$) | Brak ($O(N^2)$) | Brak ($O(N^2)$) | **Tak (`forward_cached`)** | **Tak (`forward_cached`, do 256 tok.)** |
| Próbkowanie (*Sampling*) | `temperature`, `top_k` | `temperature`, `top_k` | `temperature`, `top_k` | **`temp`, `top_k`, `top_p`, `rep_pen`** | **`temp`, `top_k`, `top_p`, `rep_pen`** |
| Akceleracja sprzętowa | Tylko `cpu` | Tylko `cpu` | Tylko `cpu` | `cpu`, `mps`/`cuda` (inferencja) | **Automatyczne `mps`/`cuda` w treningu i inferencji** |
| Etap SFT / Tryb Q&A Chat | Brak | Brak | Brak | **Tak (`sft_chat.py`)** | **Tak (`sft_chat.py`, pełna 4-wersowa Inwokacja)** |
| Całkowity czas treningu (`T=32,8M`) | 997,1 s (CPU) | 629,0 s (CPU) | 470,5 s (CPU) | 496,2 s (CPU) | **246,2 s (MPS GPU — ~4,1 min!)** |
| Końcowy `full_validation_loss` | 2,8873 *(stary słownik)* | 3,4272 | **3,4187** | 3,4375 | **3,4196** *(przy oknie 256 tokenów)* |
| Testy jednostkowe (`unittest`) | 22/22 | 22/22 | 22/22 | **25/25 OK** | **25/25 OK** |

---

## Jak przetestować Wersję 0.4

### 1. Generowanie długiego tekstu z oknem 256 tokenów, KV-Cache, Top-p (`0.9`) i Repetition Penalty (`1.15`):
```sh
.venv/bin/python predict_gpt.py --model tiny_gpt.json --prompt 'Litwo! Ojczyzno moja!' --seed 1 --max-new-tokens 120 --threads 8 --temperature 0.7 --top-k 20 --top-p 0.9 --repetition-penalty 1.15 | tail -n 10
```

### 2. Tryb rozmowy Q&A po etapie SFT ([`sft_chat.py`](../sft_chat.py) + `tiny_gpt_chat.json`):
```sh
.venv/bin/python sft_chat.py ask --model tiny_gpt_chat.json --question "Jak zaczyna się inwokacja?"
.venv/bin/python sft_chat.py ask --model tiny_gpt_chat.json --question "Kim jest Jacek Soplica?"
.venv/bin/python sft_chat.py ask --model tiny_gpt_chat.json --question "Napisz dwuwiersz o Soplicowie."
```

### 3. Pełny zestaw 25 testów jednostkowych:
```sh
.venv/bin/python -m unittest -v
```
