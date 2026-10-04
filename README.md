# Mały GPT po polsku (Wersja 0.4)

Edukacyjny, ale wyposażony we współczesne mechanizmy LLM transformer uczony od zera
na „Panu Tadeuszu”. Projekt obejmuje:
- **Tokenizację BPE** ze spacją jako prefiksem słowa (wzorzec GPT-2/LLaMA), szybkim słownikiem rang i obsługą dowolnych znaków UTF-8 (*byte/Unicode fallback*),
- **Nowoczesną architekturę Transformera (LLaMA 3 / Qwen / Mistral) z oknem `256` tokenów:** kodowanie pozycji względnych **RoPE** (0 uczonych wag pozycyjnych), grupowaną uwagę **GQA (*Grouped-Query Attention*)**, bramkowaną sieć **SwiGLU**, **RMSNorm** oraz współdzielenie wag wejścia/wyjścia (*weight tying*),
- **Silnik inferencji i treningu z akceleracją GPU (`mps` / `cuda`):** **KV-Cache**, próbkowanie **Top-p (*Nucleus Sampling*)**, **Top-k**, **Repetition Penalty** oraz automatyczny trening na Apple Silicon (`mps`) lub CUDA przy dłuższym kontekście,
- **Etap SFT (*Supervised Fine-Tuning*)** z maskowaniem straty na pytaniu użytkownika i trybem rozmowy Q&A ([`sft_chat.py`](sft_chat.py)).

Szczegółowa historia zmian (`Bazowa` → `v0.1` → `v0.2` → `v0.3` → `v0.4`) znajduje się w [changes/README.md](changes/README.md).
Praktyczny przewodnik: [dobór parametrów, przeuczenie i porównywanie treningów](#jak-dobierać-parametry-i-optymalizować-trening).

---

## Uruchomienie

Polecenia wykonuj w katalogu repozytorium. Każde jest jedną linią — możesz
wkleić je bez znaków kontynuacji `\`. Przykład sprawdzono z Pythonem 3.9 i PyTorch 2.8 na macOS (CPU oraz Apple Silicon `mps`).

### 1. Środowisko

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements-full.txt
```

### 2. Trening modelu bazowego (*Pre-training*, okno 256 tokenów)

Książka i wyuczony tokenizer są już w repozytorium. Wszystkie ustawienia zawiera
[train_config.json](train_config.json). Uruchom:

```sh
.venv/bin/python train_bpe.py --config train_config.json
```

Dla `context_size >= 128` skrypt automatycznie wykrywa kartę graficzną Apple Silicon (`mps`) lub NVIDIA (`cuda`), dzięki czemu trening **8 000 kroków** po **16 okien × 256 tokenów** (`32,77 mln` tokenów łącznie) trwa **~4 minuty**. Powstaną dwa pliki:

- `tiny_gpt.json` (`~9,16 MB`) — najlepsze wagi, architektura i tokenizer; wystarcza do predykcji oraz dalszego douczania SFT.
- `training_report.json` — ustawienia, hashe danych i modelu, przebieg walidacji, wybrany krok, końcowe wyniki i czas treningu.

Skrypt zaczyna od nowych wag i atomowo zastępuje plik wskazany w konfiguracji.
Wagi pozostają lokalne i są pomijane przez Git (`.gitignore`).

### 3. Predykcja i generowanie tekstu (z KV-Cache, Top-p i Repetition Penalty)

```sh
.venv/bin/python predict_gpt.py --model tiny_gpt.json --prompt 'Litwo! Ojczyzno moja!' --seed 1 --max-new-tokens 120 --threads 8 --temperature 0.7 --top-k 20 --top-p 0.9 --repetition-penalty 1.15 | tail -n 10
```

- `Most likely next token` — najbardziej prawdopodobny token przed ograniczeniem losowania.
- `Generated` — prompt i wylosowana kontynuacja; `\n` oznacza nową linię.
- `Stopped` — zakończenie przez limit `--max-new-tokens` albo wygenerowanie tokenu `<end>`.

Dostępne parametry próbkowania i urządzenia:
- `--temperature` — zaostrza (`< 1.0`) lub spłaszcza (`> 1.0`) rozkład prawdopodobieństw.
- `--top-k` — ogranicza losowanie do $K$ najbardziej prawdopodobnych tokenów (`--top-k 1` wybiera zachłannie najlepszy token).
- `--top-p` (domyślnie `1.0`, zalecane `0.9`) — *Nucleus Sampling*: dynamicznie odcina ogon rozkładu powyżej skumulowanego prawdopodobieństwa $p$.
- `--repetition-penalty` (domyślnie `1.0`, zalecane `1.15`) — kara za powtarzanie tokenów obecnych w oknie kontekstu.
- `--device` (`cpu`, `mps`, `cuda`, domyślnie `cpu`) — wybór urządzenia obliczeniowego.

Generowanie korzysta z **KV-Cache (`forward_cached`)**: cały prompt przechodzi przez sieć tylko raz (*prefill*), a w kolejnych krokach przeliczany jest wyłącznie **1 ostatnio wygenerowany token** aż do wypełnienia 256-tokenowego okna. Usuń `| tail -n 10`, aby zobaczyć ID tokenów, macierze attention z każdej głowy i bloku oraz prawdopodobieństwa wszystkich 512 tokenów. Znak spoza książki w prompcie nie powoduje błędu dzięki automatycznemu fallbackowi UTF-8/Unicode.

### 4. Douczanie instrukcyjne (SFT) i tryb rozmowy Q&A

Model bazowy (`tiny_gpt.json`) uczy się kontynuować poemat. Aby przekształcić go w model odpowiadający na pytania i wykonujący polecenia poetyckie, uruchom etap **Supervised Fine-Tuning (SFT)** ([`sft_chat.py`](sft_chat.py)):

```sh
.venv/bin/python sft_chat.py train --model tiny_gpt.json --output tiny_gpt_chat.json --steps 150
```

Skrypt doucza model w ~4 sekundy na parach `Pytanie: ...\nOdpowiedź: ...<end>`, nakładając **maskę straty (`ignore_index = -100`)** na tokeny pytania użytkownika (gradienty aktualizują wagi wyłącznie na odpowiedzi asystenta i końcowym znaczniku `<end>`).

Zadawanie pytań wyuczonemu modelowi konwersacyjnemu:

```sh
.venv/bin/python sft_chat.py ask --model tiny_gpt_chat.json --question "Jak zaczyna się inwokacja?"
.venv/bin/python sft_chat.py ask --model tiny_gpt_chat.json --question "Kim jest Jacek Soplica?"
.venv/bin/python sft_chat.py ask --model tiny_gpt_chat.json --question "Napisz dwuwiersz o Soplicowie."
```

Przykładowy wynik (pełne 4 wersy mieszczące się swobodnie w 256-tokenowym oknie):
```text
Pytanie: Jak zaczyna się inwokacja?
Odpowiedź: Litwo! Ojczyzno moja! ty jesteś jak zdrowie;
Ile cię trzeba cenić, ten tylko się dowie,
Kto cię stracił. Dziś piękność twą w całej ozdobie
Widzę i opisuję, bo tęsknię po tobie.
Stopped: <end>
```

---

## Wybrany model (Wersja 0.4)

| Element | Wartość |
|---|---:|
| Słownik BPE | 512 tokenów |
| Wektor tokenu (`d_model`) | 128 liczb |
| Kontekst (`context_size`) | **256 tokenów (~12–16 wersów poematu)** |
| Kodowanie pozycji | **Czyste RoPE** (*Rotary Position Embeddings*, 0 uczonych wag) |
| Bloki transformera (`n_layers`) | 2 |
| Głowy zapytań `Q` (`n_heads`) | 4 (po 32 liczby) |
| Głowy kluczy i wartości `K/V` (`n_kv_heads`, GQA) | 2 (po 32 liczby, współdzielone przez 4 głowy `Q`) |
| Projekcje uwagi w bloku | `Wq`, `Wo`: 128 × 128; `Wk`, `Wv`: 128 × 64 |
| Sieć Feed-Forward (**SwiGLU**) | `W_gate`: 128 × 384, `W1`: 128 × 384, `W2`: 384 × 128 |
| Dropout w treningu | 0,15 (w tym *attention dropout*) |
| Liczba parametrów | **459 392** |

[model.py](model.py) implementuje zarówno szybką ścieżkę treningową/ewaluacyjną opartą na natywnym kernelu C++ `F.scaled_dot_product_attention`, ścieżkę z buforem `forward_cached` (KV-Cache), jak i jawną ścieżkę `forward_with_attention` zwracającą macierze uwagi:

```text
ID → embedding tokenu → dropout

Powtórz dla każdego z dwóch bloków:
  X = RMSNorm(wektory)
  Q = X Wq (4 głowy × 32),  K = X Wk (2 głowy × 32),  V = X Wv (2 głowy × 32)
  Q, K = RoPE(Q, K)         # obrót wektorów kodujący względną odległość (i - j) na 256 pozycjach
  powielenie 2 głów K, V do 4 głów Q (Grouped-Query Attention)
  A = softmax(Q Kᵀ / √32 + maska przyszłości) → attention dropout
  połączenie głów A V → Wo → dropout → dodanie wejścia bloku (residual)
  Z = RMSNorm(wektory)
  SwiGLU(Z) = (SiLU(Z W_gate) ⊙ (Z W1)) W2 → dropout → dodanie wejścia (residual)

końcowy RMSNorm → mnożenie przez transponowaną macierz embeddingów (weight tying)
→ 512 logitów → softmax → prawdopodobieństwa następnego tokenu
```

---

## Dane i tokeny

Korpus zawiera wszystkie dwanaście ksiąg i epilog „Pana Tadeusza” Adama
Mickiewicza: **445 638 znaków** i **218 228 tokenów BPE** (wraz z końcowym znacznikiem `<end>`). Pochodzi z Wolnych Lektur; [źródło i prawa do tekstu](data/SOURCE.md) opisano osobno.

Pre-tokenizacja BPE (`SEGMENTS = re.compile(r" ?\w+| ?[^\w\s]+|\s+", re.UNICODE)`) dołącza spację poprzedzającą słowo jako prefiks segmentu. Dzięki temu częste słowa i przedrostki łączą się ze spacją w jeden token (np. `' mo'`, `' nie'`, `' się'`), co zmniejszyło udział osobnych tokenów białych znaków z **26,7%** do **5,5%** i skróciło cały strumień o ponad 40 tys. tokenów. Kodowanie wykorzystuje tablicę rang `self.ranks`, przetwarzając całą książkę w **0,21 s**.

Podgląd:

```sh
.venv/bin/python bpe_tokenizer.py inspect --tokenizer tokenizer_bpe.json --text 'Litwo! Ojczyzno moja!'
```

```text
Pieces: ['L', 'i', 't', 'wo', '!', ' ', 'O', 'j', 'czy', 'z', 'no', ' mo', 'ja', '!']
IDs: [28, 48, 59, 164, 3, 2, 31, 49, 146, 65, 173, 229, 259, 3]
Decoded: 'Litwo! Ojczyzno moja!'
```

Odtworzenie treningu tokenizera:

```sh
.venv/bin/python bpe_tokenizer.py train --data data/pan_tadeusz_full.txt --output tokenizer_bpe.json --vocab-size 512
```

Pierwsze **196 405 tokenów** (`196 149` okien po 256 tokenów) służy do treningu; ostatnie **21 823 tokeny** (`21 567` okien po 256 tokenów) do walidacji. Strumień dzielimy **przed** tworzeniem okien, więc żadne okno treningowe nie sięga do walidacji.

---

## Co dzieje się podczas treningu

- W każdym z **8 000 kroków** losowanych jest **16 okien po 256 tokenów** (`4 096` tokenów na krok, łącznie `32 768 000` tokenów).
- `AdamW` aktualizuje wagi z `weight_decay = 0,1` wyłącznie dla macierzy projekcji (`Wq, Wk, Wv, Wo, W_gate, W1, W2`), natomiast współczynniki `RMSNorm` oraz `token_embedding` mają **`weight_decay = 0,0`** (zapobiega to zanikaniu rzadkich tokenów i sztucznemu rozrostowi skali `norm_final`).
- Learning rate rośnie liniowo przez 200 kroków do `0,001`, a następnie maleje kosinusowo do `0,0001`.
- Po kroku 1 i co 500 kroków skrypt mierzy `train_loss_sample` (1024 okna) oraz `validation_loss` (okna walidacyjne ze skokiem `stride = context_size // 2 = 128`). Checkpoint `tiny_gpt.json` jest nadpisywany wyłącznie wtedy, gdy `validation_loss` osiąga nowe minimum.

---

## Porównanie kolejnych wersji projektu

Szczegółowy opis każdej wersji znajduje się w katalogu [`changes/`](changes/README.md):

| Wersja | Architektura i kluczowe zmiany | Okno kontekstu | Parametry | Czas treningu (`T=32,8M`) | `full_validation_loss` |
|---|---|---:|---:|---:|---:|
| **Bazowa** | Absolutne `position_embedding`, MHA, `GELU`, spacje osobno (26,7%), `weight_decay` na embeddingach | 64 | 467 584 | 997,1 s (~16,6 min, CPU) | 2,8873 *(stary słownik ze spacjami)* |
| **[Wersja 0.1](changes/v0.1.md)** | Spacje jako prefiks BPE (5,5%), `self.ranks` (23× szybciej), brak `weight_decay` na embeddingach, Attention Dropout, hybrydowe RoPE | 64 | 467 584 | 629,0 s (~10,5 min, CPU) | 3,4272 *(nowy słownik)* |
| **[Wersja 0.2](changes/v0.2.md)** | **Czyste RoPE** (usunięcie `position_embedding`), natywne `SDPA`, walidacja ze skokiem `stride`, kompaktowy JSON (`9,16 MB`) | 64 | **459 392** | 470,5 s (~7,8 min, CPU) | **3,4187** *(nowy słownik)* |
| **[Wersja 0.3](changes/v0.3.md)** | **SwiGLU (`d_ff=384`) + GQA (`4Q/2KV`)**, **KV-Cache**, **Top-p + Repetition Penalty**, **UTF-8 Fallback**, **SFT Chat (`sft_chat.py`)** | 64 | **459 392** | 496,2 s (~8,3 min, CPU) | 3,4375 *(nowy słownik)* |
| **[Wersja 0.4](changes/v0.4.md)** | **4× dłuższe okno (`context_size = 256`)**, automatyczny trening na **GPU Apple Silicon (`mps`) / `cuda`**, pełna 4-wersowa Inwokacja w SFT | **256** | **459 392** | **246,2 s (~4,1 min, MPS)** | **3,4196** *(przy oknie 256 tok.)* |

*(Uwaga: w nowym słowniku BPE ze strumienia zniknęło ponad 40 tys. pojedynczych tokenów spacji `" "`, które były niemal deterministyczne po każdym słowie i w wersji bazowej sztucznie zaniżały średni loss na token).*

---

## Jak dobierać parametry i optymalizować trening

### 1. Policz dane, powtórzenia i rozmiar modelu osobno

| Wielkość | Wartość w obecnej konfiguracji | Co oznacza |
|---|---:|---|
| `V` | `512` | Liczba różnych tokenów w słowniku BPE. |
| `U` | `196 405` | Liczba pozycji tokenów w części treningowej książki. |
| `P` | `459 392` | Liczba wszystkich uczonych parametrów w modelu (niezależna od `context_size` dzięki czystemu RoPE!). |
| `T` | `32 768 000` | Łączna liczba celów tokenowych w treningu (`8 000 × 16 × 256`). |

### 2. Dokładny wzór na liczbę parametrów

Dla architektury z czystym **RoPE** (bez `position_embedding`) oraz wspólną macierzą wejścia/wyjścia (*weight tying*):

- **Wariant `swiglu` + GQA (`n_kv_heads = n_heads / 2`, Wersja 0.3 i 0.4):**
  $$P = V \times D + L \times (3 \times D^2 + 3 \times D \times F + 2 \times D) + D$$
  Dla $V=512, D=128, F=384, L=2$:
  $$P = 65\,536 + 2 \times (49\,152 + 147\,456 + 256) + 128 = 459\,392$$

- **Wariant `gelu` + MHA (Wersja 0.2):**
  $$P = V \times D + L \times (4 \times D^2 + 2 \times D \times F + 2 \times D) + D$$
  Dla $V=512, D=128, F=512, L=2$:
  $$P = 65\,536 + 2 \times (65\,536 + 131\,072 + 256) + 128 = 459\,392$$

Zauważ, że:
1. Zaoszczędzenie parametrów w `Wk` i `Wv` dzięki **GQA** dokładnie równoważy trzecią macierz `W_gate` w **SwiGLU** przy $F = 384$, dając identyczną liczbę parametrów (`459 392`).
2. Dzięki czystemu **RoPE** rozmiar okna kontekstu (`C = 64` vs `C = 256`) w ogóle nie występuje we wzorze na $P$ — rozszerzenie pamięci modelu z 64 do 256 tokenów nie dodało ani jednego nowego parametru.

### 3. Rozpoznawanie przeuczenia i regularyzacja

Porównuj powtarzalne oceny ze stałych zbiorów (`train_loss_sample` i `validation_loss`). Skrypt automatycznie zachowuje w `tiny_gpt.json` checkpoint z najniższą stratą walidacyjną. Przy oznakach przeuczenia dostosowuj osobno `dropout` (obecnie `0,15`, w tym *attention dropout*) oraz `weight_decay` (`0,1` dla macierzy projekcji, `0,0` dla embeddingów i norm).

---

## Pliki w repozytorium i testy automatyczne

- [`model.py`](model.py) — architektura GPT: czyste RoPE, GQA, SwiGLU/GELU, RMSNorm, `forward` (SDPA), `forward_cached` (KV-Cache), `forward_with_attention`, zapis i odczyt.
- [`train_bpe.py`](train_bpe.py) i [`train_config.json`](train_config.json) — trening modelu bazowego z automatycznym wyborem urządzenia (`mps`/`cuda`/`cpu`), walidacją i wyborem najlepszego checkpointu.
- [`predict_gpt.py`](predict_gpt.py) — generowanie tekstu z KV-Cache, Top-p, Top-k, Repetition Penalty, podglądem wag attention i wyborem urządzenia (`cpu`/`mps`/`cuda`).
- [`sft_chat.py`](sft_chat.py) — drugi etap uczenia (**Supervised Fine-Tuning**) z maskowaniem straty na pytaniu oraz tryb rozmowy Q&A (`tiny_gpt_chat.json`).
- [`bpe_tokenizer.py`](bpe_tokenizer.py) i [`tokenizer_bpe.json`](tokenizer_bpe.json) — tokenizer BPE ze spacją jako prefiksem słowa, tablicą rang i fallbackiem UTF-8.
- [`text_data.py`](text_data.py) — wspólny znacznik `<end>` i walidatory argumentów CLI.
- [`test_bpe.py`](test_bpe.py) i [`test_model.py`](test_model.py) — zestaw 25 testów jednostkowych.
- [`training_report.json`](training_report.json) i [`evaluation_report.json`](evaluation_report.json) — raport z ostatniego treningu (`context_size = 256`) oraz archiwalne porównanie wariantów bazowych.
- [`changes/`](changes/README.md) — dokumentacja zmian w wersjach [`v0.1`](changes/v0.1.md), [`v0.2`](changes/v0.2.md), [`v0.3`](changes/v0.3.md) i [`v0.4`](changes/v0.4.md).
- [`data/`](data/SOURCE.md) — pełny tekst „Pana Tadeusza” i opis źródła.

Uruchomienie wszystkich 25 testów jednostkowych:

```sh
.venv/bin/python -m unittest -v
```
