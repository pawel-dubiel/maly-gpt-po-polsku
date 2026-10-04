# Mały GPT po polsku (Wersja 0.5)

Edukacyjny, ale wyposażony we współczesne mechanizmy LLM transformer uczony od zera
na **22,09 MB polskiej literatury (36 tomów powieści z Wolnych Lektur oraz „Pan Tadeusz”)**. Projekt obejmuje:
- **Tokenizację BPE (`vocab_size = 4 096`)** ze spacją jako prefiksem słowa (wzorzec GPT-2/LLaMA), szybkim indeksem odwróconym `pair_words`, tablicą rang i obsługą dowolnych znaków UTF-8 (*byte/Unicode fallback*),
- **Nowoczesną architekturę Transformera (`4,20 mln` parametrów, okno `256` tokenów):** kodowanie pozycji względnych **RoPE** (0 uczonych wag pozycyjnych), grupowaną uwagę **GQA (*Grouped-Query Attention*, `8Q / 4KV`)**, bramkowaną sieć **SwiGLU (`d_ff = 768`)**, **RMSNorm** oraz współdzielenie wag wejścia/wyjścia (*weight tying*),
- **Binarny zapis wag `.safetensors`:** 5,3× mniejsze pliki na dysku (`16,15 MB` zamiast `86 MB` w JSON) oraz ponad 30× szybszy zapis i odczyt z zachowaniem kompatybilności z `.json`,
- **Silnik inferencji i treningu z akceleracją GPU (`mps` / `cuda`):** **KV-Cache**, próbkowanie **Top-p (*Nucleus Sampling*)**, **Top-k**, **Repetition Penalty** oraz automatyczny trening na Apple Silicon (`mps`) lub CUDA,
- **Etap SFT (*Supervised Fine-Tuning*)** z maskowaniem straty na pytaniu użytkownika i trybem rozmowy Q&A ([`sft_chat.py`](sft_chat.py)).

Szczegółowa historia zmian (`Bazowa` → `v0.1` → `v0.2` → `v0.3` → `v0.4` → `v0.5`) znajduje się w [changes/README.md](changes/README.md).
Praktyczny przewodnik: [dobór parametrów, przeuczenie i porównywanie treningów](#jak-dobierać-parametry-i-optymalizować-trening).

---

## Uruchomienie

Polecenia wykonuj w katalogu repozytorium. Każde jest jedną linią — możesz
wkleić je bez znaków kontynuacji `\`. Przykład sprawdzono z Pythonem 3.9 i PyTorch 2.8 na macOS (CPU oraz Apple Silicon `mps`).

### 1. Środowisko i pobranie korpusu powieści (jeśli jeszcze nie pobrano)

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements-full.txt
.venv/bin/python download_corpus.py --output data/polska_proza_full.txt --target-mb 22
```

### 2. Trening modelu bazowego (*Pre-training*, 4,20 mln parametrów)

Wyuczony słownik 4 096 tokenów ([`tokenizer_bpe.json`](tokenizer_bpe.json)) jest już w repozytorium, a wszystkie ustawienia zawiera [train_config.json](train_config.json). Uruchom:

```sh
.venv/bin/python train_bpe.py --config train_config.json
```

Skrypt automatycznie wykrywa kartę graficzną Apple Silicon (`mps`) lub NVIDIA (`cuda`). Trening **12 000 kroków** po **16 okien × 256 tokenów** (`49,15 mln` tokenów łącznie) trwa **~24 minuty**. Powstaną dwa pliki:

- `tiny_gpt.safetensors` (`16,15 MB`) — najlepsze wagi w binarnym standardzie **Safetensors** wraz z metadanymi architektury i tokenizerem BPE w nagłówku `__metadata__`; wystarcza do predykcji oraz dalszego douczania SFT.
- `training_report.json` — ustawienia, hashe danych i modelu, przebieg walidacji, wybrany krok, końcowe wyniki i czas treningu.

Skrypt zaczyna od nowych wag i atomowo zastępuje plik wskazany w konfiguracji.
Wagi oraz pobrany korpus pozostają lokalne i są pomijane przez Git (`.gitignore`).

### 3. Predykcja i generowanie tekstu (z KV-Cache, Top-p i Repetition Penalty)

```sh
.venv/bin/python predict_gpt.py --model tiny_gpt.safetensors --device mps --prompt 'Wokulski wszedł do salonu i spojrzał na' --seed 1 --max-new-tokens 80 --threads 8 --temperature 0.7 --top-k 40 --top-p 0.9 --repetition-penalty 1.15 | tail -n 3
```

Przykładowy wynik:
```text
Most likely next token: ' niego'
Generated: 'Wokulski wszedł do salonu i spojrzał na niego z uwagą.\n\n— Ach, tak — rzekłem — że pani Stawska musi być bardzo szczęśliwym…\n\n— Nie, proszę pani — odparł Wokulski. — Ale pani nie może być źle o tym, że on w niej wie.\n\nPani Krzeszowska wybiegła za nią.\n\n— Zobaczy pan!… — zawołała z uśmiechem.'
Stopped: max-new-tokens limit reached
```

Dostępne parametry próbkowania i urządzenia:
- `--temperature` — zaostrza (`< 1.0`) lub spłaszcza (`> 1.0`) rozkład prawdopodobieństw.
- `--top-k` — ogranicza losowanie do $K$ najbardziej prawdopodobnych tokenów (`--top-k 1` wybiera zachłannie najlepszy token).
- `--top-p` (domyślnie `1.0`, zalecane `0.9`) — *Nucleus Sampling*: dynamicznie odcina ogon rozkładu powyżej skumulowanego prawdopodobieństwa $p$.
- `--repetition-penalty` (domyślnie `1.0`, zalecane `1.15`) — kara za powtarzanie tokenów obecnych w oknie kontekstu.
- `--device` (`cpu`, `mps`, `cuda`, domyślnie `cpu`) — wybór urządzenia obliczeniowego.

Generowanie korzysta z **KV-Cache (`forward_cached`)**: cały prompt przechodzi przez sieć tylko raz (*prefill*), a w kolejnych krokach przeliczany jest wyłącznie **1 ostatnio wygenerowany token** aż do wypełnienia 256-tokenowego okna. Usuń `| tail -n 3`, aby zobaczyć ID tokenów, macierze attention z każdej głowy i bloku oraz posortowane Top-30 najbardziej prawdopodobnych kolejnych tokenów.

### 4. Douczanie instrukcyjne (SFT) i tryb rozmowy Q&A

Model bazowy (`tiny_gpt.safetensors`) uczy się kontynuować prozę i poezję. Aby przekształcić go w model odpowiadający na pytania literackie, uruchom etap **Supervised Fine-Tuning (SFT)** ([`sft_chat.py`](sft_chat.py)):

```sh
.venv/bin/python sft_chat.py train --model tiny_gpt.safetensors --output tiny_gpt_chat.safetensors --steps 150
```

Skrypt doucza model w ~5 sekund na `mps` na parach `Pytanie: ...\nOdpowiedź: ...<end>`, nakładając **maskę straty (`ignore_index = -100`)** na tokeny pytania użytkownika.

Zadawanie pytań wyuczonemu modelowi konwersacyjnemu:

```sh
.venv/bin/python sft_chat.py ask --model tiny_gpt_chat.safetensors --question "Kim jest Stanisław Wokulski?"
.venv/bin/python sft_chat.py ask --model tiny_gpt_chat.safetensors --question "Kim jest profesor Rafał Wilczur?"
.venv/bin/python sft_chat.py ask --model tiny_gpt_chat.safetensors --question "Kim jest Nikodem Dyzma?"
.venv/bin/python sft_chat.py ask --model tiny_gpt_chat.safetensors --question "Jak zaczyna się inwokacja?"
```

---

## Wybrany model (Wersja 0.5)

| Element | Wartość |
|---|---:|
| Słownik BPE (`vocab_size`) | **4 096 tokenów** (całe polskie słowa i morfemy) |
| Wektor tokenu (`d_model`) | **256 liczb** |
| Kontekst (`context_size`) | **256 tokenów (~150–180 słów prozy)** |
| Kodowanie pozycji | **Czyste RoPE** (*Rotary Position Embeddings*, 0 uczonych wag) |
| Bloki transformera (`n_layers`) | **4** |
| Głowy zapytań `Q` (`n_heads`) | **8** (po 32 liczby) |
| Głowy kluczy i wartości `K/V` (`n_kv_heads`, GQA) | **4** (po 32 liczby, współdzielone przez 8 głów `Q`) |
| Projekcje uwagi w bloku | `Wq`, `Wo`: 256 × 256; `Wk`, `Wv`: 256 × 128 |
| Sieć Feed-Forward (**SwiGLU**) | `W_gate`: 256 × 768, `W1`: 256 × 768, `W2`: 768 × 256 |
| Dropout w treningu | 0,10 (w tym *attention dropout*) |
| Liczba parametrów | **4 196 608 (~4,20 mln)** |
| Format i rozmiar pliku wag | **`.safetensors` (`16,15 MB`)** |

---

## Dane i tokeny

Korpus `data/polska_proza_full.txt` pobierany przez [`download_corpus.py`](download_corpus.py) z otwartych zasobów *Wolnych Lektur* obejmuje **36 tomów** polskiej literatury (**22,09 MB**, `21 232 149` znaków, **`6 542 560` tokenów BPE**): powieści Tadeusza Dołęgi-Mostowicza, Bolesława Prusa, Stefana Żeromskiego, Henryka Sienkiewicza, Elizy Orzeszkowej oraz „Pana Tadeusza” Adama Mickiewicza.

Podgląd działania tokenizera BPE (`vocab_size = 4 096`):

```sh
.venv/bin/python bpe_tokenizer.py inspect --tokenizer tokenizer_bpe.json --text 'Wokulski wszedł do salonu i spojrzał na'
```

Odtworzenie treningu tokenizera:

```sh
.venv/bin/python bpe_tokenizer.py train --data data/polska_proza_full.txt --output tokenizer_bpe.json --vocab-size 4096
```

Pierwsze **5 888 304 tokeny** służą do treningu; ostatnie **654 256 tokenów** do walidacji. Strumień dzielimy **przed** tworzeniem okien, więc żadne okno treningowe nie sięga do walidacji.

---

## Porównanie kolejnych wersji projektu

Szczegółowy opis każdej wersji znajduje się w katalogu [`changes/`](changes/README.md):

| Wersja | Architektura i kluczowe zmiany | Korpus / Słownik | Parametry | Czas treningu | `full_validation_loss` |
|---|---|---|---:|---:|---:|
| **Bazowa** | Absolutne `position_embedding`, MHA, `GELU`, spacje osobno (26,7%) | 0,45 MB / `V=512` | 467 584 | 997,1 s (CPU) | 2,8873 *(stary BPE)* |
| **[Wersja 0.1](changes/v0.1.md)** | Spacje jako prefiks BPE (5,5%), `self.ranks`, brak `weight_decay` na embeddingach | 0,45 MB / `V=512` | 467 584 | 629,0 s (CPU) | 3,4272 *(V=512)* |
| **[Wersja 0.2](changes/v0.2.md)** | **Czyste RoPE**, natywne `SDPA`, walidacja ze skokiem `stride` | 0,45 MB / `V=512` | 459 392 | 470,5 s (CPU) | 3,4187 *(V=512)* |
| **[Wersja 0.3](changes/v0.3.md)** | **SwiGLU + GQA**, **KV-Cache**, **Top-p + Repetition Penalty**, **SFT (`sft_chat.py`)** | 0,45 MB / `V=512` | 459 392 | 496,2 s (CPU) | 3,4375 *(V=512)* |
| **[Wersja 0.4](changes/v0.4.md)** | **Okno `context_size = 256`**, automatyczny trening na **GPU (`mps`/`cuda`)** | 0,45 MB / `V=512` | 459 392 | 246,2 s (MPS) | 3,4196 *(V=512)* |
| **[Wersja 0.5](changes/v0.5.md)** | **Korpus 22,09 MB polskiej prozy (36 tomów)**, **`V=4096`**, **model `4,20M` (`4L × 256D`)**, format **`.safetensors` (`16,15 MB`)** | **22,09 MB / `V=4096`** | **4 196 608** | **1 464,5 s (~24,4 min, MPS)** | **4,3907 *(V=4096, top 1,9%!)*** |

---

## Jak dobierać parametry i optymalizować trening

### 1. Policz dane, powtórzenia i rozmiar modelu osobno

| Wielkość | Wartość w obecnej konfiguracji (Wersja 0.5) | Co oznacza |
|---|---:|---|
| `V` | `4 096` | Liczba różnych tokenów w słowniku BPE. |
| `U` | `5 888 304` | Liczba pozycji tokenów w części treningowej korpusu (36 tomów). |
| `P` | `4 196 608` | Liczba wszystkich uczonych parametrów w modelu (niezależna od `context_size` dzięki czystemu RoPE). |
| `T` | `49 152 000` | Łączna liczba celów tokenowych w treningu (`12 000 × 16 × 256`). |

### 2. Dokładny wzór na liczbę parametrów

Dla architektury z czystym **RoPE**, **SwiGLU** + **GQA (`n_kv_heads = n_heads / 2`)** oraz wspólną macierzą wejścia/wyjścia (*weight tying*):
$$P = V \times D + L \times (3 \times D^2 + 3 \times D \times F + 2 \times D) + D$$
Dla $V=4096, D=256, F=768, L=4$:
$$P = 1\,048\,576 + 4 \times (196\,608 + 589\,824 + 512) + 256 = 4\,196\,608$$

---

## Pliki w repozytorium i testy automatyczne

- [`model.py`](model.py) — architektura GPT: czyste RoPE, GQA, SwiGLU/GELU, RMSNorm, `forward` (SDPA), `forward_cached` (KV-Cache), `forward_with_attention`, zapis i odczyt **`.safetensors`** oraz `.json`.
- [`download_corpus.py`](download_corpus.py) — pobieranie i czyszczenie korpusu 36 tomów polskiej prozy z Wolnych Lektur (`data/polska_proza_full.txt`).
- [`train_bpe.py`](train_bpe.py) i [`train_config.json`](train_config.json) — trening modelu bazowego z automatycznym wyborem urządzenia (`mps`/`cuda`/`cpu`), walidacją i wyborem najlepszego checkpointu.
- [`predict_gpt.py`](predict_gpt.py) — generowanie tekstu z KV-Cache, Top-p, Top-k, Repetition Penalty, podglądem wag attention i wyborem urządzenia (`cpu`/`mps`/`cuda`).
- [`sft_chat.py`](sft_chat.py) — drugi etap uczenia (**Supervised Fine-Tuning**) z maskowaniem straty na pytaniu oraz tryb rozmowy Q&A (`tiny_gpt_chat.safetensors`).
- [`bpe_tokenizer.py`](bpe_tokenizer.py) i [`tokenizer_bpe.json`](tokenizer_bpe.json) — tokenizer BPE (`4 096` tokenów) ze spacją jako prefiksem słowa, indeksem `pair_words`, tablicą rang i fallbackiem UTF-8.
- [`text_data.py`](text_data.py) — wspólny znacznik `<end>` i walidatory argumentów CLI.
- [`test_bpe.py`](test_bpe.py) i [`test_model.py`](test_model.py) — zestaw 26 testów jednostkowych (w tym test formatu `.safetensors`).
- [`training_report.json`](training_report.json) i [`evaluation_report.json`](evaluation_report.json) — raport z ostatniego treningu (`4,20 mln` parametrów na `22,09 MB` polskiej prozy) oraz archiwalne porównanie wariantów bazowych.
- [`changes/`](changes/README.md) — dokumentacja zmian w wersjach [`v0.1`](changes/v0.1.md), [`v0.2`](changes/v0.2.md), [`v0.3`](changes/v0.3.md), [`v0.4`](changes/v0.4.md) i [`v0.5`](changes/v0.5.md).
- [`data/`](data/SOURCE.md) — pełny tekst „Pana Tadeusza”, opis źródła oraz pobrany korpus polskiej prozy.

Uruchomienie wszystkich 26 testów jednostkowych:

```sh
.venv/bin/python -m unittest -v
```
