# Mały GPT po polsku

Edukacyjny transformer GPT uczony od zera na „Panu Tadeuszu”. Projekt obejmuje
tokenizację BPE, attention Q/K/V, trening z walidacją, zapis najlepszego modelu
i generowanie tekstu. README i objaśnienia kodu są po polsku.

To model do nauki budowy LLM. Potrafi naśladować słownictwo i wersy książki,
ale nadal popełnia błędy językowe i nie prowadzi rozmowy jak ChatGPT.
Konfigurację dobrano przez rzeczywiste treningi i pomiar walidacji.

Praktyczny przewodnik: [dobór parametrów, przeuczenie i porównywanie treningów](#jak-dobierać-parametry-i-optymalizować-trening).

## Uruchomienie

Polecenia wykonuj w katalogu repozytorium. Każde jest jedną linią — możesz
wkleić je bez znaków kontynuacji `\`. Wymagany jest Python i CPU; przykład
sprawdzono z Pythonem 3.9 i PyTorch 2.8 na macOS.

### 1. Środowisko

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements-full.txt
```

### 2. Trening

Książka i gotowy tokenizer są już w repozytorium. Wszystkie ustawienia zawiera
[train_config.json](train_config.json). Uruchom:

```sh
.venv/bin/python train_bpe.py --config train_config.json
```

Powstaną dwa pliki:

- `tiny_gpt.json` — najlepsze wagi, architektura i tokenizer; wystarcza do predykcji.
- `training_report.json` — ustawienia, hashe danych i modelu, przebieg walidacji,
  wybrany krok, końcowe wyniki i czas treningu.

Skrypt zaczyna od nowych wag. Nie musisz usuwać starego modelu: nowy zapis
atomowo zastępuje plik wskazany w konfiguracji. Uruchomienie nie wznawia
poprzedniego treningu. Wagi pozostają lokalne i są pomijane przez Git.
Raport powstaje po zakończeniu treningu; jego `model_sha256` identyfikuje model,
którego wyniki opisuje.

Ścieżki z konfiguracji są liczone względem katalogu **pliku konfiguracji**.
Każde pole jest wymagane; brak ustawienia, błędna wartość, brak pliku lub
niezgodny format kończą się jawnym błędem. Model i raport nie mogą wskazywać
pliku książki, tokenizera ani konfiguracji.

### 3. Predykcja

```sh
.venv/bin/python predict_gpt.py --model tiny_gpt.json --prompt 'Tadeusz' --seed 1 --max-new-tokens 64 --threads 1 --temperature 0.6 --top-k 40 | tail -n 3
```

- `Most likely next token` — najbardziej prawdopodobny token przed ograniczeniem losowania.
- `Generated` — prompt i wylosowana kontynuacja; `\n` oznacza nową linię.
- `Stopped` — zakończenie przez limit albo token `<end>`.

`temperature=0.6` zaostrza rozkład prawdopodobieństw, a `top-k=40` ogranicza
losowanie do 40 najbardziej prawdopodobnych tokenów na każdym kroku.
`--top-k 1` zawsze wybiera najbardziej prawdopodobny token. Pełny rozkład
uzyskasz przez `--temperature 1 --top-k 512`. Te ustawienia zmieniają losowanie,
a nie wyuczone wagi ani wynik walidacji. Wszystkie argumenty są wymagane.

Prompt może mieć do **64 tokenów BPE**. Dalsze generowanie przesuwa okno
kontekstu. Predykcja korzysta wyłącznie z checkpointu — nie potrzebuje książki
ani osobnego tokenizera. Dropout jest wtedy wyłączony; ten sam seed i ustawienia
odtwarzają wynik w tym samym środowisku.

Usuń `| tail -n 3`, aby zobaczyć ID tokenów, attention z każdej głowy i bloku
oraz prawdopodobieństwa wszystkich 512 tokenów. Nieznany znak w prompcie
powoduje błąd; nowe słowo jest dozwolone, jeśli znamy wszystkie jego znaki.

Jeśli terminal pokazuje `dquote>`, cudzysłów nie został zamknięty.
Naciśnij Ctrl+C i wklej całe polecenie z prostymi apostrofami `'`.

**Starszy `tiny_gpt_bpe64.json` używa formatu v1.** Obecny kod wymaga v2 i nowego
treningu. Zmieniła się architektura, więc nie wystarczy zmienić nazwy starego
pliku. Docelowy model nazywa się `tiny_gpt.json`.

## Wybrany model

| Element | Wartość |
|---|---:|
| Słownik BPE | 512 tokenów |
| Wektor tokenu | 128 liczb |
| Kontekst | 64 tokenów |
| Bloki transformera | 2 |
| Głowy attention w każdym bloku | 4 |
| Rozmiar jednej głowy | 32 liczby |
| Wq, Wk, Wv, Wo w każdym bloku | 128 × 128 |
| Feed-forward | 128 → 512 → 128 |
| Dropout w treningu | 0,15 |
| Liczba parametrów | 467 584 |

Każdy z 512 wpisów słownika ma jeden uczony wektor 128D. Liczba wystąpień
słowa w książce nie tworzy nowych embeddingów. W kontekście wektory są
przekształcane przez uwagę i feed-forward, dlatego reprezentacja tego samego
tokenu zależy od poprzedzającego tekstu.

[model.py](model.py) pokazuje operacje bez ukrywania attention w gotowym bloku:

```text
ID → embedding tokenu + embedding pozycji → dropout

Powtórz dla każdego z dwóch bloków:
  X = RMSNorm(wektory)
  Q = X Wq, K = X Wk, V = X Wv
  podział Q, K, V na 4 głowy po 32 liczby
  A = softmax(Q Kᵀ / √32 + maska przyszłości)
  połączenie głów A V → Wo → dropout → dodanie wejścia bloku
  RMSNorm → W1 → GELU → W2 → dropout → dodanie wejścia tej części

końcowy RMSNorm → mnożenie przez transponowaną macierz embeddingów
→ 512 logitów → softmax → prawdopodobieństwa następnego tokenu
```

Maska zabrania patrzenia na przyszłe tokeny. Połączenia przez dodawanie to
*residual connections*. Wyjście współdzieli wagi z embeddingami wejściowymi
(*weight tying*). Dropout losowo zeruje część aktywacji podczas treningu,
a RMSNorm normalizuje ich skalę. Wszystkie macierze i współczynniki RMSNorm
uczą się przez gradienty.

Wagi zaczynają od rozkładu normalnego o odchyleniu 0,02. W projekcjach wyjściowych
attention i feed-forward odchylenie jest dodatkowo dzielone przez
`√(2 × liczba_bloków)`. Współczynniki RMSNorm zaczynają od 1.

## Dane i tokeny

Korpus zawiera wszystkie dwanaście ksiąg i epilog „Pana Tadeusza” Adama
Mickiewicza: **445 638 znaków** i **258 648 tokenów BPE**. Pochodzi z Wolnych
Lektur; [źródło i prawa do tekstu](data/SOURCE.md) opisano osobno.

BPE zaczyna od znaków i łączy częste sąsiednie pary. Token może być fragmentem
słowa, całym słowem, spacją albo interpunkcją. Łączenia nie przekraczają granic
między ciągami liter/cyfr, białych znaków i interpunkcji. Dekodowanie odtwarza
tekst wraz ze spacjami, znakami polskimi i nowymi liniami.

Podgląd:

```sh
.venv/bin/python bpe_tokenizer.py inspect --tokenizer tokenizer_bpe.json --text 'Litwo! Ojczyzno moja!'
```

```text
Pieces: ['Li', 't', 'wo', '!', ' ', 'O', 'j', 'czy', 'zno', ' ', 'mo', 'ja', '!']
IDs: [483, 59, 136, 3, 2, 31, 49, 140, 489, 2, 154, 160, 3]
```

Tokenizer jest zapisany w repo. Jeśli chcesz odtworzyć jego uczenie:

```sh
.venv/bin/python bpe_tokenizer.py train --data data/pan_tadeusz_full.txt --output tokenizer_bpe.json --vocab-size 512
```

Pierwsze **232 783 tokeny** służą do treningu; ostatnie **25 865** do walidacji.
Dzielimy strumień **przed** tworzeniem okien. Żadne okno treningowe nie sięga
do walidacji. Tokenizer wyuczono na całej książce i pozostawiono stały:
walidacja dotyczy wag transformera; nie jest niezależnym testem doboru
tokenizera ani osobnym zbiorem testowym.

## Co dzieje się podczas treningu

Wejścia i cele są przesunięte o jeden token:

```text
wejście: ['Li', 't',  'wo', '!']
cel:     ['t',  'wo', '!',  ' ']
```

W rzeczywistym batchu są 32 losowane okna po 64 tokenów. Każda pozycja
ma własny cel. Cross-entropy mierzy średnią stratę tych przewidywań, a PyTorch
oblicza gradienty wszystkich wag. Globalna norma gradientów jest ograniczana
do 1. AdamW aktualizuje wagi; weight decay wynosi 0,1 dla macierzy i 0 dla
współczynników RMSNorm. Parametry AdamW, w tym bety i epsilon, są jawne
w konfiguracji.

Trening trwa **16 000 kroków**. Okna są losowane z powtórzeniami, więc krok nie
jest epoką ani przejściem przez całą książkę. Learning rate przez pierwsze
200 kroków rośnie do 0,001, następnie maleje kosinusowo do 0,0001.

Po kroku 1, co 1000 kroków oraz na końcu skrypt ocenia:

- `train_loss_sample` — stałą próbkę 1024 równomiernie rozmieszczonych okien
  treningowych; dla krótszego tekstu używa wszystkich dostępnych okien.
- `validation_loss` — **wszystkie** okna odłożonej części książki.

Walidacja działa bez gradientów i dropout. Checkpoint jest zastępowany tylko
przy ściśle niższej stracie walidacyjnej; remis zachowuje wcześniejszy model.
Po treningu skrypt wczytuje wybrany checkpoint i mierzy loss na wszystkich
oknach treningowych i walidacyjnych. Wyniki końcowe w raporcie dotyczą tego
pliku, a nie ostatniego modelu w pamięci. Ocena całego zbioru może chwilę potrwać.

`<end>` jest zarezerwowany w słowniku, ale nie dopisujemy go jako sztucznego celu
po każdym oknie książki. Głównym ograniczeniem długości generowania jest
`--max-new-tokens`.

## Zmierzone wyniki

W repo pozostaje jedna konfiguracja. Tabela dokumentuje próby, które posłużyły
do jej wyboru.

Wspólny pomiar ocenia **25 737 tych samych docelowych tokenów** z końcowych
10% książki. Pomija pierwsze 128 tokenów tej części; każdy model dostaje swoje
maksymalne okno poprzedzające cel (16, 64 albo 128 tokenów). Liczymy cross-entropy
tylko dla ostatniej pozycji okna. Mniej oznacza lepsze przewidywanie.

| Sprawdzony wariant | Loss na tych samych celach |
|---|---:|
| Poprzedni model 64D, 1 blok, kontekst 16 | 3.062 |
| Próba 64D, 2 bloki, kontekst 64 | 3.174 |
| Próba 128D, 2 bloki, kontekst 64, 6000 kroków | 2.945 |
| Próba 128D, 2 bloki, kontekst 64, 16000 kroków | 2.866 |
| Próba 128D, 4 bloki, kontekst 64, 12000 kroków | 2.864 |
| Kod docelowy: 128D, 2 bloki, kontekst 64, 16000 kroków | 2.856 |
| Kod docelowy: 128D, 2 bloki, kontekst 128, 12000 kroków | 2.909 |

Wybrany model ma loss **2.856 zamiast 3.062**.
Odpowiada to obniżeniu perplexity o około **18.6%**. Wagi pochodzą z
kroku **16000**; cały docelowy trening wraz z oceną trwał na tym
komputerze około **16.6 minuty**. Czasy zależą od sprzętu;
część eksperymentów wykonywano równolegle.

Raport treningu podaje też loss uśredniony po **wszystkich pozycjach wszystkich
okien**: trening **2.241**, walidacja **2.887**.
To inny sposób uśredniania niż w tabeli, dlatego liczby są różne.
Przy porównywaniu modeli używaj tej samej metody pomiaru.

Pełne wyniki, hashe checkpointów i wszystkie próbki z trzech stałych promptów
i seedów 1 oraz 7 znajdują się w [evaluation_report.json](evaluation_report.json).
Zapisano również porównanie temperatur 0,6 / 0,8 / 1,0. Przykładowy początek
wyniku z polecenia powyżej (temperatura 0,6, seed 1):

```text
Tadeusz nie uśmiechnął:
«Nigdy chcąc!» — wołając Sędzia — nie wiem, że potrzeba:
```

**Poprawa lossu nie oznacza poprawnego polskiego.** Próbka nadal ma błędy
składniowe. Model naśladuje słownictwo książki, ale nie utrzymuje niezawodnie
sensu ani gramatyki. Jedna książka, mały model i jeden seed treningu ograniczają
wnioski. Są to najlepsze ustawienia w tym porównaniu, nie dowód globalnego optimum.
Walidacja służyła także do wyboru modelu, więc nie jest osobnym testem końcowym.


## Jak dobierać parametry i optymalizować trening

**Nie ma przelicznika „jedna książka = model o określonej wielkości”.** Szukamy
ustawień, które przy dostępnym czasie poprawiają przewidywanie odłożonego tekstu.
Więcej parametrów albo więcej kroków może pomóc, ale wynik trzeba zmierzyć.
Poniższa procedura dotyczy uczenia naszego modelu od zera.

Wagi, np. liczby w `Wq`, są **parametrami uczonymi**. Rozmiar wektora,
liczba bloków i learning rate to **hiperparametry**: ustawiasz je przed treningiem.
Konfiguracja zawiera jawne ustawienia wybranego eksperymentu; nie są to
wartości zastępcze dla dowolnego tekstu.

### 1. Policz dane, powtórzenia i rozmiar modelu osobno

| Wielkość | Co oznacza w tym repo |
|---|---|
| `V = 512` | Liczba różnych tokenów w słowniku, a więc liczba wierszy embeddingów. |
| `U = 232 783` | Liczba pozycji tokenów w części treningowej książki, przed powtarzaniem okien. To nie liczba różnych słów ani różnych ID. |
| `P = 467 584` | Liczba wszystkich uczonych liczb w modelu. |
| `T = 32 768 000` | Łączna liczba celów tokenowych przetworzonych w aktualizacjach wag. |

W tym skrypcie każde okno ma tę samą długość, więc:

```text
T = steps × batch_size × context_size
  = 16 000 × 32 × 64
  = 32 768 000

T / U ≈ 141
```

Ostatnia liczba opisuje skalę wielokrotnego wykorzystania danych. Nie oznacza
141 pełnych epok: okna losujemy z powtórzeniami, nakładają się na siebie,
a pozycje przy brzegach części treningowej pojawiają się rzadziej.
**Powtarzanie książki nie tworzy nowych zdań ani nowych źródeł wiedzy.**

Sprawdź także różnorodność tekstu, błędy OCR, powtarzane nagłówki i duplikaty.
Dziesięć kopii tej samej książki nie daje takiej różnorodności jak dziesięć
różnych książek. Badania pokazują malejącą korzyść z kolejnych powtórzeń danych;
nie przenoś jednak ich konkretnej liczby epok na nasz przykład z nakładającymi
się oknami. [Badanie o ograniczonej ilości danych](https://arxiv.org/abs/2305.16264).

Badania nad skalowaniem dużych LLM łączą ilość danych, rozmiar modelu i budżet
obliczeń. Nie stanowią gotowej recepty dla jednej książki i kilkuset tysięcy
parametrów: tutaj rozmiar wybieramy eksperymentalnie, na walidacji.
[Badanie o doborze skali treningu](https://arxiv.org/abs/2203.15556).

### 2. Zrozum, co zwiększa koszt i pojemność modelu

Dla **tej implementacji**, bez biasów i ze wspólną macierzą wejścia/wyjścia,
liczbę parametrów można policzyć dokładnie:

```text
V = rozmiar słownika          C = context_size
D = d_model                  F = d_ff
L = n_layers

P = (V + C) × D + L × (4 × D² + 2 × D × F + 2 × D) + D
```

Pierwszy składnik to embeddingi, nawias zawiera Q/K/V/Wo, feed-forward i dwie
normy bloku, a ostatnie `D` to końcowa norma. Dla naszej konfiguracji wynik
wynosi 467 584. To liczba wag, nie rozmiar pliku JSON ani całej pamięci treningu.
Trening przechowuje również gradienty, stan AdamW i aktywacje.

| Ustawienie | Jak je dobierać i co sprawdzać |
|---|---|
| Słownik BPE | Mniejszy słownik daje zwykle dłuższe sekwencje fragmentów. Większy zwiększa embeddingi i może zawierać wiele rzadkich fragmentów. Sprawdzaj długość sekwencji i częstości tokenów w treningu. U nas wybrano 512. |
| `d_model`, `d_ff` | Określają szerokość obliczeń. Przy utrzymaniu `d_ff = 4 × d_model` podwojenie szerokości czterokrotnie powiększa macierze projekcji attention i feed-forward; embeddingi i normy rosną dwukrotnie. Sprawdź większą szerokość, gdy zarówno trening, jak i walidacja pozostają słabe mimo stabilnego uczenia. |
| `n_layers` | Więcej kolejnych bloków pozwala na więcej przekształceń kontekstu, ale zwiększa koszt. U nas cztery bloki nie uzasadniły swojego kosztu względem dwóch. |
| `n_heads` | Przy stałym `d_model` zmienia podział uwagi, a nie liczbę wag w tym modelu. Wymagane jest `d_model % n_heads == 0`; więcej głów daje krótszy wektor każdej głowy. |
| `context_size` | Dobieraj do zależności, które model ma wykorzystywać. Podwojenie kontekstu czterokrotnie zwiększa liczbę elementów każdej macierzy attention; nie oznacza dokładnie czterokrotnie dłuższego całego treningu. U nas 128 tokenów nie poprawiło wyniku względem 64. |
| `batch_size` | Liczba okien na aktualizację. Wpływa na pamięć, szum gradientów i ilość tekstu przetwarzanego w kroku. Większy batch nie gwarantuje lepszego modelu. |

Jeśli masz więcej **nowego, przydatnego tekstu**, najpierw sprawdź na nim
obecny rozmiar modelu. Przy stałej liczbie kroków każdy fragment będzie
wykorzystany przeciętnie rzadziej; może być potrzebny dłuższy trening.
Dopiero porównanie krzywych uzasadnia zwiększanie pojemności. Przy mniejszym
korpusie sprawdź mniejszy model i wcześniejsze zakończenie, zamiast utrzymywać
za wszelką cenę liczbę kroków wybraną dla większego zbioru.

### 3. Zaprojektuj ocenę przed pierwszą próbą

Ustal, co model ma umieć: kontynuować tę książkę, podobną literaturę czy
współczesną polszczyznę. Dobierz odłożone teksty do tego celu. Dobry wynik
na epilogu „Pana Tadeusza” nie mierzy umiejętności prowadzenia rozmowy.

W większym eksperymencie wydziel trzy części: **trening** aktualizuje wagi,
**walidacja** służy do wyboru ustawień i checkpointu, a **test** do końcowej
oceny zamkniętej konfiguracji. Wielokrotne poprawianie modelu na podstawie
testu zamienia go w kolejną walidację. Liczebność i reprezentatywność odłożonych
danych są ważniejsze niż sztywny procent podziału.
[Opis podziału i przecieku danych](https://developers.google.com/machine-learning/crash-course/overfitting/dividing-datasets).

W tekście najpierw rozdziel dokumenty lub ciągłe fragmenty, **potem twórz
okna**. Losowy podział gotowych, nakładających się okien przepuszcza prawie
identyczne fragmenty do obu części. Przy wielu książkach grupuj podział
według książek; jeśli oceniasz nowych autorów, rozdziel także autorów.
Usuń duplikaty między częściami. Tysiące nakładających się okien jednej księgi
nie są tysiącami niezależnych tekstów.

Skrypt obsługuje dwie części: trening i walidację. Udział wybiera
`validation_fraction`; obecna konfiguracja daje podział 90% / 10%.
Osobny test oraz podział według dokumentów wymagałyby rozszerzenia przygotowania
danych. Przy rygorystycznym eksperymencie ucz również reguły BPE na treningu
i z góry określ obsługiwany alfabet. Obecny tokenizer zna całą książkę;
nieznany znak powoduje błąd. Nie ma automatycznej obsługi dowolnego alfabetu.

### 4. Rozpoznawaj przeuczenie po przebiegu, nie po jednej liczbie

**Overfitting, czyli przeuczenie**, oznacza coraz lepsze dopasowanie do danych
treningowych bez odpowiadającej mu poprawy na nowych danych. Typowym sygnałem
jest spadek straty treningowej przy utrzymującym się wzroście walidacyjnej.

| Obserwacja w kolejnych ocenach | Co sprawdzić lub zrobić |
|---|---|
| Obie straty maleją | Model nadal poprawia przewidywania na walidacji. Oceń, czy zysk uzasadnia dalszy czas. |
| Trening maleje, walidacja przez kilka ocen rośnie | Podejrzenie przeuczenia. Zachowaj wcześniejszy checkpoint; sprawdź mniejszy model, regularyzację lub nowe dane. |
| Obie straty są wysokie i stoją | Możliwe niedouczenie, nieodpowiedni learning rate, zbyt silna regularyzacja albo błąd danych/kodu. Sam wykres nie wskazuje jednej przyczyny. |
| Strata gwałtownie skacze lub pojawia się NaN | Sprawdź stabilność obliczeń, dane i learning rate. To nie jest typowy objaw samego przeuczenia. |

Porównuj powtarzalne oceny ze stałych zbiorów. Jeden losowy batch nie wystarcza
do diagnozy. Takie rozróżnienie pomaga odróżnić problemy optymalizacji od
przeuczenia. [Przewodnik po krzywych straty](https://developers.google.com/machine-learning/crash-course/overfitting/interpreting-loss-curves).

W naszym zapisanym treningu wygląda to następująco. Obie oceny działają bez
dropout i bez aktualizacji wag, także `train_loss_sample`.

| Krok | `train_loss_sample` | `validation_loss` |
|---:|---:|---:|
| 1 000 | 3,271 | 3,393 |
| 8 000 | 2,442 | 2,927 |
| 16 000 | 2,247 | 2,887 |

Różnica rośnie, ale walidacja nadal się poprawia. **Sam odstęp między krzywymi
nie wystarcza, żeby nakazać zatrzymanie treningu.** Części książki mogą też
różnić się trudnością. Nie ma uniwersalnego progu typu „różnica 0,5 = overfitting”.
Powyższe wyniki są z `history` w raporcie; nie mieszaj ich z końcowym
`full_training_loss`, który ocenia wszystkie okna zamiast próbki.

W odrzuconym wariancie z kontekstem 128 walidacja pogorszyła się z około 2,9348
przy 11 000 kroków do 2,9356 przy 12 000. To wystarczyło do zachowania wcześniejszych
wag, ale pojedyncze tak małe pogorszenie nie dowodzi trwałego przeuczenia.

### 5. Ustal zasadę zatrzymania i regularyzację

**Wybór najlepszego checkpointu** i **early stopping** to dwie różne czynności.
Pierwszą skrypt wykonuje: zapisuje każde ścisłe minimum walidacji. Drugiej
nie implementuje: wykonuje wszystkie zadane `steps`, nawet gdy wynik już
się nie poprawia.

Przykładowa reguła planowania przyszłego eksperymentu: po fazie warmup zakończ
trening, jeśli przez 3 kolejne oceny nie uda się poprawić dotychczasowego
istotnego minimum o co najmniej 0,005. To odpowiednio *patience* i *min_delta*.
Przy ocenie co 1000 kroków cierpliwość wynosi 3000 kroków. Dobierz te liczby
do wahań wyniku i kosztu treningu; to przykład, nie ustalony próg dla każdej książki.
Checkpoint można nadal zapisywać przy każdym, nawet mniejszym spadku straty.

Tej reguły **nie można obecnie włączyć polem w JSON**. Nieznane pola zostaną
odrzucone. Najpierw analizuj logi i raport, a automatyczne zatrzymanie wymagałoby
zmiany pętli treningowej. Przerwanie procesu pozostawia zapisany checkpoint,
ale końcowy raport powstaje dopiero po zakończeniu skryptu; raport z wcześniejszego
uruchomienia nie opisuje nowych wag. Porównuj jego hash z plikiem modelu.

Przy oznakach przeuczenia sprawdzaj osobno `dropout` i `weight_decay`.
Dropout utrudnia poleganie na tych samych aktywacjach, a weight decay ogranicza
wzrost wag. Zbyt duże wartości mogą utrudnić samo uczenie. Nie zwiększaj obu
jednocześnie, jeśli chcesz wiedzieć, która zmiana pomogła. U nas punktem
odniesienia są 0,15 i 0,1; nie ma gwarancji, że będą najlepsze dla nowych danych.
W AdamW weight decay jest oddzielone od gradientu straty. Gradient clipping
ogranicza normę gradientów, ale sam nie rozwiązuje przeuczenia.

### 6. Prowadź małe, porównywalne eksperymenty

1. **Zachowaj punkt odniesienia.** Skopiuj konfigurację, model i raport przed
   zmianami. Nadaj kolejnemu uruchomieniu osobne ścieżki `model` i `report`.
   Zanotuj hipotezę, np. „mniejszy learning rate poprawi stabilność”.
2. **Sprawdź poprawność kodu na krótkim fragmencie.** Powinien umieć mocno obniżyć
   jego stratę. To test działania uczenia, nie dowód jakości języka; repo ma
   taki test automatyczny. Nie zaczynaj długiej serii, gdy ten test zawodzi.
3. **Najpierw dobierz learning rate.** Przy tej samej architekturze porównaj
   obecną wartość z mniejszą i większą, np. o czynnik 2. Zachowaj sposób
   wyznaczania warmup i proporcję `min_learning_rate` do maksimum. Wybieraj
   na podstawie stabilności i walidacji, nie najszybszego spadku jednego batcha.
4. **Następnie zmieniaj pojemność lub regularyzację.** Jedna hipoteza na próbę:
   szerokość, liczba bloków, kontekst albo dropout. Zmienianie wszystkiego
   jednocześnie nie pozwala przypisać poprawy konkretnej przyczynie.
5. **Ustal budżet porównania.** Przy tym samym tokenizerze zapisuj `T`, czas
   i liczbę parametrów. Równa liczba kroków przy innym batchu lub kontekście
   nie oznacza równej ilości przetworzonego tekstu. Równe `T` przy różnych
   modelach też nie oznacza równego kosztu obliczeń.
6. **Sprawdź najlepsze ustawienia z kilkoma seedami treningu**, np. 1, 2 i 3.
   Podaj średnią i rozrzut wyników zamiast wybierać szczęśliwy seed.
   Seed w `predict_gpt.py` zmienia tylko losowanie tekstu; nie zastępuje tego testu.
7. **Porównaj teksty na stałych promptach.** Zachowaj te same temperatury,
   top-k, limity i seedy generowania. Oceniaj błędy słów, składnię, powtórzenia,
   sens i kopiowanie fragmentów książki. Zapisuj wszystkie próbki, także słabe.
8. **Wybierz model według wcześniej ustalonego celu.** Przy podobnym wyniku
   mniejszy lub szybszy model może być lepszym wyborem. Istotność małej różnicy
   oceń przez rozrzut między treningami, nie samą liczbę miejsc po przecinku.

Zmienianie `steps` zmienia w tym kodzie również cały harmonogram kosinusowy.
Trening zaplanowany na 8000 kroków nie jest pierwszą połową treningu zaplanowanego
na 16 000: learning rate wcześniej zacznie zbliżać się do minimum. Skrypt nie
wznawia optymalizatora z checkpointu. Zanotuj tę różnicę przy porównaniach.

### 7. Uważaj na porównywanie różnych metryk

Cross-entropy liczymy tu z logarytmem naturalnym, a `perplexity = exp(loss)`.
Mniejsza perplexity oznacza lepsze przewidywanie według tego samego pomiaru;
nie jest procentem poprawnych zdań. Zmiana temperatury generowania nie
poprawia straty walidacyjnej zapisanych wag.

Przy zmianie kontekstu używaj tych samych docelowych tokenów, tak jak w tabeli
zmierzonych wyników. Przy zmianie tokenizera samo „loss na token” przestaje
być bezpośrednio porównywalne: tokeny mają inną długość. Potrzebujesz oceny
tego samego surowego tekstu, np. sumy ujemnych log-prawdopodobieństw podzielonej
przez liczbę ocenianych bajtów UTF-8 i przez `ln(2)` — to bity na bajt.
Zachowaj te same granice tekstu, zasady kontekstu i liczenie każdego celu raz.
Obecny skrypt nie oblicza tej metryki; nasze porównanie zachowuje jeden tokenizer.

Zmiana słownika BPE wymaga nowego treningu tokenizera i modelu. Przypisanie
innego słownika do gotowych embeddingów nie jest poprawną kontynuacją uczenia.
Jeśli wszystkie małe warianty przestają poprawiać walidację, sprawdź jakość,
różnorodność i zgodność danych z zadaniem, zamiast bez końca zwiększać kroki.
Nowe teksty mogą pomóc, lecz sama liczba plików nie gwarantuje poprawy.


## Pliki i sprawdzenie obliczeń

- `model.py` — Q/K/V, głowy, bloki, normy, feed-forward, zapis i odczyt.
- `train_bpe.py` i `train_config.json` — jeden przebieg treningu z walidacją.
- `predict_gpt.py` — osobny program do przewidywania tokenów.
- `bpe_tokenizer.py` i `tokenizer_bpe.json` — tokenizer i jego słownik.
- `training_report.json` i `evaluation_report.json` — wyniki zmierzonego treningu i porównania.
- `data/` — tekst treningowy, oryginalny plik źródłowy i opis pochodzenia.

```sh
.venv/bin/python -m unittest -v
```

Testy porównują attention z rachunkiem liczbowym oraz wyjście i **gradienty
wszystkich wag** z niezależną implementacją opartą na funkcjach PyTorch.
Sprawdzają maskę przyczynową, wiele głów i bloków, BPE, podział danych,
wybór wcześniejszego checkpointu, wyłączanie dropout, wymagane ustawienia,
zapis i odczyt oraz rzeczywisty trening małego testowego tekstu i predykcję
po usunięciu tego tekstu i osobnego tokenizera.
