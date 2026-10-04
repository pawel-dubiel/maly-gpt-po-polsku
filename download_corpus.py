"""Pobieranie i czyszczenie korpusu polskiej prozy z Wolnych Lektur (~20-25 MB, Próg 1)."""

import argparse
import hashlib
import json
from pathlib import Path
import re
import ssl
import urllib.error
import urllib.request

from text_data import END, positive_integer


# Lista tomów powieściowych z Wolnych Lektur (domena publiczna, bezpośrednie pliki .txt).
# Obejmuje polską prozę XX-wieczną (Dołęga-Mostowicz, Żeromski) oraz klasykę pozytywizmu
# (Prus, Sienkiewicz, Orzeszkowa, Reymont).
WOLNE_LEKTURY_NOVELS = [
    # Tadeusz Dołęga-Mostowicz (nowoczesna polszczyzna XX-wieczna)
    ("kariera-nikodema-dyzmy", "Tadeusz Dołęga-Mostowicz — Kariera Nikodema Dyzmy"),
    ("znachor", "Tadeusz Dołęga-Mostowicz — Znachor"),
    ("profesor-wilczur", "Tadeusz Dołęga-Mostowicz — Profesor Wilczur"),
    ("dolega-mostowicz-pamietnik-pani-hanki", "Tadeusz Dołęga-Mostowicz — Pamiętnik pani Hanki"),
    ("dolega-mostowicz-prokurator-alicja-horn", "Tadeusz Dołęga-Mostowicz — Prokurator Alicja Horn"),
    ("dr-murek-zredukowany", "Tadeusz Dołęga-Mostowicz — Dr. Murek zredukowany"),
    ("dolega-mostowicz-drugie-zycie-doktora-murka", "Tadeusz Dołęga-Mostowicz — Drugie życie doktora Murka"),
    ("bracia-dalcz-i-s-ka-tom-1", "Tadeusz Dołęga-Mostowicz — Bracia Dalcz i S-ka, tom I"),
    ("bracia-dalcz-i-s-ka-tom-2", "Tadeusz Dołęga-Mostowicz — Bracia Dalcz i S-ka, tom II"),
    # Bolesław Prus
    ("lalka-tom-pierwszy", "Bolesław Prus — Lalka, tom I"),
    ("lalka-tom-drugi", "Bolesław Prus — Lalka, tom II"),
    ("faraon-tom-pierwszy", "Bolesław Prus — Faraon, tom I"),
    ("faraon-tom-drugi", "Bolesław Prus — Faraon, tom II"),
    ("faraon-tom-trzeci", "Bolesław Prus — Faraon, tom III"),
    ("prus-placowka", "Bolesław Prus — Placówka"),
    ("emancypantki-tom-i", "Bolesław Prus — Emancypantki, tom I"),
    ("emancypantki-tom-ii", "Bolesław Prus — Emancypantki, tom II"),
    # Stefan Żeromski
    ("przedwiosnie", "Stefan Żeromski — Przedwiośnie"),
    ("syzyfowe-prace", "Stefan Żeromski — Syzyfowe prace"),
    ("ludzie-bezdomni-tom-pierwszy", "Stefan Żeromski — Ludzie bezdomni, tom I"),
    ("ludzie-bezdomni-tom-drugi", "Stefan Żeromski — Ludzie bezdomni, tom II"),
    ("wierna-rzeka", "Stefan Żeromski — Wierna rzeka"),
    # Henryk Sienkiewicz
    ("quo-vadis", "Henryk Sienkiewicz — Quo vadis"),
    ("w-pustyni-i-w-puszczy", "Henryk Sienkiewicz — W pustyni i w puszczy"),
    ("rodzina-polanieckich", "Henryk Sienkiewicz — Rodzina Połanieckich"),
    ("bez-dogmatu", "Henryk Sienkiewicz — Bez dogmatu"),
    ("potop-tom-pierwszy", "Henryk Sienkiewicz — Potop, tom I"),
    ("potop-tom-drugi", "Henryk Sienkiewicz — Potop, tom II"),
    ("potop-tom-trzeci", "Henryk Sienkiewicz — Potop, tom III"),
    # Eliza Orzeszkowa
    ("nad-niemnem-tom-pierwszy", "Eliza Orzeszkowa — Nad Niemnem, tom I"),
    ("nad-niemnem-tom-drugi", "Eliza Orzeszkowa — Nad Niemnem, tom II"),
    ("nad-niemnem-tom-trzeci", "Eliza Orzeszkowa — Nad Niemnem, tom III"),
    ("orzeszkowa-marta", "Eliza Orzeszkowa — Marta"),
    ("orzeszkowa-cham", "Eliza Orzeszkowa — Cham"),
    ("dziurdziowie", "Eliza Orzeszkowa — Dziurdziowie"),
    # Władysław Stanisław Reymont
    ("ziemia-obiecana-tom-pierwszy", "Władysław Stanisław Reymont — Ziemia obiecana, tom I"),
    ("ziemia-obiecana-tom-drugi", "Władysław Stanisław Reymont — Ziemia obiecana, tom II"),
    ("komediantka", "Władysław Stanisław Reymont — Komediantka"),
]

INLINE_FOOTNOTE = re.compile(r"\s*\[[^\]]*przypis[^\]]*\]")


def clean_wolne_lektury_text(raw_text):
    text = raw_text.replace("\r\n", "\n").replace("\r", "\n")
    # Odcięcie stopki wydawniczej Wolnych Lektur (poprzedzonej linią '-----').
    if "\n-----\n" in text:
        text = text.split("\n-----\n")[0]
    # Odcięcie sekcji przypisów na końcu utworu, jeśli występuje.
    if "\nPrzypisy:\n" in text:
        text = text.split("\nPrzypisy:\n")[0]
    lines = []
    for line in text.split("\n"):
        stripped = line.strip()
        if stripped.startswith("ISBN ") or stripped.startswith("ISBN-"):
            continue
        line = INLINE_FOOTNOTE.sub("", line)
        lines.append(line)
    cleaned = "\n".join(lines).strip()
    cleaned = cleaned.replace(END, "")
    cleaned = re.sub(r"\n{4,}", "\n\n\n", cleaned)
    return cleaned + "\n"


def fetch_book(slug, cache_dir):
    cache_path = cache_dir / f"{slug}.txt"
    if cache_path.is_file() and cache_path.stat().st_size > 1000:
        return cache_path.read_text(encoding="utf-8")
    url = f"https://wolnelektury.pl/media/book/txt/{slug}.txt"
    req = urllib.request.Request(url, headers={"User-Agent": "MalyGPTPoPolsku/0.5 (educational LLM project)"})
    ssl_ctx = ssl._create_unverified_context()
    with urllib.request.urlopen(req, timeout=30, context=ssl_ctx) as response:
        data = response.read().decode("utf-8")
    cache_path.write_text(data, encoding="utf-8")
    return data


def build_corpus(output_path, target_mb=22, include_pan_tadeusz=True):
    root = Path(__file__).parent
    cache_dir = root / "data" / "wolnelektury"
    cache_dir.mkdir(parents=True, exist_ok=True)
    target_bytes = target_mb * 1024 * 1024
    parts = []
    manifest = []
    total_bytes = 0

    if include_pan_tadeusz:
        pt_path = root / "data" / "pan_tadeusz_full.txt"
        if pt_path.is_file():
            pt_text = pt_path.read_text(encoding="utf-8").strip() + "\n"
            parts.append(pt_text)
            size = len(pt_text.encode("utf-8"))
            total_bytes += size
            manifest.append({"slug": "pan-tadeusz", "title": "Adam Mickiewicz — Pan Tadeusz", "bytes": size})
            print(f"[1] Pan Tadeusz: {size / (1024 * 1024):.2f} MB (łącznie: {total_bytes / (1024 * 1024):.2f} MB)", flush=True)

    for slug, title in WOLNE_LEKTURY_NOVELS:
        if total_bytes >= target_bytes:
            break
        try:
            raw = fetch_book(slug, cache_dir)
        except (urllib.error.URLError, TimeoutError, OSError) as error:
            print(f"Pominięto {slug}: {error}", flush=True)
            continue
        cleaned = clean_wolne_lektury_text(raw)
        size = len(cleaned.encode("utf-8"))
        if size < 10_000:
            continue
        parts.append(cleaned)
        total_bytes += size
        manifest.append({"slug": slug, "title": title, "bytes": size})
        print(f"[{len(manifest)}] {title}: {size / (1024 * 1024):.2f} MB (łącznie: {total_bytes / (1024 * 1024):.2f} MB)", flush=True)

    combined = "\n\n".join(part.strip() for part in parts) + "\n"
    output_path = Path(output_path)
    output_path.write_text(combined, encoding="utf-8")
    manifest_path = output_path.with_suffix(".manifest.json")
    summary = {
        "output": str(output_path),
        "books_count": len(manifest),
        "total_bytes": len(combined.encode("utf-8")),
        "total_megabytes": round(len(combined.encode("utf-8")) / (1024 * 1024), 2),
        "total_chars": len(combined),
        "sha256": hashlib.sha256(combined.encode("utf-8")).hexdigest(),
        "books": manifest,
    }
    manifest_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"\nZapisano korpus: {output_path} ({summary['total_megabytes']} MB, {summary['books_count']} tomów, {summary['total_chars']} znaków)", flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="data/polska_proza_full.txt")
    parser.add_argument("--target-mb", type=positive_integer, default=22)
    args = parser.parse_args()
    build_corpus(args.output, target_mb=args.target_mb)


if __name__ == "__main__":
    main()
