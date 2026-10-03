"""Wspólny znacznik końca i walidacja wymaganych argumentów liczbowych."""

import argparse
import math


END = "<end>"


def positive_integer(text):
    value = int(text)
    if value <= 0:
        raise argparse.ArgumentTypeError("Expected a positive integer.")
    return value


def positive_float(text):
    value = float(text)
    if not math.isfinite(value) or value <= 0:
        raise argparse.ArgumentTypeError("Expected a finite positive number.")
    return value


def fraction(text):
    value = float(text)
    if not math.isfinite(value) or not 0 < value < 1:
        raise argparse.ArgumentTypeError("Expected a finite fraction strictly between 0 and 1.")
    return value
