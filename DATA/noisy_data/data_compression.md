Data Compression — Lossless Block Sorting

A lossless compressor for textual data. The idea is to find repeated substrings
within a sliding window and encode each match as a distance-length pair
instead of literal characters.

The reference implementation sorts a set of candidate strings by their suffixes
using quicksort with a three-way partition, then locates the previous occurrence
of each suffix to emit a match.

The compression ratio depends heavily on the input. Natural-language text
typically compresses to between 25% and 40% of its original size. Already
compressed data — JPEG, PNG, MP4 — usually grows slightly when passed through a
general-purpose text compressor, because the bit patterns that a text
compressor is looking for have been deliberately removed.

The window size bounds the maximum match distance. A larger window finds longer
matches but requires a larger history buffer and slows the search.
