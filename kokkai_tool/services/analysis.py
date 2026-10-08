"""Keyword-frequency and topic-modeling analysis."""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from itertools import combinations
from math import log
from pathlib import Path

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import japanize_matplotlib  # noqa: F401 - configures Japanese-capable plot fonts
from matplotlib.figure import Figure
import pandas as pd
import spacy
from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import TfidfVectorizer

LogCallback = Callable[[str], None]

PALETTE = ["#2563EB", "#0D9488", "#F59E0B", "#7C3AED", "#E11D48", "#0891B2"]


def apply_figure_style() -> None:
    """Apply one readable, publication-friendly style to every figure."""
    plt.rcParams.update({
        "figure.facecolor": "white", "axes.facecolor": "#F8FAFC",
        "axes.edgecolor": "#CBD5E1", "axes.labelcolor": "#334155",
        "axes.titlecolor": "#0F172A", "axes.titleweight": "bold",
        "axes.grid": True, "grid.color": "#E2E8F0", "grid.linewidth": 0.8,
        "axes.axisbelow": True, "font.size": 10, "legend.frameon": False,
        "xtick.color": "#475569", "ytick.color": "#475569",
    })


@dataclass
class AnalysisResult:
    data: pd.DataFrame
    figure: Figure
    csv_path: Path
    image_path: Path


class SpeechAnalyzer:
    """Lazily tokenizes a Diet CSV and reuses the tokens across analyses."""

    def __init__(self, csv_path: str | Path, stopwords_path: str | Path, log: LogCallback | None = None):
        self.csv_path = Path(csv_path)
        self.data = pd.read_csv(self.csv_path, encoding="utf-8-sig")
        required = {"comment-body", "date"}
        missing = required.difference(self.data.columns)
        if missing:
            raise ValueError(f"CSV is missing required columns: {', '.join(sorted(missing))}")
        self.data["comment-body"] = self.data["comment-body"].fillna("").astype(str)
        self.data["date"] = pd.to_datetime(self.data["date"], errors="coerce")
        self.stopwords = {
            line.strip() for line in Path(stopwords_path).read_text(encoding="utf-8").splitlines()
            if line.strip()
        }
        self.log = log or (lambda _message: None)
        self.tokens: list[list[str]] | None = None
        self.vectorizer: TfidfVectorizer | None = None
        self.matrix = None
        self.document_topics = None
        self.topic_labels: list[str] = []
        self.output_dir = self.csv_path.parent / "outputs"
        self.output_dir.mkdir(exist_ok=True)

    def _ensure_tokens(self) -> None:
        if self.tokens is not None:
            return
        self.log("Starting initial document tokenization…")
        try:
            nlp = spacy.load("ja_core_news_lg", disable=["ner"])
        except OSError as exc:
            raise RuntimeError(
                "Japanese spaCy model not found. Run: python -m spacy download ja_core_news_lg"
            ) from exc
        total = len(self.data)
        tokenized: list[list[str]] = []
        texts = self.data["comment-body"].tolist()
        for index, text in enumerate(texts, start=1):
            # Preserve the original backend's reliable behavior: long Diet
            # speeches are processed sentence-by-sentence instead of being
            # passed to spaCy as one oversized input.
            segments = text.split("。") if len(text) > 1_500 else [text]
            words: list[str] = []
            for segment in segments:
                if not segment.strip():
                    continue
                doc = nlp(segment)
                words.extend(
                    token.lemma_.strip() for token in doc
                    if token.lemma_.strip() and not token.is_space and not token.is_punct
                    and not token.is_stop and token.lemma_ not in self.stopwords
                )
            tokenized.append(words)
            if index == total or index % max(1, total // 10) == 0:
                self.log(f"Tokenized {index:,} of {total:,} documents ({index / max(total, 1):.0%}).")
        self.tokens = tokenized
        self.log("Initial tokenization complete.")

    def _ensure_vectors(self) -> None:
        self._ensure_tokens()
        if self.matrix is None:
            self.vectorizer = TfidfVectorizer(token_pattern=r"(?u)\b\w+\b", min_df=1)
            self.matrix = self.vectorizer.fit_transform(" ".join(words) for words in self.tokens or [])

    def keyword_frequency(self, top_n: int = 30) -> AnalysisResult:
        self._ensure_tokens()
        counts = Counter(word for document in self.tokens or [] for word in document)
        result = pd.DataFrame(counts.most_common(), columns=["keyword", "count"])
        csv_path = self.output_dir / f"{self.csv_path.stem}_keyword_frequency.csv"
        image_path = self.output_dir / f"{self.csv_path.stem}_keyword_frequency.png"
        result.to_csv(csv_path, index=False, encoding="utf-8-sig")

        display = result.head(top_n).sort_values("count")
        apply_figure_style()
        fig = Figure(figsize=(10, max(5.5, top_n * 0.25)))
        ax = fig.subplots()
        ax.barh(display["keyword"], display["count"], color=PALETTE[0], height=0.72)
        ax.set(title=f"Top {min(top_n, len(display))} keywords", xlabel="Occurrences", ylabel="")
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.grid(axis="x"); ax.grid(axis="y", visible=False)
        fig.tight_layout()
        fig.savefig(image_path, dpi=220, bbox_inches="tight", facecolor="white")
        return AnalysisResult(result, fig, csv_path, image_path)

    def coherence_scores(self, minimum: int, maximum: int, top_words: int) -> AnalysisResult:
        self._ensure_vectors()
        scores = []
        documents = [set(words) for words in self.tokens or []]
        document_count = max(len(documents), 1)
        word_documents = Counter(word for document in documents for word in document)

        def topic_npmi(words: list[str]) -> float:
            pair_scores = []
            for first, second in combinations(words, 2):
                together = sum(first in document and second in document for document in documents)
                if not together:
                    pair_scores.append(-1.0)
                    continue
                p_pair = together / document_count
                p_first = word_documents[first] / document_count
                p_second = word_documents[second] / document_count
                pair_scores.append(
                    1.0 if p_pair == 1.0
                    else log(p_pair / (p_first * p_second)) / -log(p_pair)
                )
            return sum(pair_scores) / max(len(pair_scores), 1)

        assert self.vectorizer is not None
        terms = self.vectorizer.get_feature_names_out()
        for count in range(minimum, maximum + 1):
            model = NMF(n_components=count, init="nndsvda", random_state=42, max_iter=500)
            model.fit(self.matrix)
            topics = [
                [str(terms[index]) for index in weights.argsort()[::-1][:top_words]]
                for weights in model.components_
            ]
            scores.append(sum(topic_npmi(words) for words in topics) / len(topics))
        frame = pd.DataFrame({"topics": range(minimum, maximum + 1), "coherence_npmi": scores})
        csv_path = self.output_dir / f"{self.csv_path.stem}_topic_coherence.csv"
        image_path = self.output_dir / f"{self.csv_path.stem}_topic_coherence.png"
        frame.to_csv(csv_path, index=False)
        apply_figure_style()
        fig = Figure(figsize=(9, 5.5))
        ax = fig.subplots()
        ax.plot(frame["topics"], frame["coherence_npmi"], color=PALETTE[0], marker="o", linewidth=2.4)
        ax.set(title="Topic coherence by number of topics", xlabel="Number of topics", ylabel="Mean NPMI coherence")
        ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
        ax.spines[["top", "right"]].set_visible(False)
        fig.tight_layout(); fig.savefig(image_path, dpi=220, bbox_inches="tight")
        return AnalysisResult(frame, fig, csv_path, image_path)

    def extract_topics(self, number: int, top_words: int) -> tuple[str, pd.DataFrame]:
        self._ensure_vectors()
        model = NMF(n_components=number, init="nndsvda", random_state=42, max_iter=500)
        self.document_topics = model.fit_transform(self.matrix)
        assert self.vectorizer is not None
        terms = self.vectorizer.get_feature_names_out()
        lines, rows, labels = [], [], []
        for topic_index, weights in enumerate(model.components_, start=1):
            indices = weights.argsort()[::-1][:top_words]
            words = [str(terms[index]) for index in indices]
            label = f"Topic {topic_index}: {', '.join(words[:3])}"
            labels.append(label)
            lines.append(f"Topic {topic_index}\n  {'  ·  '.join(words)}")
            rows.extend({"topic": topic_index, "keyword": terms[index], "weight": weights[index]} for index in indices)
        self.topic_labels = labels
        frame = pd.DataFrame(rows)
        frame.to_csv(self.output_dir / f"{self.csv_path.stem}_topics.csv", index=False, encoding="utf-8-sig")
        return "\n\n".join(lines), frame

    def yearly_topic_distribution(self, mode: str = "Share") -> AnalysisResult:
        if self.document_topics is None:
            raise RuntimeError("Extract topics before creating a yearly distribution.")
        years = self.data["date"].dt.year
        dominant = self.document_topics.argmax(axis=1)
        valid = years.notna()
        if not valid.any():
            raise RuntimeError("The selected CSV has no valid dates for yearly analysis.")
        counts = pd.crosstab(years[valid].astype(int), dominant[valid.to_numpy()]).sort_index()
        counts = counts.reindex(columns=range(self.document_topics.shape[1]), fill_value=0)
        counts.columns = self.topic_labels
        shown = counts.div(counts.sum(axis=1), axis=0).fillna(0) * 100 if mode == "Share" else counts
        csv_path = self.output_dir / f"{self.csv_path.stem}_yearly_topic_{mode.lower()}.csv"
        image_path = self.output_dir / f"{self.csv_path.stem}_yearly_topic_{mode.lower()}.png"
        shown.to_csv(csv_path, encoding="utf-8-sig")
        apply_figure_style()
        fig = Figure(figsize=(11, 6.5))
        ax = fig.subplots()
        shown.plot.area(ax=ax, color=[PALETTE[i % len(PALETTE)] for i in range(shown.shape[1])], alpha=0.9)
        ax.set(title=f"Yearly Topic Distribution — {mode.lower()}", xlabel="Year", ylabel="Share (%)" if mode == "Share" else "Dominant-topic speeches")
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(title="Topics", bbox_to_anchor=(1.02, 1), loc="upper left")
        ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
        fig.tight_layout(); fig.savefig(image_path, dpi=220, bbox_inches="tight", facecolor="white")
        return AnalysisResult(shown, fig, csv_path, image_path)
