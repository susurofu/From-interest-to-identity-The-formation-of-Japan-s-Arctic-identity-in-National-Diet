"""National Diet Speech Analysis Toolkit release application."""

from __future__ import annotations

import queue
import re
import threading
from datetime import date
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from tkinter.scrolledtext import ScrolledText

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

from services.analysis import AnalysisResult, SpeechAnalyzer
from services.downloader import download_speeches

APP_DIR = Path(__file__).resolve().parent
STOPWORDS = APP_DIR / "services" / "stopwords_japanese.txt"

COLORS = {
    "bg": "#FFFFFF", "card": "#FFFFFF", "text": "#000000",
    "muted": "#000000", "entry": "#FFFFFF", "log": "#000000",
    "log_text": "#FFFFFF",
}


class App(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("National Diet Speech Analysis Toolkit")
        self.geometry("920x680")
        self.minsize(820, 600)
        self.configure(bg=COLORS["bg"])
        self.selected_file = tk.StringVar()
        self.status = tk.StringVar(value="Ready")
        self.events: queue.Queue[tuple] = queue.Queue()
        self.busy = False
        self.analyzer: SpeechAnalyzer | None = None
        self.analyzer_path = ""
        self._configure_style()
        self._build_shell()
        self.after(80, self._drain_events)

    def _configure_style(self) -> None:
        style = ttk.Style(self)
        if "clam" in style.theme_names():
            style.theme_use("clam")
        style.configure("TFrame", background=COLORS["bg"])
        style.configure("Card.TFrame", background=COLORS["card"])
        style.configure("TLabel", background=COLORS["bg"], foreground=COLORS["text"], font=("TkDefaultFont", 10))
        style.configure("Card.TLabel", background=COLORS["card"], foreground=COLORS["text"], font=("TkDefaultFont", 10))
        style.configure("Title.TLabel", background=COLORS["bg"], foreground=COLORS["text"], font=("TkDefaultFont", 15, "bold"))
        style.configure("Subtitle.TLabel", background=COLORS["bg"], foreground=COLORS["muted"], font=("TkDefaultFont", 9))
        style.configure("CardTitle.TLabel", background=COLORS["card"], foreground=COLORS["text"], font=("TkDefaultFont", 10, "bold"))
        style.configure("Muted.Card.TLabel", background=COLORS["card"], foreground=COLORS["muted"], font=("TkDefaultFont", 9))
        style.configure("TEntry", padding=3, fieldbackground="white", foreground="black", bordercolor="black", lightcolor="black", darkcolor="black")
        style.configure("TCombobox", padding=2, fieldbackground="white", foreground="black", background="white", bordercolor="black", arrowcolor="black")
        style.map("TCombobox", fieldbackground=[("readonly", "white")], foreground=[("readonly", "black")], selectbackground=[("readonly", "black")], selectforeground=[("readonly", "white")])
        style.configure("TButton", font=("TkDefaultFont", 9), padding=(8, 4), foreground="black", background="white", bordercolor="black", lightcolor="white", darkcolor="black")
        style.configure("Primary.TButton", font=("TkDefaultFont", 9, "bold"), padding=(8, 4), foreground="white", background="black", bordercolor="black", lightcolor="black", darkcolor="black")
        style.configure("Secondary.TButton", font=("TkDefaultFont", 9), padding=(8, 4), foreground="black", background="white", bordercolor="black", lightcolor="white", darkcolor="black")
        style.map("TButton", background=[("active", "black")], foreground=[("active", "white")])
        style.map("Primary.TButton", background=[("active", "white")], foreground=[("active", "black")])
        style.map("Secondary.TButton", background=[("active", "black")], foreground=[("active", "white")])
        style.configure("TNotebook", background="white", bordercolor="black", borderwidth=1, relief="solid")
        style.configure("TNotebook.Tab", font=("TkDefaultFont", 9), padding=(12, 5), foreground="black", background="white", bordercolor="black")
        style.map("TNotebook.Tab", background=[("selected", "black")], foreground=[("selected", "white")])

    def _build_shell(self) -> None:
        menu_options = {
            "tearoff": False, "bg": "white", "fg": "black",
            "activebackground": "black", "activeforeground": "white",
            "selectcolor": "black", "borderwidth": 1,
        }
        menu = tk.Menu(self, **menu_options)
        file_menu = tk.Menu(menu, **menu_options)
        file_menu.add_command(label="Open CSV...", command=lambda: self.pages["data"].browse(), accelerator="Ctrl+O")
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.destroy)
        menu.add_cascade(label="File", menu=file_menu)
        analysis_menu = tk.Menu(menu, **menu_options)
        analysis_menu.add_command(label="Keyword Frequency", command=lambda: self.show_page("analysis"))
        analysis_menu.add_command(label="Topic Modeling", command=lambda: self.show_page("topics"))
        menu.add_cascade(label="Analysis", menu=analysis_menu)
        help_menu = tk.Menu(menu, **menu_options)
        help_menu.add_command(label="About", command=lambda: messagebox.showinfo("About", "National Diet Speech Analysis Toolkit\nRelease version", parent=self))
        menu.add_cascade(label="Help", menu=help_menu)
        self.configure(menu=menu)
        self.bind_all("<Control-o>", lambda _event: self.pages["data"].browse())

        filebar = tk.Frame(self, bg=COLORS["bg"], relief="solid", borderwidth=1)
        filebar.pack(fill="x", padx=3, pady=(3, 0))
        ttk.Button(filebar, text="Open CSV...", command=lambda: self.pages["data"].browse()).pack(side="left", padx=4, pady=4)
        tk.Label(filebar, text="Data file:", bg=COLORS["bg"], font=("TkDefaultFont", 9)).pack(side="left", padx=(8, 4))
        tk.Label(filebar, textvariable=self.selected_file, bg="white", anchor="w", relief="sunken", borderwidth=1, font=("TkDefaultFont", 9)).pack(side="left", fill="x", expand=True, padx=(0, 4), pady=4, ipady=3)

        statusbar = tk.Label(self, textvariable=self.status, bg=COLORS["bg"], fg=COLORS["text"], anchor="w", relief="solid", borderwidth=1, font=("TkDefaultFont", 9), padx=5)
        statusbar.pack(side="bottom", fill="x", padx=3, pady=3, ipady=2)
        self.content = ttk.Notebook(self)
        self.content.pack(fill="both", expand=True, padx=3, pady=3)
        self.pages = {
            "data": DataPage(self.content, self),
            "analysis": KeywordPage(self.content, self),
            "topics": TopicPage(self.content, self),
        }
        for key, label in (("data", "Data Source"), ("analysis", "Keyword Frequency"), ("topics", "Topic Modeling")):
            self.content.add(self.pages[key], text=label)
        self.show_page("data")

    def show_page(self, name: str) -> None:
        self.content.select(self.pages[name])

    def set_file(self, path: str | Path) -> None:
        resolved = str(Path(path).resolve())
        if resolved != self.selected_file.get():
            self.analyzer = None
            self.analyzer_path = ""
        self.selected_file.set(resolved)
        self.status.set(f"Dataset ready: {Path(resolved).name}")

    def get_analyzer(self) -> SpeechAnalyzer:
        path = self.selected_file.get()
        if not path:
            raise ValueError("Select or download a CSV dataset first.")
        if self.analyzer is None or self.analyzer_path != path:
            self.analyzer = SpeechAnalyzer(path, STOPWORDS, log=lambda message: self.post("log", message))
            self.analyzer_path = path
        return self.analyzer

    def run_task(self, function, *, success=None, busy="Working…") -> None:
        if self.busy:
            messagebox.showinfo("Please wait", "Another operation is still running.", parent=self)
            return
        self.busy = True
        self.status.set(busy)
        def worker():
            try:
                result = function()
                self.post("success", result, success)
            except Exception as exc:
                self.post("error", exc)
        threading.Thread(target=worker, daemon=True).start()

    def post(self, kind: str, *payload) -> None:
        self.events.put((kind, *payload))

    def _drain_events(self) -> None:
        try:
            while True:
                event = self.events.get_nowait()
                kind, payload = event[0], event[1:]
                if kind == "log":
                    message = str(payload[0])
                    self.status.set(message)
                    self.pages["data"].append_log(message)
                elif kind == "success":
                    result, callback = payload
                    self.busy = False
                    self.status.set("Completed")
                    if callback:
                        callback(result)
                elif kind == "error":
                    self.busy = False
                    self.status.set("Could not complete the operation")
                    messagebox.showerror("Operation failed", str(payload[0]), parent=self)
        except queue.Empty:
            pass
        self.after(80, self._drain_events)


class Page(ttk.Frame):
    def __init__(self, parent, app: App, title: str, subtitle: str):
        super().__init__(parent, padding=(14, 12))
        self.app = app
        ttk.Label(self, text=title, style="Title.TLabel").pack(anchor="w")
        ttk.Label(self, text=subtitle, style="Subtitle.TLabel").pack(anchor="w", pady=(2, 12))

    def card(self) -> ttk.Frame:
        frame = ttk.Frame(self, style="Card.TFrame", padding=10, relief="groove", borderwidth=2)
        frame.pack(fill="x", pady=(0, 9))
        return frame


def field(parent, label: str, variable=None, width=28) -> ttk.Entry:
    wrapper = ttk.Frame(parent, style="Card.TFrame")
    wrapper.pack(side="left", padx=(0, 9), fill="x", expand=True)
    ttk.Label(wrapper, text=label, style="Muted.Card.TLabel").pack(anchor="w", pady=(0, 3))
    entry = ttk.Entry(wrapper, textvariable=variable, width=width)
    entry.pack(fill="x")
    return entry


class DataPage(Page):
    def __init__(self, parent, app: App):
        super().__init__(parent, app, "Choose your data", "Download speeches from the National Diet API or open an existing export.")
        card = self.card()
        ttk.Label(card, text="Download from the Diet API", style="CardTitle.TLabel").pack(anchor="w", pady=(0, 14))
        row = ttk.Frame(card, style="Card.TFrame"); row.pack(fill="x")
        self.keywords = field(row, "Keywords (space separated)")
        self.start_date = field(row, "Start date")
        self.end_date = field(row, "End date")
        self.start_date.insert(0, "2000-01-01"); self.end_date.insert(0, date.today().isoformat())
        row2 = ttk.Frame(card, style="Card.TFrame"); row2.pack(fill="x", pady=(14, 0))
        self.filename = field(row2, "Output filename", width=42)
        self.filename.insert(0, "diet_speeches")
        ttk.Button(row2, text="Download speeches", style="Primary.TButton", command=self.download).pack(side="right", anchor="s", pady=(20, 0))

        open_card = self.card()
        ttk.Label(open_card, text="Use an existing dataset", style="CardTitle.TLabel").pack(anchor="w")
        ttk.Label(open_card, text="Choose a CSV exported by this tool or containing date and comment-body columns.", style="Muted.Card.TLabel").pack(anchor="w", pady=(4, 12))
        ttk.Button(open_card, text="Browse for CSV", style="Secondary.TButton", command=self.browse).pack(anchor="w")

        log_card = self.card()
        header = ttk.Frame(log_card, style="Card.TFrame"); header.pack(fill="x")
        ttk.Label(header, text="Download & tokenization log", style="CardTitle.TLabel").pack(side="left")
        ttk.Button(header, text="Clear", command=lambda: self.log.delete("1.0", "end")).pack(side="right")
        ttk.Label(log_card, text="Only Diet API download and initial tokenization messages appear here.", style="Muted.Card.TLabel").pack(anchor="w", pady=(3, 8))
        self.log = ScrolledText(log_card, height=7, relief="sunken", borderwidth=2, bg=COLORS["log"], fg=COLORS["log_text"], insertbackground="white", font=("Courier", 9), padx=6, pady=5, state="disabled")
        self.log.pack(fill="both", expand=True)

    def append_log(self, message: str) -> None:
        self.log.configure(state="normal"); self.log.insert("end", message.rstrip() + "\n"); self.log.see("end"); self.log.configure(state="disabled")

    def browse(self) -> None:
        path = filedialog.askopenfilename(parent=self, title="Select Diet speech CSV", filetypes=[("CSV files", "*.csv")])
        if path:
            self.app.set_file(path); self.app.show_page("analysis")

    def download(self) -> None:
        keywords = self.keywords.get().strip(); start = self.start_date.get().strip(); end = self.end_date.get().strip()
        name = re.sub(r"[^\w\-.]+", "_", self.filename.get().strip()).strip("._")
        if not keywords or not name:
            messagebox.showwarning("Missing information", "Enter at least one keyword and an output filename.", parent=self); return
        try:
            start_value, end_value = date.fromisoformat(start), date.fromisoformat(end)
            if start_value > end_value: raise ValueError
        except ValueError:
            messagebox.showwarning("Invalid dates", "Use YYYY-MM-DD dates and ensure the start is before the end.", parent=self); return
        target = APP_DIR / "speeches" / f"{name}.csv"
        self.app.run_task(lambda: download_speeches(keywords, start, end, target, log=lambda msg: self.app.post("log", msg)), success=self._downloaded, busy="Downloading Diet speeches…")

    def _downloaded(self, path: Path) -> None:
        self.app.set_file(path); self.app.show_page("analysis")


class KeywordPage(Page):
    def __init__(self, parent, app: App):
        super().__init__(parent, app, "Keyword frequency", "Count filtered Japanese lemmas, save the full CSV, and preview a polished chart.")
        card = self.card()
        ttk.Label(card, text="Analysis settings", style="CardTitle.TLabel").pack(anchor="w", pady=(0, 14))
        row = ttk.Frame(card, style="Card.TFrame"); row.pack(fill="x")
        self.top_n = field(row, "Keywords shown in chart", width=18); self.top_n.insert(0, "30")
        ttk.Button(row, text="Run keyword analysis", style="Primary.TButton", command=self.run).pack(side="right", pady=(20, 0))
        note = self.card()
        ttk.Label(note, text="What is saved", style="CardTitle.TLabel").pack(anchor="w")
        ttk.Label(note, text="The complete frequency table is saved as UTF-8 CSV. The displayed top-keyword chart is saved as a high-resolution PNG in an outputs folder next to your dataset.", style="Muted.Card.TLabel", wraplength=700, justify="left").pack(anchor="w", pady=(5, 0))

    def run(self) -> None:
        try:
            top_n = int(self.top_n.get()); assert 5 <= top_n <= 100
        except (ValueError, AssertionError):
            messagebox.showwarning("Invalid value", "Choose between 5 and 100 keywords.", parent=self); return
        self.app.run_task(lambda: self.app.get_analyzer().keyword_frequency(top_n), success=lambda result: show_plot(self.app, result, "Keyword frequency"), busy="Analyzing keyword frequency…")


class TopicPage(Page):
    def __init__(self, parent, app: App):
        super().__init__(parent, app, "Topic modeling", "Evaluate, extract, and inspect how topics change from year to year.")
        evaluation = self.card(); ttk.Label(evaluation, text="1. Evaluate topic counts", style="CardTitle.TLabel").pack(anchor="w", pady=(0, 14))
        row = ttk.Frame(evaluation, style="Card.TFrame"); row.pack(fill="x")
        self.minimum = field(row, "Minimum", width=10); self.minimum.insert(0, "3")
        self.maximum = field(row, "Maximum", width=10); self.maximum.insert(0, "10")
        self.eval_words = field(row, "Words per topic", width=12); self.eval_words.insert(0, "10")
        ttk.Button(row, text="Evaluate models", style="Secondary.TButton", command=self.evaluate).pack(side="right", pady=(20, 0))
        extraction = self.card(); ttk.Label(extraction, text="2. Extract topics", style="CardTitle.TLabel").pack(anchor="w", pady=(0, 14))
        row2 = ttk.Frame(extraction, style="Card.TFrame"); row2.pack(fill="x")
        self.topic_count = field(row2, "Number of topics", width=14); self.topic_count.insert(0, "5")
        self.topic_words = field(row2, "Keywords per topic", width=16); self.topic_words.insert(0, "10")
        ttk.Button(row2, text="Extract topics", style="Primary.TButton", command=self.extract).pack(side="right", pady=(20, 0))
        distribution = self.card(); ttk.Label(distribution, text="3. Yearly distribution", style="CardTitle.TLabel").pack(anchor="w")
        ttk.Label(distribution, text="Available after topics have been extracted for the selected dataset.", style="Muted.Card.TLabel").pack(anchor="w", pady=(3, 12))
        row3 = ttk.Frame(distribution, style="Card.TFrame"); row3.pack(fill="x")
        self.mode = ttk.Combobox(row3, values=["Share", "Count"], state="readonly", width=18); self.mode.set("Share"); self.mode.pack(side="left")
        ttk.Button(row3, text="Create yearly figure", style="Secondary.TButton", command=self.distribute).pack(side="left", padx=12)

    def _ints(self, *entries, minimum=1):
        values = tuple(int(entry.get()) for entry in entries)
        if any(value < minimum for value in values): raise ValueError
        return values

    def evaluate(self) -> None:
        try:
            low, high, words = self._ints(self.minimum, self.maximum, self.eval_words, minimum=2)
            if low > high or high > 30: raise ValueError
        except ValueError:
            messagebox.showwarning("Invalid settings", "Use a topic range from 2 to 30, with minimum ≤ maximum.", parent=self); return
        self.app.run_task(lambda: self.app.get_analyzer().coherence_scores(low, high, words), success=lambda result: show_plot(self.app, result, "Topic model evaluation"), busy="Evaluating topic models…")

    def extract(self) -> None:
        try:
            topics, words = self._ints(self.topic_count, self.topic_words, minimum=2)
            if topics > 30 or words > 50: raise ValueError
        except ValueError:
            messagebox.showwarning("Invalid settings", "Use 2–30 topics and 2–50 keywords.", parent=self); return
        self.app.run_task(lambda: self.app.get_analyzer().extract_topics(topics, words), success=self._show_topics, busy="Extracting topics…")

    def _show_topics(self, result) -> None:
        text, _frame = result
        win = tk.Toplevel(self); win.title("Extracted Topics"); win.geometry("720x520"); win.configure(bg=COLORS["bg"])
        ttk.Label(win, text="Extracted Topics", style="Title.TLabel").pack(anchor="w", padx=10, pady=(10, 5))
        widget = ScrolledText(win, wrap="word", font=("Courier", 10), relief="sunken", borderwidth=2, padx=8, pady=8)
        widget.pack(fill="both", expand=True, padx=10, pady=(0, 6)); widget.insert("1.0", text); widget.configure(state="disabled")
        ttk.Button(win, text="Close", command=win.destroy).pack(side="right", padx=10, pady=(0, 10))

    def distribute(self) -> None:
        self.app.run_task(lambda: self.app.get_analyzer().yearly_topic_distribution(self.mode.get()), success=lambda result: show_plot(self.app, result, "Yearly topic distribution"), busy="Creating yearly distribution…")


def show_plot(parent: App, result: AnalysisResult, title: str) -> None:
    win = tk.Toplevel(parent); win.title(title); win.geometry("900x680"); win.configure(bg=COLORS["bg"])
    heading = tk.Frame(win, bg=COLORS["bg"], relief="groove", borderwidth=2); heading.pack(fill="x", padx=5, pady=5)
    tk.Label(heading, text=title, bg=COLORS["bg"], fg=COLORS["text"], font=("TkDefaultFont", 11, "bold")).pack(anchor="w", padx=6, pady=(5, 0))
    tk.Label(heading, text=f"CSV: {result.csv_path}\nImage: {result.image_path}", bg=COLORS["bg"], fg=COLORS["text"], justify="left", font=("TkDefaultFont", 8)).pack(anchor="w", padx=6, pady=(2, 5))
    canvas = FigureCanvasTkAgg(result.figure, master=win); canvas.draw(); canvas.get_tk_widget().pack(fill="both", expand=True, padx=5)
    toolbar = NavigationToolbar2Tk(canvas, win, pack_toolbar=False); toolbar.update(); toolbar.pack(fill="x", padx=5, pady=(0, 4))
    ttk.Button(win, text="Close", command=win.destroy).pack(side="right", padx=7, pady=(0, 7))


if __name__ == "__main__":
    App().mainloop()
