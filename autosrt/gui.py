"""Interface gráfica do AutoSRT.

Mantém a aparência e os controles do aplicativo original e acrescenta o menu
de sincronia. A lógica fica toda nos outros módulos: aqui só há montagem de
widgets e marshalling das callbacks das threads de trabalho para a thread do
Tk, via ``root.after``.
"""

import os
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog

from . import srt_io, sync
from .language import language_name
from .pipeline import translate_file
from .translate import TranslationCancelled

DARK = {
    "bg": "#2C2C2C", "fg": "white", "entry_bg": "white", "entry_fg": "black",
    "status": "lightgray", "progress": "#66FFFF",
}
LIGHT = {
    "bg": "white", "fg": "black", "entry_bg": "white", "entry_fg": "black",
    "status": "black", "progress": "blue",
}


class AutoSRTApp:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("AutoSRT - O Melhor Tradutor de Legendas")
        self.root.geometry("820x480")
        self.root.configure(bg=DARK["bg"])
        self.root.resizable(True, True)

        self.cancel_event = None
        self.worker = None

        self._build_menu()
        self._build_widgets()
        self.apply_theme(DARK)

    # ------------------------------------------------------------------ UI

    def _build_menu(self):
        menu_bar = tk.Menu(self.root)

        file_menu = tk.Menu(menu_bar, tearoff=0)
        file_menu.add_command(label="Abrir", command=self.select_file)
        file_menu.add_separator()
        file_menu.add_command(label="Sair", command=self.root.quit)
        menu_bar.add_cascade(label="Arquivo", menu=file_menu)

        sync_menu = tk.Menu(menu_bar, tearoff=0)
        sync_menu.add_command(label="Deslocamento fixo...",
                              command=self.sync_shift)
        sync_menu.add_command(label="Ajuste de dois pontos...",
                              command=self.sync_two_point)
        sync_menu.add_separator()
        sync_menu.add_command(label="Automático (vídeo ou legenda)...",
                              command=self.sync_auto)
        menu_bar.add_cascade(label="Sincronia", menu=sync_menu)

        theme_menu = tk.Menu(menu_bar, tearoff=0)
        theme_menu.add_command(label="Dark", command=lambda: self.apply_theme(DARK))
        theme_menu.add_command(label="Light", command=lambda: self.apply_theme(LIGHT))
        menu_bar.add_cascade(label="Tema", menu=theme_menu)

        help_menu = tk.Menu(menu_bar, tearoff=0)
        help_menu.add_command(label="Sobre", command=self.show_about)
        help_menu.add_command(label="Instruções", command=self.show_instructions)
        menu_bar.add_cascade(label="Ajuda", menu=help_menu)

        self.root.config(menu=menu_bar)

    def _build_widgets(self):
        self.top_frame = tk.Frame(self.root, bg=DARK["bg"], padx=10, pady=10)
        self.top_frame.pack(fill="both", expand=True)

        self.btn_select = tk.Button(
            self.top_frame, text="📁 ESCOLHER ARQUIVO", command=self.select_file,
            font=("Arial", 14, "bold"), bd=0, relief="flat",
            highlightthickness=0, bg="#0d6efd", fg="white",
            activebackground="#0b5ed7")
        self.btn_select.pack(anchor="w", pady=5)

        self.label_arquivo = tk.Label(
            self.top_frame, text="Arquivo:", font=("Arial", 11, "bold"),
            fg=DARK["fg"], bg=DARK["bg"])
        self.label_arquivo.pack(anchor="w")

        self.entry_file_path = tk.Entry(self.top_frame, font=("Arial", 10), bd=2)
        self.entry_file_path.pack(pady=5, fill="x")

        self.status_label = tk.Label(
            self.top_frame, text="", font=("Arial", 10, "bold"),
            fg=DARK["status"], bg=DARK["bg"])
        self.status_label.pack(pady=10)

        self.progress_label = tk.Label(
            self.top_frame, text="", font=("Arial", 10),
            fg=DARK["progress"], bg=DARK["bg"])
        self.progress_label.pack()

        self.bottom_frame = tk.Frame(self.root, bg=DARK["bg"], padx=10, pady=10)
        self.bottom_frame.pack(side="bottom", fill="x")

        self.btn_translate = tk.Button(
            self.bottom_frame, text="▶ TRADUZIR", command=self.start_translation,
            font=("Arial", 14, "bold"), bd=0, relief="flat",
            highlightthickness=0, bg="#198754", fg="white",
            activebackground="#157347")
        self.btn_translate.pack(side="left", padx=10)

        self.btn_cancel = tk.Button(
            self.bottom_frame, text="⛔ CANCELAR", command=self.cancel_translation,
            font=("Arial", 14, "bold"), bd=0, relief="flat",
            highlightthickness=0, bg="#dc3545", fg="white",
            activebackground="#bb2d3b")
        self.btn_cancel.pack(side="left", padx=10)

    def apply_theme(self, theme):
        self.root.configure(bg=theme["bg"])
        self.top_frame.configure(bg=theme["bg"])
        self.bottom_frame.configure(bg=theme["bg"])
        self.label_arquivo.configure(bg=theme["bg"], fg=theme["fg"])
        self.entry_file_path.configure(bg=theme["entry_bg"], fg=theme["entry_fg"])
        self.status_label.configure(bg=theme["bg"], fg=theme["status"])
        self.progress_label.configure(bg=theme["bg"], fg=theme["progress"])

    # ------------------------------------------------------------- helpers

    def current_file(self):
        """Arquivo selecionado, ou ``None`` com aviso já exibido."""
        path = self.entry_file_path.get().strip()
        if not path or not os.path.exists(path):
            messagebox.showerror("Erro", "Escolha um arquivo .srt existente.")
            return None
        return path

    def set_status(self, text, color=None):
        self.status_label.config(text=text)
        if color:
            self.status_label.config(fg=color)

    def set_progress(self, text):
        self.progress_label.config(text=text)

    def select_file(self):
        path = filedialog.askopenfilename(filetypes=[("Legendas SRT", "*.srt")])
        if path:
            self.entry_file_path.delete(0, tk.END)
            self.entry_file_path.insert(0, path)

    # ----------------------------------------------------------- tradução

    def start_translation(self):
        if self.worker and self.worker.is_alive():
            messagebox.showinfo("Em andamento", "Já existe uma tradução rodando.")
            return

        path = self.current_file()
        if not path:
            return

        self.cancel_event = threading.Event()
        self.set_status("⏳ Preparando...", "#FFFF66")
        self.set_progress("")

        self.worker = threading.Thread(
            target=self._run_translation, args=(path,), daemon=True)
        self.worker.start()

    def _run_translation(self, path):
        """Roda na thread de trabalho. Toda atualização de tela passa por
        ``root.after``, porque o Tk só aceita chamadas da thread principal."""
        def status(message):
            self.root.after(0, lambda: self.set_status(f"⏳ {message}", "#FFFF66"))

        def progress(done, total):
            self.root.after(
                0, lambda: self.set_progress(f"{done} / {total} linhas processadas"))

        try:
            result = translate_file(
                path, progress=progress, status=status,
                cancel_event=self.cancel_event)
        except TranslationCancelled:
            self.root.after(0, self._on_cancelled)
        except Exception as exc:
            message = str(exc)
            self.root.after(0, lambda: self._on_error(message))
        else:
            self.root.after(0, lambda: self._on_finished(result))

    def cancel_translation(self):
        if self.cancel_event and self.worker and self.worker.is_alive():
            self.cancel_event.set()
            self.set_status("⚠️ Cancelando...", "#FF6666")
        else:
            self.set_status("Nada em andamento.", "#FF6666")

    def _on_cancelled(self):
        self.set_status("⚠️ Tradução cancelada. O arquivo não foi alterado.",
                        "#FF6666")
        self.set_progress("")

    def _on_error(self, message):
        self.set_status("Erro na tradução.", "#FF6666")
        self.set_progress("")
        messagebox.showerror("Erro", message)

    def _on_finished(self, result):
        self.set_status("Tradução concluída!", "#80FF80")
        self.set_progress("")

        linhas = [
            "Tradução concluída!",
            f"Linhas traduzidas: {result.translated} de {result.total}",
            f"Idioma detectado: {language_name(result.detected_lang)}",
        ]
        if result.failure_count:
            linhas.append(
                f"\n{result.failure_count} linha(s) não puderam ser traduzidas "
                "e ficaram no idioma original.")
        if result.backup_path:
            linhas.append(f"\nBackup: {os.path.basename(result.backup_path)}")
        messagebox.showinfo("Sucesso", "\n".join(linhas))

    # ---------------------------------------------------------- sincronia

    def _apply_and_save(self, path, mutate, description):
        """Carrega, aplica uma mudança de tempos e grava, com backup."""
        try:
            cues = srt_io.load_cues(path)
            if not cues:
                messagebox.showerror("Erro", "O arquivo está vazio.")
                return
            mutate(cues)
            backup = srt_io.backup_path(path)
            if not os.path.exists(backup):
                import shutil as _shutil
                _shutil.copy(path, backup)
            srt_io.save_cues(cues, path)
        except (sync.SyncError, OSError, ValueError) as exc:
            messagebox.showerror("Erro", str(exc))
            return

        self.set_status(f"Sincronia aplicada: {description}", "#80FF80")
        messagebox.showinfo("Sincronia", f"{description}\nArquivo atualizado.")

    def sync_shift(self):
        path = self.current_file()
        if not path:
            return
        answer = simpledialog.askstring(
            "Deslocamento fixo",
            "Quanto deslocar, em segundos?\n"
            "Positivo atrasa a legenda, negativo adianta.\n"
            "Exemplo: -2.5")
        if answer is None:
            return
        try:
            seconds = float(answer.replace(",", "."))
        except ValueError:
            messagebox.showerror("Erro", f"Valor inválido: {answer!r}")
            return

        milliseconds = round(seconds * 1000)
        self._apply_and_save(
            path, lambda cues: sync.shift_cues(cues, milliseconds),
            f"deslocamento de {seconds:+.3f} s")

    def sync_two_point(self):
        path = self.current_file()
        if not path:
            return
        try:
            cues = srt_io.load_cues(path)
        except (OSError, ValueError) as exc:
            messagebox.showerror("Erro", str(exc))
            return
        if not cues:
            messagebox.showerror("Erro", "O arquivo está vazio.")
            return

        first, last = cues[0], cues[-1]
        prompts = [
            ("Primeira legenda",
             f"A primeira legenda está em {sync.format_timecode(first.start)}.\n"
             "Em que tempo ela deveria estar?"),
            ("Última legenda",
             f"A última legenda está em {sync.format_timecode(last.start)}.\n"
             "Em que tempo ela deveria estar?"),
        ]
        targets = []
        for title, message in prompts:
            answer = simpledialog.askstring(title, message)
            if answer is None:
                return
            try:
                targets.append(sync.parse_timecode(answer))
            except sync.SyncError as exc:
                messagebox.showerror("Erro", str(exc))
                return

        def mutate(loaded):
            sync.linear_sync(loaded, first.start, targets[0],
                             last.start, targets[1])

        self._apply_and_save(path, mutate, "ajuste linear de dois pontos")

    def sync_auto(self):
        path = self.current_file()
        if not path:
            return

        if not sync.ffsubsync_available():
            messagebox.showerror(
                "ffsubsync ausente",
                "A sincronia automática precisa do ffsubsync.\n\n"
                "Instale com:  pip install ffsubsync")
            return

        reference = filedialog.askopenfilename(
            title="Escolha o vídeo ou uma legenda já sincronizada",
            filetypes=[
                ("Vídeo ou legenda", "*.mp4 *.mkv *.avi *.mov *.srt *.ass *.ssa"),
                ("Todos os arquivos", "*.*"),
            ])
        if not reference:
            return

        self.set_status("⏳ Sincronizando... isso pode demorar.", "#FFFF66")
        threading.Thread(target=self._run_autosync,
                         args=(path, reference), daemon=True).start()

    def _run_autosync(self, path, reference):
        try:
            backup = srt_io.backup_path(path)
            if not os.path.exists(backup):
                import shutil as _shutil
                _shutil.copy(path, backup)
            sync.autosync(path, reference, path)
        except (sync.SyncError, OSError) as exc:
            message = str(exc)
            self.root.after(0, lambda: self._on_error(message))
            return

        self.root.after(0, lambda: self.set_status(
            "Sincronia automática concluída!", "#80FF80"))
        self.root.after(0, lambda: messagebox.showinfo(
            "Sincronia", "Legenda sincronizada com sucesso."))

    # --------------------------------------------------------------- ajuda

    def show_about(self):
        messagebox.showinfo(
            "Sobre",
            "AutoSRT\n"
            "Tradução automática de legendas SRT.\n\n"
            "Tradução por Google Translate (deep-translator).")

    def show_instructions(self):
        messagebox.showinfo("Instruções", (
            "Instruções de uso:\n\n"
            "1. Clique em 'Escolher Arquivo' e selecione um .srt.\n"
            "2. Se a legenda estiver fora de sincronia, use o menu Sincronia "
            "ANTES de traduzir.\n"
            "3. Clique em 'Traduzir'.\n"
            "4. O progresso aparece como '100 / 500 linhas processadas'.\n"
            "5. O arquivo original é sobrescrito, e um backup "
            "(_backup.srt) é criado automaticamente.\n\n"
            "Cancelar interrompe a tradução e não altera o arquivo."))

    def run(self):
        self.root.mainloop()


def main():
    AutoSRTApp().run()
