import os
import time
import pysrt
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from deep_translator import GoogleTranslator
from langdetect import detect
import threading
from concurrent.futures import ThreadPoolExecutor
import chardet
import re
import shutil
from datetime import datetime

# Nomes de idioma exibidos na interface
LANG_NAME = {
    "en": "Inglês", "es": "Espanhol", "pt": "Português", "cs": "Tcheco",
    "fr": "Francês", "de": "Alemão", "it": "Italiano", "nl": "Holandês",
    "ru": "Russo", "ja": "Japonês", "ar": "Árabe", "tr": "Turco",
    "pl": "Polonês", "sv": "Sueco", "fi": "Finlandês", "da": "Dinamarquês",
    "no": "Norueguês", "he": "Hebraico", "hi": "Hindi", "th": "Tailandês",
    "vi": "Vietnamita", "id": "Indonésio", "ms": "Malaio", "hu": "Húngaro",
    "ro": "Romeno", "bg": "Búlgaro", "uk": "Ucraniano", "hr": "Croata",
    "sr": "Sérvio", "sk": "Eslovaco", "sl": "Esloveno", "lt": "Lituano",
    "mt": "Maltês", "sq": "Albanês", "is": "Islandês", "ga": "Irlandês",
}

# Estado global
cancel_translation = False
progress_count = 0
selected_files = []


def detect_encoding(file_path):
    with open(file_path, "rb") as f:
        result = chardet.detect(f.read())
    return result["encoding"]


def convert_to_utf8_if_needed(input_srt):
    file_encoding = detect_encoding(input_srt)
    if file_encoding and file_encoding.lower() != "utf-8":
        with open(input_srt, "r", encoding=file_encoding) as f:
            content = f.read()
        with open(input_srt, "w", encoding="utf-8") as f:
            f.write(content)
        return True
    return False


def clean_text(text):
    """Remove tags HTML, emojis e caracteres invisíveis do texto."""
    text = re.sub(r"<.*?>", "", text)
    emoji_pattern = re.compile(
        "["
        "\U0001F600-\U0001F64F"
        "\U0001F300-\U0001F5FF"
        "\U0001F680-\U0001F6FF"
        "\U0001F1E0-\U0001F1FF"
        "]+",
        flags=re.UNICODE)
    text = emoji_pattern.sub("", text)
    text = text.replace("﻿", "").replace("​", "")
    return text.strip()


def detect_language(input_srt):
    convert_to_utf8_if_needed(input_srt)
    subs = pysrt.open(input_srt, encoding="utf-8")
    sample_text = " ".join([sub.text for sub in subs[:10]])
    return detect(sample_text)


def cancel_process():
    global cancel_translation
    cancel_translation = True
    status_label.config(text="⚠️ Tradução Cancelada!", fg="#FF6666")
    progress_label.config(text="")


def translate_srt_fast(input_srt, output_srt):
    """Tradução otimizada em lotes, exibindo progresso em texto."""
    global cancel_translation, progress_count

    cancel_translation = False
    progress_count = 0

    convert_to_utf8_if_needed(input_srt)
    subs = pysrt.open(input_srt, encoding="utf-8")
    total = len(subs)
    detected_lang = detect_language(input_srt)
    language_name = LANG_NAME.get(detected_lang, detected_lang.upper())

    root.after(0, lambda: status_label.config(
        text=f"⏳ Traduzindo de {language_name} para Português...", fg="#FFFF66"))
    root.after(0, lambda: progress_label.config(text=f"0 / {total} linhas processadas"))
    batch_size = 200

    def process_batch(start_index):
        global progress_count
        if cancel_translation:
            return
        batch = subs[start_index:start_index + batch_size]
        texts = [clean_text(sub.text.replace("\n", " ")) for sub in batch]
        translator = GoogleTranslator(source=detected_lang, target="pt")
        translated_texts = translator.translate_batch(texts)
        for i, sub in enumerate(batch):
            sub.text = translated_texts[i]
        progress_count += len(batch)
        root.after(0, lambda: progress_label.config(
            text=f"{progress_count} / {total} linhas processadas"))

    with ThreadPoolExecutor() as executor:
        for i in range(0, total, batch_size):
            executor.submit(process_batch, i)
            if cancel_translation:
                return

    subs.save(output_srt, encoding="utf-8")
    root.after(0, lambda: finish_translation(total, detected_lang))


def finish_translation(total, detected_lang):
    status_label.config(
        text="Tradução Concluída!",
        fg="#80FF80",
        font=("Arial", 12, "bold"))

    summary = (
        f"Tradução concluída!\nLinhas traduzidas: {total}"
        f"\nIdioma detectado: {LANG_NAME.get(detected_lang, detected_lang.upper())}")

    progress_label.config(text="")
    messagebox.showinfo("Sucesso", summary)


def start_translation():
    input_file = entry_file_path.get()
    if not os.path.exists(input_file):
        messagebox.showerror("Erro", "Arquivo não encontrado!")
        return

    # Cria um backup do arquivo original antes de sobrescrever
    backup_file = os.path.join(os.path.dirname(input_file),
                               os.path.splitext(os.path.basename(input_file))[0] + "_backup.srt")
    try:
        shutil.copy(input_file, backup_file)
    except Exception as e:
        messagebox.showwarning("Backup", f"Não foi possível criar backup: {e}")

    output_file = input_file
    status_label.config(text="⏳ Detectando idioma...", fg="#FFFF66", font=("Arial", 10, "bold"))
    threading.Thread(target=translate_srt_fast, args=(input_file, output_file), daemon=True).start()


def select_file():
    file_path = filedialog.askopenfilename(filetypes=[("SRT files", "*.srt")])
    if file_path:
        entry_file_path.delete(0, tk.END)
        entry_file_path.insert(0, file_path)


def select_files():
    global selected_files
    files = filedialog.askopenfilenames(filetypes=[("SRT files", "*.srt")])
    if files:
        selected_files = files
        if len(files) == 1:
            entry_file_path.delete(0, tk.END)
            entry_file_path.insert(0, files[0])
        else:
            entry_file_path.delete(0, tk.END)
            entry_file_path.insert(0, f"{len(files)} arquivos selecionados...")


def set_dark_theme():
    root.configure(bg="#2C2C2C")
    top_frame.configure(bg="#2C2C2C")
    bottom_frame.configure(bg="#2C2C2C")
    label_arquivo.configure(bg="#2C2C2C", fg="white")
    entry_file_path.configure(bg="white", fg="black")
    status_label.configure(bg="#2C2C2C", fg="lightgray")
    progress_label.configure(bg="#2C2C2C", fg="#66FFFF")


def set_light_theme():
    root.configure(bg="white")
    top_frame.configure(bg="white")
    bottom_frame.configure(bg="white")
    label_arquivo.configure(bg="white", fg="black")
    entry_file_path.configure(bg="white", fg="black")
    status_label.configure(bg="white", fg="black")
    progress_label.configure(bg="white", fg="blue")


def exit_app():
    root.quit()


def about():
    messagebox.showinfo("Sobre", "AutoSRT\nVersão 1.0\nTradução automática de legendas SRT.")


def instructions():
    msg = (
        "Instruções de Uso:\n"
        "1. Clique em 'Escolher Arquivo' para selecionar um arquivo .srt ou múltiplos arquivos.\n"
        "2. Clique em 'Traduzir' para iniciar a tradução.\n"
        "3. O progresso será exibido em texto (ex: '100 / 500 linhas processadas').\n"
        "4. O arquivo original será alterado com a nova tradução.\n"
        "5. Um backup (_backup.srt) do arquivo original será criado automaticamente."
    )
    messagebox.showinfo("Instruções", msg)


def features():
    msg = (
        "Recursos do Sistema:\n\n"
        "- Detecção automática de idioma.\n"
        "- Tradução otimizada em lotes.\n"
        "- Progresso exibido em texto (leve).\n"
        "- Suporte a seleção múltipla de arquivos.\n"
        "- Opção de tema claro ou escuro.\n"
        "- Backup automático do arquivo original.\n"
        "- Sobrescreve o arquivo SRT com a tradução.\n"
        "- Menu superior com Ajuda, Instruções e Recursos.\n\n"
        "Atualize este menu conforme novas funcionalidades forem adicionadas!"
    )
    messagebox.showinfo("Recursos", msg)


# Janela principal
root = tk.Tk()
root.title("AutoSRT - O Melhor Tradutor de Legendas")
root.geometry("800x450")
root.configure(bg="#2C2C2C")
root.resizable(True, True)

# Menu superior
menu_bar = tk.Menu(root)
file_menu = tk.Menu(menu_bar, tearoff=0)
file_menu.add_command(label="Abrir", command=select_files)
file_menu.add_separator()
file_menu.add_command(label="Sair", command=exit_app)
menu_bar.add_cascade(label="Arquivo", menu=file_menu)

theme_menu = tk.Menu(menu_bar, tearoff=0)
theme_menu.add_command(label="Dark", command=set_dark_theme)
theme_menu.add_command(label="Light", command=set_light_theme)
menu_bar.add_cascade(label="Tema", menu=theme_menu)

help_menu = tk.Menu(menu_bar, tearoff=0)
help_menu.add_command(label="Sobre", command=about)
help_menu.add_command(label="Instruções", command=instructions)
help_menu.add_command(label="Recursos", command=features)
menu_bar.add_cascade(label="Ajuda", menu=help_menu)

root.config(menu=menu_bar)

# Área superior
top_frame = tk.Frame(root, bg="#2C2C2C", padx=10, pady=10)
top_frame.pack(fill="both", expand=True)

btn_select = tk.Button(
    top_frame,
    text="📁 ESCOLHER ARQUIVO",
    command=select_file,
    font=("Arial", 14, "bold"),
    bd=0,
    relief="flat",
    highlightthickness=0,
    bg="#0d6efd",
    fg="white",
    activebackground="#0b5ed7")
btn_select.pack(anchor="w", pady=5)

label_arquivo = tk.Label(
    top_frame,
    text="Arquivo:",
    font=("Arial", 11, "bold"),
    fg="white",
    bg="#2C2C2C")
label_arquivo.pack(anchor="w")

entry_file_path = tk.Entry(top_frame, font=("Arial", 10), bd=2)
entry_file_path.pack(pady=5, fill="x")

status_label = tk.Label(top_frame, text="", font=("Arial", 10, "bold"), fg="lightgray", bg="#2C2C2C")
status_label.pack(pady=10)

progress_label = tk.Label(top_frame, text="", font=("Arial", 10), fg="#66FFFF", bg="#2C2C2C")
progress_label.pack()

# Área inferior
bottom_frame = tk.Frame(root, bg="#2C2C2C", padx=10, pady=10)
bottom_frame.pack(side="bottom", fill="x")

btn_translate = tk.Button(
    bottom_frame,
    text="▶ TRADUZIR",
    command=start_translation,
    font=("Arial", 14, "bold"),
    bd=0,
    relief="flat",
    highlightthickness=0,
    bg="#198754",
    fg="white",
    activebackground="#157347")
btn_translate.pack(side="left", padx=10)

btn_cancel = tk.Button(
    bottom_frame,
    text="⛔ CANCELAR",
    command=cancel_process,
    font=("Arial", 14, "bold"),
    bd=0,
    relief="flat",
    highlightthickness=0,
    bg="#dc3545",
    fg="white",
    activebackground="#bb2d3b")
btn_cancel.pack(side="left", padx=10)

root.mainloop()
