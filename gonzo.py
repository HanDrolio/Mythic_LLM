#!/usr/bin/env python3
"""
Gonzo CLI Journal 🪮🎸🕺
Full-screen terminal edition powered by curses.
Includes modular chatbot mode with Myth.OS glyph awareness
and live journal memory syncing.
"""

import os
import datetime
import itertools
import curses
import textwrap
import difflib
import chatbot  # Modular chatbot logic

JOURNAL_DIR = os.path.expanduser("~/.gonzo_journal")
os.makedirs(JOURNAL_DIR, exist_ok=True)

COLOR_PRISM = ["🟥", "🟧", "🟨", "🟩", "🟦", "🟪", "🟫"]

def prism_line(length=30):
    cycle = itertools.cycle(COLOR_PRISM)
    return "".join(next(cycle) for _ in range(length))

def timestamp():
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def list_journal_files(filter_tag=None):
    files = sorted(os.listdir(JOURNAL_DIR))
    if filter_tag:
        filtered = []
        for f in files:
            path = os.path.join(JOURNAL_DIR, f)
            with open(path, "r", encoding="utf-8") as file:
                if filter_tag in file.read():
                    filtered.append(f)
        return filtered
    return files

def read_entry_text(filename):
    path = os.path.join(JOURNAL_DIR, filename)
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

def save_entry(text, tags, chatbot_instance=None):
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = os.path.join(JOURNAL_DIR, f"entry_{ts}.rpl")
    with open(filename, "w", encoding="utf-8") as f:
        f.write(f"🪮🕺🪩🎸😈 {timestamp()} {' '.join(tags)}\n{text}\n")
    if chatbot_instance:
        chatbot_instance.load_journal_memory()

class GonzoApp:
    def __init__(self, stdscr):
        self.stdscr = stdscr
        curses.curs_set(0)
        self.height, self.width = self.stdscr.getmaxyx()
        self.color_pairs_setup()
        self.current_cmd = ""
        self.status_msg = "Welcome to Gonzo CLI Journal 🎸🪮 - Press F1 for Help"
        self.entries = list_journal_files()
        self.filtered_entries = self.entries
        self.selected_idx = 0
        self.mode = "list"
        self.read_buffer = ""
        self.input_buffer = []
        self.input_prompt = ""
        self.tags_buffer = []
        self.filter_tag = None

        self.chatbot = chatbot.Chatbot()
        self.chat_history = []
        self.chat_input = ""

    def color_pairs_setup(self):
        curses.start_color()
        curses.use_default_colors()
        curses.init_pair(1, curses.COLOR_BLACK, curses.COLOR_CYAN)
        curses.init_pair(2, curses.COLOR_BLACK, curses.COLOR_YELLOW)
        curses.init_pair(3, curses.COLOR_RED, -1)
        curses.init_pair(4, curses.COLOR_GREEN, -1)

    def safe_addstr(self, y, x, string, attr=0):
        max_len = self.width - x - 1
        safe_str = string[:max_len]
        try:
            if attr:
                self.stdscr.addstr(y, x, safe_str, attr)
            else:
                self.stdscr.addstr(y, x, safe_str)
        except curses.error:
            pass

    def draw_header(self):
        header_str = f"Gonzo CLI Journal 🪮🎸 - Mode: {self.mode.upper()}  Entries: {len(self.filtered_entries)}"
        self.safe_addstr(0, 0, header_str, curses.color_pair(1))

    def draw_footer(self):
        footer_str = self.status_msg
        self.safe_addstr(self.height-1, 0, footer_str, curses.color_pair(1))

    def draw_list(self):
        start_y = 2
        max_display = self.height - 4
        total = len(self.filtered_entries)
        if total == 0:
            self.safe_addstr(start_y, 0, "(No entries found)", curses.color_pair(3))
            return

        if self.selected_idx < 0:
            self.selected_idx = 0
        if self.selected_idx >= total:
            self.selected_idx = total - 1

        top_idx = max(0, min(self.selected_idx, total - max_display))
        entries_to_show = self.filtered_entries[top_idx:top_idx + max_display]

        for i, filename in enumerate(entries_to_show):
            y = start_y + i
            attr = curses.A_REVERSE if (top_idx + i) == self.selected_idx else 0
            self.safe_addstr(y, 0, filename, attr)

    def draw_chat(self):
        self.stdscr.clear()
        start_y = 2
        max_lines = self.height - 6  # leave room for header/footer/input

        # Show last chat history lines fitting in window
        chat_to_show = self.chatbot.chat_history[-max_lines:]

        for i, line in enumerate(chat_to_show):
            self.safe_addstr(start_y + i, 0, line)

        # Draw input prompt at bottom
        prompt = "Chat> " + self.chat_input
        self.safe_addstr(self.height - 3, 0, "-" * (self.width - 1), curses.color_pair(2))
        self.safe_addstr(self.height - 2, 0, prompt)

    # Stub methods for read, write, search, help (implement as needed)
    def draw_read(self):
        self.stdscr.clear()
        lines = self.read_buffer.splitlines()
        max_lines = self.height - 4
        for i, line in enumerate(lines[:max_lines]):
            self.safe_addstr(2 + i, 0, line)
        self.safe_addstr(self.height - 2, 0, "Press 'q' or ESC to return")
    
    def draw_write(self):
        self.stdscr.clear()
        for i, line in enumerate(self.input_buffer[-(self.height - 4):]):
            self.safe_addstr(2 + i, 0, line)
        self.safe_addstr(self.height - 2, 0, "Writing mode - Enter text. Blank line to finish.")
    
    def draw_search(self):
        self.stdscr.clear()
        self.safe_addstr(2, 0, "Search: " + self.current_cmd)
        self.safe_addstr(self.height - 2, 0, "Press ENTER to search or ESC to cancel.")
    
    def draw_help(self):
        self.stdscr.clear()
        help_text = [
            "Gonzo CLI Journal Help:",
            "Arrow keys: Navigate",
            "ENTER: Read entry",
            "w: Write new entry",
            "s: Search entries",
            "t: Filter by tag",
            "c: Chat mode",
            "q or ESC: Quit/Back",
            "F1: Show Help",
        ]
        for i, line in enumerate(help_text):
            self.safe_addstr(2 + i, 0, line)
        self.safe_addstr(self.height - 2, 0, "Press any key to return.")

    def input_loop(self):
        while True:
            self.height, self.width = self.stdscr.getmaxyx()
            self.stdscr.clear()
            self.draw_header()
            if self.mode == "list":
                self.draw_list()
            elif self.mode == "read":
                self.draw_read()
            elif self.mode == "write":
                self.draw_write()
            elif self.mode == "search":
                self.draw_search()
            elif self.mode == "help":
                self.draw_help()
            elif self.mode == "chat":
                self.draw_chat()

            self.draw_footer()
            self.stdscr.refresh()

            c = self.stdscr.getch()

            if self.mode == "list":
                if c == curses.KEY_UP and self.selected_idx > 0:
                    self.selected_idx -= 1
                elif c == curses.KEY_DOWN and self.selected_idx < len(self.filtered_entries) - 1:
                    self.selected_idx += 1
                elif c in [10, 13]:
                    if self.filtered_entries:
                        filename = self.filtered_entries[self.selected_idx]
                        content = read_entry_text(filename)
                        if content:
                            self.read_buffer = content
                            self.mode = "read"
                            self.status_msg = f"Reading {filename}"
                elif c in [ord("q"), 27]:
                    confirm = self.confirm_dialog("Quit Gonzo? (y/n)")
                    if confirm:
                        break
                elif c == ord("w"):
                    self.mode = "write"
                    self.input_buffer = []
                    self.tags_buffer = []
                    self.input_prompt = "Enter tags (space-separated): "
                    self.status_msg = "Write your tags and press ENTER"
                    curses.curs_set(1)
                    self.get_tags_input()
                elif c == ord("s"):
                    self.mode = "search"
                    self.current_cmd = ""
                    curses.curs_set(1)
                elif c == ord("l"):
                    self.filtered_entries = list_journal_files()
                    self.selected_idx = 0
                elif c == ord("t"):
                    self.status_msg = "Enter tag to filter by:"
                    curses.curs_set(1)
                    tag = self.get_input_line("Tag: ")
                    if tag.strip():
                        self.filter_tag = tag.strip()
                        self.filtered_entries = list_journal_files(self.filter_tag)
                        self.selected_idx = 0
                        self.status_msg = f"Filtered by tag '{self.filter_tag}'"
                    else:
                        self.filter_tag = None
                        self.filtered_entries = list_journal_files()
                        self.status_msg = "Tag filter cleared."
                    curses.curs_set(0)
                elif c == curses.KEY_F1:
                    self.mode = "help"
                elif c == ord("c"):
                    self.mode = "chat"
                    self.chat_input = ""
                    self.chat_history = []
                    self.status_msg = "Entered chat mode. Type and hit ENTER to chat."
                    curses.curs_set(1)
                else:
                    self.status_msg = "Use arrows to navigate, ENTER to read, w-write, s-search, t-tag filter, c-chat, q-quit, F1-help"

            elif self.mode == "read":
                if c in [27, ord("q")]:
                    self.mode = "list"
                    self.status_msg = "Returned to list."

            elif self.mode == "write":
                finished = self.get_entry_input(c)
                if finished:
                    text = "\n".join(self.input_buffer).strip()
                    tags = self.tags_buffer
                    save_entry(text, tags, chatbot_instance=self.chatbot)  # Live update
                    self.status_msg = f"Entry saved with tags: {' '.join(tags)}"
                    self.mode = "list"
                    curses.curs_set(0)
                    self.filtered_entries = list_journal_files()
                    self.selected_idx = 0

            elif self.mode == "search":
                if c in [10, 13]:
                    if self.current_cmd.strip():
                        self.perform_search(self.current_cmd.strip())
                    else:
                        self.status_msg = "Empty search cancelled."
                    self.mode = "list"
                    curses.curs_set(0)
                elif c == 27:
                    self.mode = "list"
                    self.status_msg = "Search cancelled."
                    curses.curs_set(0)
                elif c in (curses.KEY_BACKSPACE, 127):
                    self.current_cmd = self.current_cmd[:-1]
                elif 32 <= c <= 126:
                    self.current_cmd += chr(c)

            elif self.mode == "chat":
                if c in [27]:
                    self.mode = "list"
                    self.status_msg = "Returned from chat."
                    curses.curs_set(0)
                elif c in [10, 13]:
                    if self.chat_input.strip():
                        self.chat_history.append("You: " + self.chat_input)
                        response = self.chatbot.ask(self.chat_input.strip())
                        self.chat_history.append("Bot: " + response)
                        self.chat_input = ""
                    else:
                        self.status_msg = "Empty message, try again."
                elif c in (curses.KEY_BACKSPACE, 127):
                    self.chat_input = self.chat_input[:-1]
                elif 32 <= c <= 126:
                    self.chat_input += chr(c)

            elif self.mode == "help":
                self.mode = "list"
                self.status_msg = "Returned from help."

    def confirm_dialog(self, prompt):
        self.status_msg = prompt
        self.stdscr.refresh()
        while True:
            c = self.stdscr.getch()
            if c in [ord("y"), ord("Y")]:
                return True
            elif c in [ord("n"), ord("N")]:
                return False

    def get_input_line(self, prompt):
        curses.echo()
        self.safe_addstr(self.height - 2, 0, prompt)
        self.stdscr.clrtoeol()
        self.stdscr.refresh()
        input_str = self.stdscr.getstr(self.height - 2, len(prompt), 60).decode("utf-8")
        curses.noecho()
        return input_str

    def get_tags_input(self):
        tags_line = self.get_input_line(self.input_prompt)
        self.tags_buffer = tags_line.strip().split()
        self.status_msg = "Enter your journal entry now. Press ENTER on blank line to finish."
        self.mode = "write"
        self.input_buffer = []

    def get_entry_input(self, c):
        if c in [10, 13]:
            if self.input_buffer and self.input_buffer[-1] == "":
                return True
            else:
                self.input_buffer.append("")
        elif c == 27:
            self.mode = "list"
            self.status_msg = "Write cancelled."
            return True
        elif c in (curses.KEY_BACKSPACE, 127):
            if self.input_buffer:
                if self.input_buffer[-1]:
                    self.input_buffer[-1] = self.input_buffer[-1][:-1]
                else:
                    self.input_buffer.pop()
        elif 32 <= c <= 126:
            char = chr(c)
            if not self.input_buffer:
                self.input_buffer.append(char)
            else:
                self.input_buffer[-1] += char
        return False

    def perform_search(self, term):
        self.filtered_entries = list_journal_files(term)
        self.selected_idx = 0
        self.status_msg = f"Search results for '{term}': {len(self.filtered_entries)} entries"

def main(stdscr):
    app = GonzoApp(stdscr)
    global gonzo_app_instance
    gonzo_app_instance = app
    app.input_loop()

if __name__ == "__main__":
    gonzo_app_instance = None
    curses.wrapper(main)
