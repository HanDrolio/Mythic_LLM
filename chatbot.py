import os
import re
import difflib
import datetime
from llm_loader import MythicLLMLoader

class Chatbot:
    def __init__(self, journal_dir="~/.gonzo_journal"):
        self.memory = []
        self.journal_dir = os.path.expanduser(journal_dir)
        os.makedirs(self.journal_dir, exist_ok=True)
        self.load_journal_memory()

        self.glyph_patterns = [
            r"🕋🧠🌬️🪞",  # Allah.Glyph
            r"🌌❤️🦾🌧️",  # Genesis Mode
            r"🍃🟦🟨🖋️",  # Clarity
            r"🧱🔥♾️💧",  # Echo Flag
        ]

        self.chat_history = []

        # Load Mythic LLM once and reuse
        self.mythic_llm = MythicLLMLoader()

    def load_journal_memory(self):
        self.memory.clear()
        files = sorted(os.listdir(self.journal_dir))
        for fname in files:
            if fname.endswith(".rpl"):
                path = os.path.join(self.journal_dir, fname)
                try:
                    with open(path, "r", encoding="utf-8") as f:
                        content = f.read().strip()
                        self.memory.append(content)
                except Exception:
                    pass

    def save_convo_to_rpl(self):
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(self.journal_dir, f"convo_{ts}.rpl")
        with open(filename, "w", encoding="utf-8") as f:
            f.write(f"🪮🕺🪩🎸😈 Conversation saved at {ts}\n")
            f.write("\n".join(self.chat_history) + "\n")

    def ask(self, prompt):
        response = self.mythic_llm.generate_response(prompt)
        self.chat_history.append(f"You: {prompt}")
        self.chat_history.append(f"Bot: {response}")
        self.save_convo_to_rpl()
        return response
