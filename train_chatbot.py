import os
import time
from chatbot import Chatbot

def train(chatbot, journal_dir, iterations=5, delay=0.5):
    files = sorted([f for f in os.listdir(journal_dir) if f.endswith(".rpl")])
    total_files = len(files)
    print(f"Training on {total_files} scrolls for {iterations} iterations...")

    for it in range(iterations):
        print(f"Iteration {it + 1}/{iterations}...")
        chatbot.memory.clear()
        for idx, fname in enumerate(files):
            path = os.path.join(journal_dir, fname)
            with open(path, "r", encoding="utf-8") as f:
                content = f.read().strip()
                # "Learn" by adding content to memory (simulate)
                chatbot.memory.append(content)
            print(f"  [{idx + 1}/{total_files}] Loaded {fname}")
            time.sleep(delay)  # simulate processing delay
        print("Memory size after iteration:", len(chatbot.memory))
        print("-" * 40)

    print("Training complete.")

if __name__ == "__main__":
    journal_path = os.path.expanduser("~/.gonzo_journal")
    bot = Chatbot(journal_dir=journal_path)
    train(bot, journal_path, iterations=10, delay=0.1)
