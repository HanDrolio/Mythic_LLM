from llm_loader import MythicLLMLoader

def main():
    llm = MythicLLMLoader()
    print("🖋️✨ Mythic LLM CLI test — type 'exit' to quit")
    while True:
        prompt = input("You> ").strip()
        if prompt.lower() in ("exit", "quit"):
            print("👋 Mythic LLM signing off.")
            break
        response = llm.generate_response(prompt)
        print("Bot>", response)

if __name__ == "__main__":
    main()
