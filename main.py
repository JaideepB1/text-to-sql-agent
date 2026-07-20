from SQLAgent import run


def main() -> None:
    print("=" * 60)
    print("Text-to-SQL Agent")
    print("Database: Chinook digital music store")
    print("Type 'help' for examples or 'quit' to exit.")
    print("=" * 60)

    while True:
        question = input("\nQuestion: ").strip()

        if question.lower() == "help":
            print("""
        Example Questions

        - How many customers are in the database?
        - Which artist has the most albums?
        - Which customer spent the most money?
        - Which genre has the longest tracks?
        - List the first five artists.
        """)
            continue

        if question.lower() in {"quit", "exit", "q"}:
            print("Goodbye!")
            break

        if not question:
            continue

        try:
            answer = run(question)

            print("\n" + "=" * 60)
            print("FINAL ANSWER")
            print("=" * 60)
            print(answer)

        except Exception as error:
            print(f"\nAgent error: {error}")


if __name__ == "__main__":
    main()