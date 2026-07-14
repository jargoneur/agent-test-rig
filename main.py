from harness.model_adapter import OllamaModel


def main():
    model = OllamaModel("qwen2.5-coder:3b")

    prompt = """
Return only Python code.

Write a function called add.
It should take two arguments: a and b.
It should return their sum.
"""

    answer = model.generate(prompt)
    print(answer)


if __name__ == "__main__":
    main()