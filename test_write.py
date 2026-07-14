from harness.tools import FileTools


def main():
    tools = FileTools("repos/demo_repo")

    tools.write_file(
        "src/calculator.py",
        "def add(a, b):\n    return a + b\n"
    )

    result = tools.run_tests("pytest")

    print("PASSED:", result["passed"])
    print(result["stdout"])
    print(result["stderr"])


if __name__ == "__main__":
    main()