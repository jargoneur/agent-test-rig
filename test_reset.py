from harness.tools import FileTools


def main():
    tools = FileTools("repos/demo_repo")

    print("1. Fixing file...")
    tools.write_file(
        "src/calculator.py",
        "def add(a, b):\n    return a + b\n"
    )

    print("2. Tests after fix:")
    result = tools.run_tests("pytest")
    print("PASSED:", result["passed"])

    print("3. Resetting repo...")
    reset = tools.reset_repo()
    print("RESET SUCCESS:", reset["success"])

    print("4. Tests after reset:")
    result = tools.run_tests("pytest")
    print("PASSED:", result["passed"])
    print(result["stdout"])


if __name__ == "__main__":
    main()