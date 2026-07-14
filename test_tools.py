from harness.tools import FileTools


def main():
    tools = FileTools("repos/demo_repo")

    print("FILES:")
    print(tools.list_files())

    print("\ncalculator.py:")
    print(tools.read_file("src/calculator.py"))

    print("\nTEST RESULT:")
    result = tools.run_tests("pytest")
    print(result["passed"])
    print(result["stdout"])
    print(result["stderr"])


if __name__ == "__main__":
    main()