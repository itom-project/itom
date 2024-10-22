import os
from typing import Dict, List


def count_lines_of_code(directory: str, extensions: Dict[str, str]) -> Dict[str, int]:
    """
    Count the lines of code for each file extension in the specified directory.

    Args:
        directory (str): The directory where the project is located.
        extensions (Dict[str, str]): A dictionary mapping file extensions to their programming languages.

    Returns:
        Dict[str, int]: A dictionary mapping file extensions to the total lines of code counted.
    """
    totalLines = {ext: 0 for ext in extensions}  # Initialize with 0 for each extension
    for ext, language in extensions.items():
        total = 0
        for dirpath, _, filenames in os.walk(directory):
            for filename in filenames:
                if filename.endswith(ext):
                    file_path = os.path.join(dirpath, filename)
                    try:
                        with open(file_path, encoding="utf-8") as file:
                            lines = file.readlines()
                            lines_count = len(lines)
                            total += lines_count
                    except (UnicodeDecodeError, FileNotFoundError):
                        print(f"Could not read file: {file_path}")
        totalLines[ext] = total
    return totalLines


def sum_lines_of_code_for_multiple_directories(
    directories: List[str], extensions: Dict[str, str]
) -> Dict[str, Dict[str, int]]:
    """
    Count the lines of code for each file extension across multiple project directories.

    Args:
        directories (List[str]): A list of directories representing different projects.
        extensions (Dict[str, str]): A dictionary mapping file extensions to their programming languages.

    Returns:
        Dict[str, Dict[str, int]]: A dictionary with project names as keys and
            another dictionary mapping file extensions to their total lines of code in each project.
    """
    all_project_lines = {}
    total_lines_per_extension = {ext: 0 for ext in extensions}

    for directory in directories:
        project_name = os.path.basename(directory)
        lines_in_project = count_lines_of_code(directory, extensions)
        all_project_lines[project_name] = lines_in_project

        # Sum up total lines across all projects for each extension
        for ext, lines in lines_in_project.items():
            total_lines_per_extension[ext] += lines

    all_project_lines["total"] = total_lines_per_extension
    return all_project_lines


def generate_markdown_table_for_multiple_projects(
    project_lines: Dict[str, Dict[str, int]], extensions: Dict[str, str]
) -> str:
    """
    Generate a Markdown-formatted table showing lines of code for each file extension across multiple projects.

    Args:
        project_lines (Dict[str, Dict[str, int]]): A dictionary mapping project names to
            another dictionary mapping file extensions to the total lines of code.
        extensions (Dict[str, str]): A dictionary mapping file extensions to their programming languages.

    Returns:
        str: A Markdown-formatted table showing lines of code across multiple projects and their totals.
    """
    # Create Markdown table header
    header_projects = " | ".join(project_lines.keys())
    markdown_table = f"| File Extension | Language         | {header_projects} |\n"

    # Create separator row
    separator = f"|{'-' * 15}|{'-' * 18}|{'-' * (len(header_projects) + 4)}|\n"
    markdown_table += separator

    # Populate table rows
    for ext, language in extensions.items():
        row = f"| {ext.ljust(13)} | {language.ljust(16)} "
        for project in project_lines:
            row += f"| {str(project_lines[project].get(ext, 0)).rjust(6)} "
        row += "|\n"
        markdown_table += row

    return markdown_table


if __name__ == "__main__":
    # Define file extensions and their corresponding programming languages
    extensions = {
        ".py": "Python",
        ".cpp": "C++",
        ".h": "C++",
        ".c": "C",
        ".cmake": "CMake",
        ".txt": "Text",
        ".rst": "RestructuredText",
        ".css": "CSS",
    }

    # Define the list of project directories
    project_directories = [
        r"C:\itom\sources\itom",  # itom core
        r"C:\itom\sources\plugins",  # plugins
        r"C:\itom\sources\designerPlugins",  # designer plugins
    ]

    # Count lines of code for all project directories
    total_lines = sum_lines_of_code_for_multiple_directories(project_directories, extensions)

    # Generate and print Markdown table for all projects
    markdown_output = generate_markdown_table_for_multiple_projects(total_lines, extensions)
    print(markdown_output)
