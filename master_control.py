#!/usr/bin/env python3
"""
Master Control Panel for Amazon KDP/FBA Ecosystem

This script provides a unified interface to all three systems:
- FBA Product Research
- Coloring Book Generator
- Puzzle Book Generator
"""

import sys
import subprocess
from pathlib import Path
import argparse


class Colors:
    """Terminal colors for better UI."""
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    END = '\033[0m'
    BOLD = '\033[1m'


def print_header(text):
    """Print a formatted header."""
    print(f"\n{Colors.BOLD}{Colors.CYAN}{'='*70}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.CYAN}{text.center(70)}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.CYAN}{'='*70}{Colors.END}\n")


def print_section(text):
    """Print a section header."""
    print(f"\n{Colors.BOLD}{Colors.GREEN}{text}{Colors.END}")
    print(f"{Colors.GREEN}{'-'*len(text)}{Colors.END}")


def print_option(number, title, description):
    """Print a menu option."""
    print(f"{Colors.BOLD}{number}.{Colors.END} {Colors.YELLOW}{title}{Colors.END}")
    print(f"   {description}")


def run_command(cmd, cwd=None):
    """Run a command and return success status."""
    try:
        result = subprocess.run(
            cmd,
            shell=True,
            cwd=cwd,
            check=True,
            capture_output=False
        )
        return result.returncode == 0
    except subprocess.CalledProcessError as e:
        print(f"{Colors.RED}Error: Command failed with exit code {e.returncode}{Colors.END}")
        return False
    except Exception as e:
        print(f"{Colors.RED}Error: {e}{Colors.END}")
        return False


def show_main_menu():
    """Display the main menu."""
    print_header("AMAZON KDP/FBA ECOSYSTEM - MASTER CONTROL")

    print_section("PROJECT SELECTION")
    print_option(1, "FBA Product Research", "Research products, analyze competition, forecast inventory")
    print_option(2, "Coloring Book Generator", "Create professional adult coloring books for KDP")
    print_option(3, "Puzzle Book Generator", "Generate quality-validated puzzle books for KDP")
    print()
    print_option(4, "Show Status", "Check installation status of all systems")
    print_option(5, "Documentation", "Open ecosystem guide and documentation")
    print_option(0, "Exit", "Exit master control")
    print()


def fba_menu():
    """FBA Product Research menu."""
    print_header("FBA PRODUCT RESEARCH")

    print_section("AVAILABLE ACTIONS")
    print_option(1, "Quick Test", "Test FBA tools installation")
    print_option(2, "Full Demo", "Run complete research demo")
    print_option(3, "Open Documentation", "View FBA setup guide")
    print_option(0, "Back", "Return to main menu")
    print()

    choice = input(f"{Colors.BOLD}Select option: {Colors.END}").strip()

    if choice == "1":
        print("\nRunning FBA quick test...")
        run_command("python3 quick_test.py", cwd="FBA")
    elif choice == "2":
        print("\nRunning FBA full demo...")
        run_command("python3 full_demo.py", cwd="FBA")
    elif choice == "3":
        print("\nOpening FBA documentation...")
        print(f"\nRead: {Path('FBA/README.md').absolute()}")
        print(f"Setup: {Path('FBA/SETUP.md').absolute()}")
    elif choice == "0":
        return
    else:
        print(f"{Colors.RED}Invalid option{Colors.END}")

    input("\nPress Enter to continue...")


def coloring_book_menu():
    """Coloring Book Generator menu."""
    print_header("ADULT COLORING BOOK GENERATOR")

    print_section("AVAILABLE ACTIONS")
    print_option(1, "Quick Test", "Generate 1 test page to verify quality")
    print_option(2, "Create Complete Book", "Generate book with interior + cover + PDF")
    print_option(3, "Batch Generate", "Create multiple books at once")
    print_option(4, "Generate Cover Only", "Create professional book cover")
    print_option(5, "List Themes", "Show all available themes")
    print_option(6, "Open Documentation", "View coloring book guides")
    print_option(0, "Back", "Return to main menu")
    print()

    choice = input(f"{Colors.BOLD}Select option: {Colors.END}").strip()

    if choice == "1":
        print("\nGenerating test page...")
        run_command("python3 test_improved_lineart.py --quick", cwd="coloring-books")

    elif choice == "2":
        print("\n" + Colors.BOLD + "Create Complete Book" + Colors.END)
        print("\nAvailable themes: mandalas, animals, nature, geometric, fantasy, patterns, inspirational")
        theme = input("Enter theme: ").strip()
        pages = input("Enter number of pages (default: 30): ").strip() or "30"

        cmd = f"python3 complete_book_workflow.py --theme {theme} --pages {pages}"
        print(f"\nRunning: {cmd}")
        run_command(cmd, cwd="coloring-books")

    elif choice == "3":
        print("\n" + Colors.BOLD + "Batch Generate Books" + Colors.END)
        print("\n1. All themes (7 books)")
        print("2. Specific themes")
        batch_choice = input("Select option: ").strip()

        if batch_choice == "1":
            pages = input("Pages per book (default: 30): ").strip() or "30"
            cmd = f"python3 batch_generator.py --all-themes --pages {pages}"
            print(f"\nRunning: {cmd}")
            run_command(cmd, cwd="coloring-books")
        elif batch_choice == "2":
            print("\nEnter themes separated by spaces")
            print("(e.g., mandalas animals nature)")
            themes = input("Themes: ").strip()
            pages = input("Pages per book (default: 30): ").strip() or "30"
            cmd = f"python3 batch_generator.py --themes {themes} --pages {pages}"
            print(f"\nRunning: {cmd}")
            run_command(cmd, cwd="coloring-books")

    elif choice == "4":
        print("\nAvailable themes: mandalas, animals, nature, geometric, fantasy, patterns, inspirational")
        theme = input("Enter theme: ").strip()
        cmd = f"python3 cover_generator.py --theme {theme}"
        run_command(cmd, cwd="coloring-books")

    elif choice == "5":
        run_command("python3 coloring_book_generator.py --list-themes", cwd="coloring-books")

    elif choice == "6":
        print("\nDocumentation files:")
        docs = [
            "README.md - Overview",
            "QUICK_REFERENCE.md - Command reference",
            "IMPROVEMENTS.md - Usage guide",
            "SETUP_GUIDE.md - Installation",
            "PROJECT_STATUS.md - Project status"
        ]
        for doc in docs:
            print(f"  • coloring-books/{doc}")

    elif choice == "0":
        return
    else:
        print(f"{Colors.RED}Invalid option{Colors.END}")

    input("\nPress Enter to continue...")


def puzzle_book_menu():
    """Puzzle Book Generator menu."""
    print_header("PUZZLE BOOK GENERATOR")

    print_section("AVAILABLE ACTIONS")
    print_option(1, "Generate One Book", "Generate a single puzzle book")
    print_option(2, "Generate All Niches", "Generate books for all configured niches")
    print_option(3, "Start Daemon", "Start 24/7 automated generation")
    print_option(4, "Check Status", "View generation status")
    print_option(5, "Open Documentation", "View puzzle book guide")
    print_option(0, "Back", "Return to main menu")
    print()

    choice = input(f"{Colors.BOLD}Select option: {Colors.END}").strip()

    if choice == "1":
        print("\nAvailable niches:")
        print("  sudoku_easy, sudoku_medium, sudoku_hard")
        print("  word_search_easy, word_search_medium")
        print("  cryptogram_easy, cryptogram_medium")
        niche = input("Enter niche: ").strip()
        cmd = f"python3 run_daemon.py --niche {niche}"
        run_command(cmd, cwd="kdp-quality-pipeline")

    elif choice == "2":
        print("\nGenerating all niches...")
        run_command("python3 run_daemon.py --once", cwd="kdp-quality-pipeline")

    elif choice == "3":
        print("\nStarting daemon mode (Ctrl+C to stop)...")
        run_command("python3 run_daemon.py", cwd="kdp-quality-pipeline")

    elif choice == "4":
        run_command("python3 run_daemon.py --status", cwd="kdp-quality-pipeline")

    elif choice == "5":
        print(f"\nRead: {Path('kdp-quality-pipeline/README.md').absolute()}")

    elif choice == "0":
        return
    else:
        print(f"{Colors.RED}Invalid option{Colors.END}")

    input("\nPress Enter to continue...")


def show_status():
    """Show installation status of all systems."""
    print_header("SYSTEM STATUS")

    base_dir = Path(__file__).parent

    systems = [
        ("FBA Product Research", "FBA", ["src/sp_api_client.py", "requirements.txt"]),
        ("Coloring Books", "coloring-books", ["coloring_book_generator.py", "requirements.txt"]),
        ("Puzzle Books", "kdp-quality-pipeline", ["run_daemon.py", "config.yaml"])
    ]

    for name, folder, files in systems:
        print_section(name)
        system_path = base_dir / folder

        if not system_path.exists():
            print(f"{Colors.RED}✗ Not found at {system_path}{Colors.END}")
            continue

        print(f"{Colors.GREEN}✓ Installed at {system_path}{Colors.END}")

        # Check key files
        all_found = True
        for file in files:
            file_path = system_path / file
            if file_path.exists():
                print(f"  {Colors.GREEN}✓{Colors.END} {file}")
            else:
                print(f"  {Colors.RED}✗{Colors.END} {file} (missing)")
                all_found = False

        # Check Python dependencies
        req_file = system_path / "requirements.txt"
        if req_file.exists():
            print(f"\n  To install dependencies:")
            print(f"    cd {folder} && pip install -r requirements.txt")

    print()
    input("Press Enter to continue...")


def show_documentation():
    """Show documentation links."""
    print_header("DOCUMENTATION")

    print_section("Ecosystem Overview")
    print(f"  📖 ECOSYSTEM_GUIDE.md - Complete integration guide")
    print(f"  📖 README.md - Main overview")

    print_section("FBA Product Research")
    print(f"  📖 FBA/README.md - Overview")
    print(f"  📖 FBA/SETUP.md - Setup instructions")
    print(f"  📖 FBA/PRODUCT_RESEARCH_TOOLS.md - Tool reference")

    print_section("Coloring Books")
    print(f"  📖 coloring-books/README.md - Overview")
    print(f"  📖 coloring-books/QUICK_REFERENCE.md - Command reference")
    print(f"  📖 coloring-books/IMPROVEMENTS.md - Usage guide")
    print(f"  📖 coloring-books/SETUP_GUIDE.md - Installation")
    print(f"  📖 coloring-books/PROJECT_STATUS.md - Project status")
    print(f"  📖 coloring-books/CHANGELOG.md - Version history")

    print_section("Puzzle Books")
    print(f"  📖 kdp-quality-pipeline/README.md - Overview")
    print(f"  📖 kdp-quality-pipeline/config.yaml - Configuration")

    print()
    input("Press Enter to continue...")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Master control for Amazon KDP/FBA Ecosystem",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--system", choices=["fba", "coloring", "puzzle"],
                       help="Launch directly into a system")
    args = parser.parse_args()

    # Direct launch
    if args.system == "fba":
        fba_menu()
        return
    elif args.system == "coloring":
        coloring_book_menu()
        return
    elif args.system == "puzzle":
        puzzle_book_menu()
        return

    # Interactive menu
    while True:
        show_main_menu()
        choice = input(f"{Colors.BOLD}Select option: {Colors.END}").strip()

        if choice == "1":
            fba_menu()
        elif choice == "2":
            coloring_book_menu()
        elif choice == "3":
            puzzle_book_menu()
        elif choice == "4":
            show_status()
        elif choice == "5":
            show_documentation()
        elif choice == "0":
            print(f"\n{Colors.GREEN}Goodbye!{Colors.END}\n")
            sys.exit(0)
        else:
            print(f"{Colors.RED}Invalid option. Please try again.{Colors.END}")
            input("Press Enter to continue...")


if __name__ == "__main__":
    main()
