"""Backward-compatible CLI entry point: redirects to main.py"""
from main import run_cli

if __name__ == "__main__":
    run_cli()
