"""Interactive REPL - the user-facing terminal interface."""

import sys
import os
import argparse

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from prompt_toolkit import prompt as pt_prompt
from prompt_toolkit.history import FileHistory

from .agent import Agent
from .llm import LLM
from .config import Config
from .session import save_session, load_session, list_sessions, delete_session
from . import __version__

console = Console()


def _parse_args():
    p = argparse.ArgumentParser(
        prog="nanocoder",
        description="Minimal AI coding agent. Works with any OpenAI-compatible LLM.",
    )
    p.add_argument("-m", "--model", help="Model name (default: $NANOCODER_MODEL or gpt-4o)")
    p.add_argument("--base-url", help="API base URL (default: $OPENAI_BASE_URL)")
    p.add_argument("--api-key", help="API key (default: $OPENAI_API_KEY)")
    p.add_argument("-p", "--prompt", help="One-shot prompt (non-interactive mode)")
    p.add_argument("-r", "--resume", metavar="ID", help="Resume a saved session")
    p.add_argument("-v", "--version", action="version", version=f"%(prog)s {__version__}")
    return p.parse_args()


def main():
    args = _parse_args()
    config = Config.from_env()

    # CLI args override env vars
    if args.model:
        config.model = args.model
    if args.base_url:
        config.base_url = args.base_url
    if args.api_key:
        config.api_key = args.api_key

    if not config.api_key:
        console.print("[red bold]No API key found.[/]")
        console.print(
            "Set one of: OPENAI_API_KEY, DEEPSEEK_API_KEY, or NANOCODER_API_KEY\n"
            "\nExamples:\n"
            "  # OpenAI\n"
            "  export OPENAI_API_KEY=sk-...\n"
            "\n"
            "  # DeepSeek\n"
            "  export OPENAI_API_KEY=sk-... OPENAI_BASE_URL=https://api.deepseek.com\n"
            "\n"
            "  # Ollama (local)\n"
            "  export OPENAI_API_KEY=ollama OPENAI_BASE_URL=http://localhost:11434/v1 NANOCODER_MODEL=qwen2.5-coder\n"
        )
        sys.exit(1)

    llm = LLM(
        model=config.model,
        api_key=config.api_key,
        base_url=config.base_url,
        temperature=config.temperature,
        max_tokens=config.max_tokens,
    )
    agent = Agent(llm=llm, max_context_tokens=config.max_context_tokens)

    # resume saved session
    if args.resume:
        loaded = load_session(args.resume)
        if loaded:
            agent.messages, loaded_model = loaded
            # restore the model from the saved session unless overridden by CLI
            if not args.model:
                agent.llm.model = loaded_model
                config.model = loaded_model
            console.print(f"[green]Resumed session: {args.resume} (model: {agent.llm.model})[/green]")
        else:
            console.print(f"[red]Session '{args.resume}' not found.[/red]")
            sys.exit(1)

    # one-shot mode
    if args.prompt:
        _run_once(agent, args.prompt)
        return

    # interactive REPL
    _repl(agent, config, args.resume)


def _run_once(agent: Agent, prompt: str):
    """Non-interactive: run one prompt and exit."""
    def on_token(tok):
        print(tok, end="", flush=True)

    def on_tool(name, kwargs):
        console.print(f"\n[dim]> {name}({_brief(kwargs)})[/dim]")

    def on_reasoning(reasoning):
        for chunk in reasoning:
            console.print(f"[dim]{chunk['text']}[/dim]", end="")

    agent.chat(prompt, on_token=on_token, on_tool=on_tool, on_reasoning=on_reasoning)
    print()


def _repl(agent: Agent, config: Config, initial_session: str | None = None):
    """Interactive read-eval-print loop."""
    console.print(Panel(
        f"[bold]NanoCoder[/bold] v{__version__}\n"
        f"Model: [cyan]{config.model}[/cyan]"
        + (f"  Base: [dim]{config.base_url}[/dim]" if config.base_url else "")
        + "\nType [bold]/help[/bold] for commands, [bold]Ctrl+C[/bold] to cancel, [bold]quit[/bold] to exit.",
        border_style="blue",
    ))

    hist_path = os.path.expanduser("~/.nanocoder_history")
    history = FileHistory(hist_path)

    current_session = initial_session  # tracks the active session name for /save

    while True:
        try:
            user_input = pt_prompt("You > ", history=history).strip()
        except (EOFError, KeyboardInterrupt):
            _show_tokens(agent)
            console.print("Bye!")
            break

        if not user_input:
            continue

        # built-in commands
        if user_input.lower() in ("quit", "exit", "/quit", "/exit"):
            break
        if user_input == "/help":
            _show_help()
            continue
        if user_input == "/reset":
            agent.reset()
            console.print("[yellow]Conversation reset.[/yellow]")
            continue
        if user_input == "/rewind":
            agent.rewind()
            console.print("[yellow]Conversation rewind.[/yellow]")
            continue
        if user_input == "/retry":
            _do_retry(agent)
            continue
        if user_input == "/md":
            _show_last_md(agent)
            continue
        if user_input == "/tokens":
            _show_tokens(agent)
            continue
        if user_input.startswith("/model"):
            new_model = user_input[7:].strip()
            if new_model:
                agent.llm.model = new_model
                config.model = new_model
                console.print(f"Switched to [cyan]{new_model}[/cyan]")
            else:
                console.print(f"[yellow]Current model: [cyan]{config.model}[cyan][/yellow]")
            continue
        if user_input == "/compact":
            from .context import estimate_tokens
            before = estimate_tokens(agent.messages)
            compressed = agent.context.maybe_compress(agent.messages, agent.llm)
            after = estimate_tokens(agent.messages)
            if compressed:
                console.print(f"[green]Compressed: {before} → {after} tokens ({len(agent.messages)} messages)[/green]")
            else:
                console.print(f"[dim]Nothing to compress ({before} tokens, {len(agent.messages)} messages)[/dim]")
            continue
        if user_input.startswith("/save"):
            name = user_input[6:].strip() or current_session
            sid = save_session(agent.messages, config.model, name)
            current_session = sid
            console.print(f"[green]Session saved: {sid}[/green]")
            console.print(f"Resume with: nanocoder -r {sid}")
            continue
        if user_input.startswith("/load"):
            resume = user_input[6:].strip()
            if resume:
                loaded = load_session(resume)
                if loaded:
                    agent.messages, loaded_model = loaded
                    agent.llm.model = loaded_model
                    config.model = loaded_model
                    current_session = resume
                    console.print(f"[green]Resumed session: {resume}[/green]")
                    _show_last_md(agent)
                else:
                    console.print(f"[red]Session '{resume}' not found.[/red]")
            continue
        if user_input == "/sessions":
            sessions = list_sessions()
            if not sessions:
                console.print("[dim]No saved sessions.[/dim]")
            else:
                for s in sessions:
                    console.print(f"  [cyan]{s['id']}[/cyan] ({s['model']}, {s['saved_at']}) {s['preview']}")
            continue
        if user_input.startswith("/delete"):
            sid = user_input[8:].strip()
            if sid:
                if delete_session(sid):
                    console.print(f"[green]Session '{sid}' deleted.[/green]")
                else:
                    console.print(f"[red]Session '{sid}' not found.[/red]")
            else:
                console.print("[yellow]Usage: /delete <session_id>[/yellow]")
            continue

        # call the agent
        _run_agent(agent, user_input, retry=False)


def _run_agent(agent: Agent, user_input: str, retry: bool = False):
    """Run the agent with streaming callbacks and error handling."""
    streamed = False
    reasoning_streamed = False

    def on_token(tok):
        nonlocal reasoning_streamed, streamed
        if reasoning_streamed:
            reasoning_streamed = False
            print()
        streamed = True
        print(tok, end="", flush=True)

    def on_tool(name, kwargs):
        console.print(f"\n[dim]> {name}({_brief(kwargs)})[/dim]")

    def on_reasoning(reasoning):
        nonlocal reasoning_streamed, streamed
        reasoning_streamed = True
        for chunk in reasoning:
            console.print(f"[dim]{chunk['text']}[/dim]", end="")

    try:
        if retry:
            response = agent.retry(on_token=on_token, on_tool=on_tool, on_reasoning=on_reasoning)
        else:
            response = agent.chat(user_input, on_token=on_token, on_tool=on_tool, on_reasoning=on_reasoning)
        if reasoning_streamed:
            print()
        if streamed:
            print()
        else:
            console.print(Markdown(response))
    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted.[/yellow]")
    except Exception as e:
        console.print(f"\n[red]Error: {e}[/red]")


def _do_retry(agent: Agent):
    """Re-run the LLM with the current message history."""
    if not agent.messages:
        console.print("[yellow]Nothing to retry.[/yellow]")
        return
    _run_agent(agent, "", retry=True)


def _show_last_md(agent: Agent):
    """Show the last assistant message rendered as Markdown."""
    for m in reversed(agent.messages):
        if m.get("role") == "assistant" and m.get("content"):
            console.print(Markdown(m["content"]))
            return
    console.print("[dim]No assistant messages yet.[/dim]")


def _show_tokens(agent: Agent):
    """Print token usage."""
    p = agent.llm.total_prompt_tokens
    c = agent.llm.total_completion_tokens
    console.print(f"Tokens used this session: [cyan]{p}[/cyan] prompt + [cyan]{c}[/cyan] completion = [bold]{p+c}[/bold] total")


def _show_help():
    console.print(Panel(
        "[bold]Commands:[/bold]\n"
        "  /help          Show this help\n"
        "  /reset         Clear conversation history\n"
        "  /rewind        Rewind before last userinput\n"
        "  /retry         Retry the last prompt (after server errors)\n"
        "  /md            Show last assistant message as Markdown\n"
        "  /model <name>  Switch model mid-conversation\n"
        "  /tokens        Show token usage\n"
        "  /compact       Compress conversation context\n"
        "  /save [name]   Save session to disk\n"
        "  /sessions      List saved sessions\n"
        "  /delete <id>   Delete a saved session\n"
        "  quit           Exit NanoCoder",
        title="NanoCoder Help",
        border_style="dim",
    ))


def _brief(kwargs: dict, maxlen: int = 80) -> str:
    s = ", ".join(f"{k}={repr(v)[:40]}" for k, v in kwargs.items())
    return s[:maxlen] + ("..." if len(s) > maxlen else "")
