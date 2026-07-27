import argparse
import sys
import signal
import uuid
from .database import AgentDatabase
from .loop import AgentLoop

# Global database variable so the signal handler can close it
db_instance = None

def handle_graceful_shutdown(sig, frame):
    """Catches SIGINT (Ctrl+C) and SIGTERM (kill commands)."""
    print("\nReceived kill signal. Shutting down safely...")
    if db_instance:
        db_instance.close()
        print("Database connection closed safely.")
    sys.exit(0)

def main():
    global db_instance
    
    # Register the signal handlers for graceful shutdown
    signal.signal(signal.SIGINT, handle_graceful_shutdown)
    signal.signal(signal.SIGTERM, handle_graceful_shutdown)

    parser = argparse.ArgumentParser(description="Agentic Developer Runtime")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Command: run
    run_parser = subparsers.add_parser("run", help="Start a new task")
    run_parser.add_argument("--task", type=str, required=True, help="The task for the agent")

    # Command: resume
    resume_parser = subparsers.add_parser("resume", help="Resume a killed task")
    resume_parser.add_argument("run_id", type=str, help="The ID of the run to resume")

    # Command: replay
    replay_parser = subparsers.add_parser("replay", help="Replay a trace")
    replay_parser.add_argument("run_id", type=str, help="The ID of the trace to replay")

    args = parser.parse_args()

    # Dependency Injection Setup
    db_instance = AgentDatabase()
    agent = AgentLoop(db=db_instance)

    try:
        if args.command == "run":
            run_id = str(uuid.uuid4())
            print(f"Starting new run: {run_id}")
            agent.run_task(run_id, args.task)
            
        elif args.command == "resume":
            print(f"Resuming run: {args.run_id}")
            # Logic to load state and resume would go here
            
        elif args.command == "replay":
            print(f"Replaying trace: {args.run_id}")
            # Logic to read JSONL and print trace would go here
            
    except Exception as e:
        print(f"Fatal Error: {e}")
    finally:
        if db_instance:
            db_instance.close()

if __name__ == "__main__":
    main()