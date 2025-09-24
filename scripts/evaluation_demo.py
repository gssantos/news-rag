#!/usr/bin/env python
"""End-to-end demo of the evaluation framework."""

import asyncio
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

import mlflow
from rich.console import Console
from rich.table import Table
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker

from app.core.config import settings
from app.models.evaluation import EvaluationType, GoldenDataset
from app.services.evaluation_service import EvaluationService
from app.services.llm_service import get_llm_service

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
console = Console()


class EvaluationDemo:
    """Demonstration of the evaluation framework capabilities."""

    def __init__(self):
        self.engine = create_async_engine(str(settings.DATABASE_URL))
        self.SessionLocal = sessionmaker(
            self.engine, class_=AsyncSession, expire_on_commit=False
        )

    async def run_demo(self):
        """Run the complete evaluation demo."""
        console.print("\n[bold cyan]News RAG Evaluation Framework Demo[/bold cyan]\n")

        async with self.SessionLocal() as db:
            llm_service = get_llm_service()
            eval_service = EvaluationService(db, llm_service)

            # Step 1: Check for golden dataset
            console.print("[yellow]Step 1: Checking for golden dataset...[/yellow]")
            result = await db.execute(
                select(GoldenDataset).where(GoldenDataset.is_active == True)
            )
            dataset = result.scalars().first()

            if not dataset:
                console.print(
                    "[red]No active golden dataset found. Please run init_golden_dataset.py first.[/red]"
                )
                return

            console.print(
                f"[green]✓ Found dataset: {dataset.name} v{dataset.version}[/green]"
            )

            # Step 2: Run retrieval evaluation
            console.print("\n[yellow]Step 2: Running retrieval evaluation...[/yellow]")
            retrieval_run = await eval_service.run_evaluation(
                dataset_id=dataset.id,
                evaluation_type=EvaluationType.RETRIEVAL,
                config={"demo": True, "timestamp": datetime.utcnow().isoformat()},
            )
            self._display_metrics("Retrieval Evaluation", retrieval_run.metrics)

            # Step 3: Run generation evaluation
            console.print("\n[yellow]Step 3: Running generation evaluation...[/yellow]")
            generation_run = await eval_service.run_evaluation(
                dataset_id=dataset.id,
                evaluation_type=EvaluationType.GENERATION,
                config={"demo": True, "timestamp": datetime.utcnow().isoformat()},
            )
            self._display_metrics("Generation Evaluation", generation_run.metrics)

            # Step 4: Run end-to-end evaluation
            console.print(
                "\n[yellow]Step 4: Running end-to-end RAG evaluation...[/yellow]"
            )
            e2e_run = await eval_service.run_evaluation(
                dataset_id=dataset.id,
                evaluation_type=EvaluationType.END_TO_END,
                config={"demo": True, "timestamp": datetime.utcnow().isoformat()},
            )
            self._display_metrics("End-to-End RAG Evaluation", e2e_run.metrics)

            # Step 5: Compare evaluations
            console.print("\n[yellow]Step 5: Comparing evaluation runs...[/yellow]")
            comparison = await eval_service.compare_evaluations(
                [
                    retrieval_run.id,
                    generation_run.id,
                    e2e_run.id,
                ]
            )
            self._display_comparison(comparison)

            # Step 6: Show MLFlow tracking info
            console.print("\n[yellow]Step 6: MLFlow Tracking Information[/yellow]")
            self._display_mlflow_info([retrieval_run, generation_run, e2e_run])

            # Step 7: Get evaluation history
            console.print("\n[yellow]Step 7: Recent Evaluation History[/yellow]")
            history = await eval_service.get_evaluation_history(
                dataset_id=dataset.id,
                limit=5,
            )
            self._display_history(history)

            console.print("\n[bold green]✨ Demo completed successfully![/bold green]")
            console.print(
                f"\n[cyan]View detailed metrics in MLFlow UI: {settings.MLFLOW_TRACKING_URI}[/cyan]"
            )

    def _display_metrics(self, title: str, metrics: dict):
        """Display evaluation metrics in a table."""
        if not metrics:
            console.print(f"[red]No metrics available for {title}[/red]")
            return

        table = Table(title=title, show_header=True, header_style="bold magenta")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", justify="right", style="green")

        for metric, value in metrics.items():
            if isinstance(value, float):
                table.add_row(metric, f"{value:.4f}")
            else:
                table.add_row(metric, str(value))

        console.print(table)

    def _display_comparison(self, comparison: dict):
        """Display comparison of evaluation runs."""
        table = Table(
            title="Evaluation Comparison", show_header=True, header_style="bold magenta"
        )
        table.add_column("Metric", style="cyan")
        table.add_column("Mean", justify="right", style="green")
        table.add_column("Std Dev", justify="right", style="yellow")
        table.add_column("Min", justify="right", style="red")
        table.add_column("Max", justify="right", style="blue")

        for metric, stats in comparison.get("metric_trends", {}).items():
            if isinstance(stats, dict):
                table.add_row(
                    metric,
                    f"{stats.get('mean', 0):.4f}",
                    f"{stats.get('std', 0):.4f}",
                    f"{stats.get('min', 0):.4f}",
                    f"{stats.get('max', 0):.4f}",
                )

        console.print(table)

    def _display_mlflow_info(self, runs):
        """Display MLFlow tracking information."""
        table = Table(
            title="MLFlow Tracking", show_header=True, header_style="bold magenta"
        )
        table.add_column("Evaluation Type", style="cyan")
        table.add_column("Run ID", style="yellow")
        table.add_column("Experiment ID", style="green")
        table.add_column("Status", style="blue")

        for run in runs:
            table.add_row(
                run.evaluation_type.value,
                run.mlflow_run_id or "N/A",
                run.mlflow_experiment_id or "N/A",
                run.status,
            )

        console.print(table)

    def _display_history(self, history):
        """Display evaluation history."""
        table = Table(
            title="Recent Evaluation History",
            show_header=True,
            header_style="bold magenta",
        )
        table.add_column("Type", style="cyan")
        table.add_column("Status", style="green")
        table.add_column("Started At", style="yellow")
        table.add_column("Key Metric", style="blue")

        for run in history:
            # Pick a key metric to display
            key_metric = "N/A"
            if run.metrics:
                if "retrieval_f1" in run.metrics:
                    key_metric = f"F1: {run.metrics['retrieval_f1']:.3f}"
                elif "generation_rougeL" in run.metrics:
                    key_metric = f"ROUGE-L: {run.metrics['generation_rougeL']:.3f}"
                elif "answer_relevancy" in run.metrics:
                    key_metric = f"Relevancy: {run.metrics['answer_relevancy']:.3f}"

            table.add_row(
                run.evaluation_type.value,
                run.status,
                run.started_at.strftime("%Y-%m-%d %H:%M"),
                key_metric,
            )

        console.print(table)


async def main():
    """Main function."""
    try:
        demo = EvaluationDemo()
        await demo.run_demo()
    except Exception as e:
        console.print(f"[red]Error running demo: {e}[/red]")
        logger.error(f"Demo failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    # Check if rich is installed
    try:
        from rich.console import Console
    except ImportError:
        print("Please install 'rich' package: pip install rich")
        sys.exit(1)

    asyncio.run(main())
