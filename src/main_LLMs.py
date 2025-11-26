import argparse
import os
import queue
import sys
from threading import Lock

from class_utils import OllamaConnectionManager, VisitFileProcessor, Config
from class_utils import Statistics

def main(model, anchor_strategy, context):
    conf_obj = Config(model=model, anchor_strategy=anchor_strategy, prompt_context=context)
    logger = conf_obj.setup_logging()
    stats = Statistics()
    write_lock = Lock()
    
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="VeronaCard Tourist Behavior Prediction System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Process all files:
    python %(prog)s
    
  Process with max 100 users per file:
    python %(prog)s --max-users 100
    
  Process single file:
    python %(prog)s --file data/verona/visits_2014.csv
    
  Resume from previous run:
    python %(prog)s --append
    
  Force reprocessing:
    python %(prog)s --force
        """
    )
    
    parser.add_argument(
        "--file",
        type=str,
        help="Process only this specific file"
    )
    parser.add_argument(
        "--max-users",
        type=int,
        default=None,
        help="Maximum number of users to process per file"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force reprocessing even if output exists"
    )
    parser.add_argument(
        "--append",
        action="store_true",
        help="Resume from previous run (append mode)"
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.force and args.append:
        parser.error("Cannot use both --force and --append")
    
    # Setup GPU environment if not already set
    if "CUDA_VISIBLE_DEVICES" not in os.environ:
        os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"  # Use all 4 A100 GPUs
    
    try:
        # Inizializzazione passo-passo con controlli
        logger.info("Initializing Ollama connection manager...")
        ollama_manager = OllamaConnectionManager(conf_obj, logger)
        
        # Setup connections DEVE essere chiamato
        logger.info("Setting up connections...")
        hosts = ollama_manager.setup_connections()
        if not hosts:
            raise RuntimeError("No hosts configured")
            
        # Verifica inizializzazione corretta
        if ollama_manager.rate_limiter is None:
            raise RuntimeError("Rate limiter not properly initialized")
        if ollama_manager.health_monitor is None:
            raise RuntimeError("Health monitor not properly initialized")
            
        logger.info(f"Initialized with {len(hosts)} hosts")
        
        # Wait for services to be ready
        if not ollama_manager.wait_for_services():
            raise RuntimeError("Ollama services failed to start")
        
        # Create and run processor
        processor = VisitFileProcessor(ollama_manager)
        
        processor.process_all_files(
            max_users=args.max_users,
            force=args.force,
            append=args.append,
            single_file=args.file
        )
        
    except KeyboardInterrupt:
        logger.info("\nInterrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":

    for anchor_strategy in ['middle','penultimate']:
        for model in ['llama3.1:8b', 'deepseek-coder:33b', 'qwen2.5:7b', 'qwen2.5:14b', 'mixtral:8x7b', 'mistral:7b']:
            for context in ['base',  'spatio','spatio-temporal','spatio-temporal-popularity','spatio-temporal-pref']:
                main(model, anchor_strategy, context)