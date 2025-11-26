import sys
from threading import Lock

from class_utils import ConfigBaseline, VisitFileProcessor, Statistics

def main(context, anchor_strategy):

    obj = ConfigBaseline(context = context, anchor_strategy=anchor_strategy)
    logger = obj.setup_logging()

    stats = Statistics()
    write_lock = Lock() 

    max_users = None # type : int | None
    force = False   # type : bool
    append = False  # type : bool
    single_file = None # type : str | None


    try:
        # Log della strategia scelta
        logger.info(f"Starting baseline test with strategy: {context}, anchor: {anchor_strategy}")
        
        # Crea e avvia il processore (ora molto più semplice)
        processor = VisitFileProcessor()
        
        processor.process_all_files(
            strategy=context,
            max_users=max_users,
            force=force,
            append=append,
            single_file=single_file
        )
        
    except KeyboardInterrupt:
        logger.info("\nInterrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        # Aggiungi traceback per un debug migliore
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)

if __name__ == "__main__":
    for context in ['random', 'nearest', 'popular', 'absolute']:
        for anchor_strategy in ['middle', 'penultimate']:
            main(context=context, anchor_strategy=anchor_strategy)