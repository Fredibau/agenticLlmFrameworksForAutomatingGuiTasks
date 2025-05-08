import logging

def setup_logger(name="ui-tars", level=logging.INFO):
    """
    Configure and return a logger with the specified name and level.
    
    Args:
        name: Logger name
        level: Logging level
        
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    
    # Remove all handlers to prevent duplicate logs
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
        
    # Add a single handler
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(level)
    
    return logger