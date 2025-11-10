"""
GPIO Handler - PC Development Version (No Hardware)
"""
import logging

logger = logging.getLogger(__name__)

class GPIOHandler:
    """Dummy GPIO handler for PC development"""
    
    def __init__(self):
        self.is_initialized = False
        logger.info("GPIO Handler: PC mode (no hardware)")
    
    def add_button_callback(self, callback):
        """Dummy method - no GPIO on PC"""
        logger.debug("GPIO button callback skipped (PC mode)")
    
    def set_led(self, state: bool):
        """Dummy method - no LED on PC"""
        logger.debug(f"GPIO LED state: {state} (PC mode)")
    
    def cleanup(self):
        """Dummy cleanup"""
        logger.debug("GPIO cleanup (PC mode)")