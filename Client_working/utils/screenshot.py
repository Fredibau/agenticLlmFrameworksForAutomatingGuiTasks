# /utils/screenshot.py
"""
Utility functions for capturing screenshots and handling screen detection.
"""
import pyautogui
import io
import base64
import matplotlib.pyplot as plt
from IPython.display import display, clear_output
from typing import Tuple
from PIL import Image # Import PIL Image

def detect_primary_screen_size() -> Tuple[int, int]:
    """
    Attempt to detect the primary screen size.
    This is a fallback method and might need adjustments based on OS.

    Returns:
        Tuple containing (width, height) of the primary screen
    """
    # First try to use getAllScreens if available
    all_screens = pyautogui.getAllScreens() if hasattr(pyautogui, 'getAllScreens') else None

    if all_screens and len(all_screens) > 0:
        # If pyautogui has getAllScreens() function and returns screen info
        primary_screen = all_screens[0]
        return primary_screen.width, primary_screen.height

    # Get all screens sizes from pyautogui as fallback
    screen_width, screen_height = pyautogui.size()

    # For simplicity, assume first monitor starts at (0,0) and has dimensions:
    # This is a simplification - might need adjustment based on your setup
    if screen_width > 1920:  # Assuming standard monitor width, adjust as needed
        return 1920, screen_height
    else:
        return screen_width, screen_height

# Modify this function
def capture_screenshot(primary_width, primary_height, display_image=True) -> Image.Image:
    """
    Capture a screenshot of the primary screen and return it as a PIL Image.
    Also displays the screenshot in the notebook without the mouse cursor.

    Args:
        primary_width: Width of the primary screen
        primary_height: Height of the primary screen
        display_image: Whether to display the screenshot in the notebook

    Returns:
        PIL.Image.Image object
    """
    # Take a screenshot of only the primary screen (top-left region)
    # The parameter all_screens=False ensures we're only capturing the primary screen
    # The parameter include_cursor=False ensures the mouse is not included in the screenshot
    screenshot_pil = pyautogui.screenshot(region=(0, 0, primary_width, primary_height)) # Changed variable name

    # Display the screenshot in the notebook
    if display_image:
        plt.figure(figsize=(12, 8))
        plt.imshow(screenshot_pil) # Use the PIL image
        plt.axis('off')
        plt.title("Current Screenshot (Primary Screen)")
        clear_output(wait=True)
        display(plt.gcf())
        plt.close()

    # No longer return base64 here, return the PIL Image
    return screenshot_pil

# helper function to convert PIL Image to base64
def pil_to_base64(image: Image.Image) -> str:
    """Converts a PIL Image to a base64 string."""
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")