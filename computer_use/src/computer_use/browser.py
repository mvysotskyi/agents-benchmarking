"""Browser module for Playwright setup and management."""

from dataclasses import dataclass
from pathlib import Path

from playwright.sync_api import Browser, BrowserContext, Page, Playwright, sync_playwright

from computer_use.config import DisplayConfig


@dataclass
class BrowserSession:
    """Container for browser session objects."""

    playwright: Playwright
    browser: Browser
    context: BrowserContext
    page: Page

    def close(self) -> None:
        """Close all browser resources."""
        try:
            self.playwright.stop()
        except Exception:
            pass


def get_screen_position(screen_number: int) -> tuple[int, int]:
    """Get the x,y position of a specific screen.

    Args:
        screen_number: Screen number (1 = primary, 2 = secondary, etc.).

    Returns:
        Tuple of (x, y) position for the screen.

    Raises:
        ValueError: If screen number is invalid or not found.
    """
    try:
        from screeninfo import get_monitors
    except ImportError as e:
        raise ValueError("screeninfo library required for multi-monitor support") from e

    monitors = list(get_monitors())

    if not monitors:
        raise ValueError("No monitors detected")

    if screen_number < 1 or screen_number > len(monitors):
        available = ", ".join(str(i + 1) for i in range(len(monitors)))
        raise ValueError(
            f"Screen {screen_number} not found. Available screens: {available}"
        )

    monitor = monitors[screen_number - 1]
    return monitor.x, monitor.y


def create_browser(
    display_config: DisplayConfig | None = None,
    start_url: str | None = None,
    screen: int | None = None,
    headless: bool = False,
) -> BrowserSession:
    """Create a browser session with Playwright.

    Args:
        display_config: Display configuration for viewport size.
        start_url: Initial URL to navigate to (optional).
        screen: Screen number to open browser on (1 = primary, 2 = secondary, etc.).
        headless: Whether to launch Chromium in headless mode.

    Returns:
        BrowserSession containing all browser objects.
    """
    config = display_config or DisplayConfig()

    browser_args = [
        f"--window-size={config.width},{config.height}",
        "--disable-blink-features=AutomationControlled",
    ]

    if screen is not None:
        x, y = get_screen_position(screen)
        browser_args.append(f"--window-position={x},{y}")

    playwright = sync_playwright().start()

    browser = playwright.chromium.launch(
        headless=headless,
        args=browser_args,
    )

    context = browser.new_context(
        viewport={"width": config.width, "height": config.height},
        user_agent=(
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        ),
    )

    page = context.new_page()
    if start_url:
        page.goto(start_url)

    return BrowserSession(
        playwright=playwright,
        browser=browser,
        context=context,
        page=page,
    )


def take_screenshot(page: Page, save_path: Path | None = None) -> bytes:
    """Capture a screenshot of the current page.

    Args:
        page: Playwright page object.
        save_path: Optional path to save the screenshot file.

    Returns:
        Screenshot as bytes (PNG format).
    """
    screenshot_bytes = page.screenshot(type="png", full_page=False)

    if save_path:
        save_path.write_bytes(screenshot_bytes)

    return screenshot_bytes
