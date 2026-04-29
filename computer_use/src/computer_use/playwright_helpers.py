"""Playwright helper utilities."""

from playwright.sync_api import Page


def get_element_at_point(page: Page, x: int, y: int) -> str | None:
    """Get a concise representation of the element at the specified coordinates.

    Args:
        page: Playwright page object.
        x: X coordinate.
        y: Y coordinate.

    Returns:
        String representation of the element, or None if not found.
    """
    try:
        result = page.evaluate(
            """([x, y]) => {
                const el = document.elementFromPoint(x, y);
                if (!el) return null;

                const tag = el.tagName.toLowerCase();
                const id = el.id ? `#${el.id}` : '';
                const classes = el.className && typeof el.className === 'string'
                    ? '.' + el.className.trim().split(/\\s+/).join('.')
                    : '';
                const text = el.textContent?.trim().slice(0, 50) || '';
                const attrs = [];

                ['href', 'type', 'name', 'placeholder', 'aria-label', 'role'].forEach(attr => {
                    const val = el.getAttribute(attr);
                    if (val) attrs.push(`${attr}="${val.slice(0, 30)}"`);
                });

                const attrStr = attrs.length ? ' ' + attrs.join(' ') : '';
                const selector = `<${tag}${id}${classes}${attrStr}>`;
                const content = text ? ` "${text.slice(0, 40)}${text.length > 40 ? '...' : ''}"` : '';

                return selector + content;
            }""",
            [x, y],
        )
        return result
    except Exception:
        return None
