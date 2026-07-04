"""Chip: the smallest hover-UI primitive - one piece of text styled as a
pill, painted over the conversation."""

from .ansi import SGR_RESET


class Chip:
    """a single chip. text is the one-line body; sgr, when given, is the
    SGR prefix styling it. position, when given, is a (row, col) viewport
    anchor for the chip's top-left corner - without it the screen stacks
    the chip with the other hover widgets (top-right). timeout, when given,
    removes the chip that many seconds after it is added."""

    def __init__(self, text, sgr='', position=None, timeout=None):
        self.text = text
        self.sgr = sgr
        self.position = position
        self.timeout = timeout

    def lines(self):
        """the chip body as pre-styled lines - a single padded pill row.
        lines() stays a list so placement code needn't care how tall a
        chip is."""
        return [f'{self.sgr} {self.text} {SGR_RESET}']
