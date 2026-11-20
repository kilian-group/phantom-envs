"""Snippets: a retrieved document cut to its first tokens, as BrowseComp-Plus's protocol serves them.

Cutting happens in the server, after retrieval, so it is the same for every retriever.

Usage:

    cutter = SnippetCutter("Qwen/Qwen3-Embedding-4B", max_tokens=512)
    snippet = cutter.cut(document.contents)
"""

from tokenizers import Tokenizer

# A token is rarely longer than this many characters, so this much text holds max_tokens tokens
CHARS_PER_TOKEN: int = 16


class SnippetCutter:
    """Cuts texts to their first max_tokens tokens. Safe to call from many threads at once."""

    def __init__(self, tokenizer: str, max_tokens: int) -> None:
        """Load the tokenizer.

        Args:
            tokenizer: HuggingFace id of the model whose tokenizer counts the tokens.
            max_tokens: The most tokens of a text to keep.
        """
        # The raw Rust tokenizer: its encode() never mutates truncation or padding state, so it is
        # safe under concurrent requests (sharing an encoder's tokenizer raises "Already borrowed")
        self.tokenizer = Tokenizer.from_pretrained(tokenizer)
        self.max_tokens = max_tokens

    def cut(self, text: str) -> str:
        """Return the first max_tokens tokens of text.

        Args:
            text: The text.

        Returns:
            The start of the text, as the tokenizer decodes it.
        """
        # Cut characters first so multi-MB pages don't get fully tokenized on every hit
        text = text[: CHARS_PER_TOKEN * self.max_tokens]
        token_ids = self.tokenizer.encode(text, add_special_tokens=False).ids
        if len(token_ids) <= self.max_tokens:
            return text
        return self.tokenizer.decode(token_ids[: self.max_tokens], skip_special_tokens=True)
