"""Proveedores de LLM intercambiables.

`MockProvider` permite probar todo sin red ni claves. `AnthropicProvider` usa el SDK
oficial si está instalado y las variables de entorno están definidas. Los precios NO
están embebidos: si querés costo estimado, definí NORMRAG_PRICE_IN_PER_MTOK y
NORMRAG_PRICE_OUT_PER_MTOK con los valores vigentes de tu proveedor.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Callable, Optional, Protocol


@dataclass
class Completion:
    text: str
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    model: str = ""


class Provider(Protocol):
    name: str

    def complete(self, system: str, user: str, max_tokens: int = 600) -> Completion: ...


class MockProvider:
    """Proveedor determinista para pruebas: `fn(system, user) -> str`."""

    name = "mock"

    def __init__(self, fn: Callable[[str, str], str]):
        self.fn = fn
        self.calls = 0

    def complete(self, system: str, user: str, max_tokens: int = 600) -> Completion:
        self.calls += 1
        text = self.fn(system, user)
        return Completion(text=text, input_tokens=len((system + user).split()), output_tokens=len(text.split()), model="mock")


class AnthropicProvider:
    name = "anthropic"

    def __init__(self, model: Optional[str] = None):
        try:
            import anthropic  # noqa: WPS433
        except ImportError as e:  # pragma: no cover
            raise RuntimeError("Instalá el extra: pip install -e '.[llm]'") from e
        self.model = model or os.environ.get("NORMRAG_MODEL")
        if not self.model:
            raise RuntimeError("Definí NORMRAG_MODEL con el identificador del modelo a usar.")
        if not os.environ.get("ANTHROPIC_API_KEY"):
            raise RuntimeError("Falta ANTHROPIC_API_KEY en el entorno.")
        self._client = anthropic.Anthropic()

    def complete(self, system: str, user: str, max_tokens: int = 600) -> Completion:  # pragma: no cover
        r = self._client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            temperature=0,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        text = "".join(b.text for b in r.content if getattr(b, "type", "") == "text")
        return Completion(text, r.usage.input_tokens, r.usage.output_tokens, self.model)


def estimate_cost_usd(input_tokens: Optional[int], output_tokens: Optional[int]) -> Optional[float]:
    """Costo estimado solo si el usuario declaró precios por millón de tokens."""
    pin = os.environ.get("NORMRAG_PRICE_IN_PER_MTOK")
    pout = os.environ.get("NORMRAG_PRICE_OUT_PER_MTOK")
    if pin is None or pout is None or input_tokens is None or output_tokens is None:
        return None
    return (input_tokens * float(pin) + output_tokens * float(pout)) / 1_000_000
