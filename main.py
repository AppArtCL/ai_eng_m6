"""Punto de entrada: ejecuta el orquestador multi-agente con una pregunta.

Uso:
    uv run python main.py
    uv run python main.py "¿Puedo hacer asados en mi terraza?"
"""
import logging
import sys
import textwrap

from langchain_core.messages import HumanMessage

import config
from state import AgentState

logger = logging.getLogger(__name__)

PREGUNTA_POR_DEFECTO = "¿Cuántas horas a la semana se permiten mudanzas en el edificio?"


def configurar_logging() -> None:
    """Configura el logging de toda la aplicación (solo aquí, en el punto de entrada)."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    # openai 3.x usa httpx2; versiones anteriores usan httpx. Se silencian ambos.
    for nombre in ("httpx", "httpx2"):
        logging.getLogger(nombre).setLevel(logging.WARNING)


def estado_inicial(pregunta: str) -> AgentState:
    """Construye el estado inicial con todos los campos de AgentState."""
    return AgentState(
        messages=[HumanMessage(content=pregunta)],
        next_agent=None,
        instruccion="",
        razon="",
        contribuciones=[],
        pasos=0,
        task_completed=False,
    )


def _sangria(texto: str) -> str:
    return textwrap.indent(str(texto), "   ")


def ejecutar(pregunta: str) -> str:
    """Recorre el grafo mostrando cada delegación y devuelve la respuesta final."""
    # Import tardío: construir el grafo crea los agentes, y eso exige credenciales válidas.
    from graph import app

    respuesta_final = ""
    for evento in app.stream(estado_inicial(pregunta), stream_mode="updates"):
        for nodo, cambios in evento.items():
            if nodo == "supervisor":
                print(f"🧭 supervisor → {cambios['next_agent']}")
                print(f"   razón: {cambios['razon']}")
                if cambios.get("instruccion"):
                    print(f"   instrucción: {cambios['instruccion']}")
            elif nodo in ("investigador", "analista"):
                print(f"🔹 {nodo}:")
                print(_sangria(cambios["contribuciones"][0]["aporte"]))
            elif nodo == "sintesis":
                respuesta_final = str(cambios["messages"][-1].content)
                print("📝 sintesis: respuesta final redactada")
            print()
    return respuesta_final


def main() -> None:
    configurar_logging()
    try:
        config.validar_credenciales()
    except ValueError as e:
        logger.debug("Validación de credenciales fallida", exc_info=True)
        print(f"❌ Configuración incompleta: {e}\n   Revisa tu archivo .env (usa .env.example como guía).")
        sys.exit(1)

    pregunta = " ".join(sys.argv[1:]) or PREGUNTA_POR_DEFECTO
    print(f"❓ Pregunta: {pregunta}\n")

    respuesta = ejecutar(pregunta)

    print("=" * 70)
    print("✅ RESPUESTA FINAL\n")
    print(respuesta)


if __name__ == "__main__":
    main()
