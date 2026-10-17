from langchain.agents import create_agent
from langchain_core.messages import AIMessage, HumanMessage

import config
from state import AgentState
from agents.tools import calculadora

PROMPT_ANALISTA = """Eres el agente analista del Reglamento de Copropiedad del Edificio Riesco 6540, encargado de hacer todos los cálculos que se requieran.
Debes trabajar con los datos que te entregan, sin buscar o inventar otros datos.
Debes usar la calculadora para realizar cualquier cálculo que se haga, aunque sea sencillo.
Muestra la operación realizada y el resultado, incluyendo las unidades (por ejemplo, m², $, horas, UTM, días).
Si faltan datos para hacer los cálculos, no hagas ningún cálculo e informa explícitamente la falta de datos.
"""

agente_analista = create_agent(
    model=config.get_llm(),
    tools=[calculadora],
    system_prompt=PROMPT_ANALISTA,
)

SIN_DATOS = "(sin datos del investigador)"


def _ultimo_aporte_investigador(state: AgentState) -> str:
    """Devuelve el aporte más reciente del investigador (no el del propio analista)."""
    return next(
        (c["aporte"] for c in reversed(state.get("contribuciones", [])) if c["agente"] == "investigador"),
        SIN_DATOS,
    )


def nodo_analista(state: AgentState) -> dict:
    """Ejecuta al analista con la instrucción del supervisor y solo los datos del investigador."""
    instruccion = state.get("instruccion") or state["messages"][0].content
    datos = _ultimo_aporte_investigador(state)
    mensaje = f"Tarea: {instruccion}\n\nDatos disponibles:\n{datos}"

    resultado = agente_analista.invoke({"messages": [HumanMessage(content=mensaje)]})
    respuesta = resultado["messages"][-1].content
    return {
        "messages": [AIMessage(content=respuesta, name="analista")],
        "contribuciones": [{"agente": "analista", "instruccion": instruccion, "aporte": respuesta}],
        "pasos": 1,
    }


if __name__ == "__main__":
    aporte_investigador = (
        "Los horarios permitidos para mudanzas en el Edificio Riesco 6540 son:\n"
        "- Lunes a viernes: entre las 07:00 y las 17:30 horas.\n"
        "- Sábados: entre las 09:00 y las 14:00 horas.\n"
        "- Domingos y días festivos: está estrictamente prohibida la realización de mudanzas.\n"
        "Fuente: Protocolo de Mudanzas y Carga en Ascensores, obras_mudanzas.txt."
    )
    def estado_de_prueba(contribuciones: list) -> AgentState:
        return AgentState(
            messages=[HumanMessage(content="¿Cuántas horas a la semana se permiten mudanzas?")],
            next_agent="analista",
            instruccion="Calcula cuántas horas a la semana se permiten mudanzas",
            razon="prueba",
            contribuciones=contribuciones,
            pasos=len(contribuciones),
            task_completed=False,
        )

    print("=== Caso 1: con datos del investigador (esperado: 57.5 horas) ===")
    con_datos = estado_de_prueba([{"agente": "investigador", "instruccion": "Busca los horarios de mudanzas",
                                   "aporte": aporte_investigador}])
    print(nodo_analista(con_datos)["contribuciones"][0]["aporte"])

    print("\n=== Caso 2: sin datos (esperado: informa que faltan datos) ===")
    print(nodo_analista(estado_de_prueba([]))["contribuciones"][0]["aporte"])
    
    print("\n=== Verificación: ¿usó la calculadora? ===")
    mensaje = f"Tarea: Calcula cuántas horas a la semana se permiten mudanzas\n\nDatos disponibles:\n{aporte_investigador}"
    resultado = agente_analista.invoke({"messages": [HumanMessage(content=mensaje)]})
    llamadas = [tc["args"] for m in resultado["messages"] for tc in getattr(m, "tool_calls", [])]
    print("Llamadas a la calculadora:", llamadas or "NINGUNA ❌")
