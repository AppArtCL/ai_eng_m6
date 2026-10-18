import logging
import re

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from pydantic import BaseModel, Field

import config
from state import AgentState, Decision

logger = logging.getLogger(__name__)

PROMPT_SUPERVISOR = """Eres el supervisor de un equipo que responde preguntas sobre el Reglamento de Copropiedad del Edificio Riesco 6540.
Tu trabajo es decidir quién interviene ahora o si la tarea está completa. No respondes la pregunta tú mismo.

Especialistas disponibles:
- investigador: busca en el reglamento normas, horarios, montos, plazos y prohibiciones, y cita la fuente (archivo y sección). No calcula.
- analista: hace todos los cálculos con los datos que ya entregó el investigador. No busca información.

Orden de trabajo:
1. Si falta un dato del reglamento, envía al investigador.
2. Si la respuesta es un número que NO aparece literalmente en los datos del investigador (un total, una suma, una diferencia, un promedio, un porcentaje, una duración calculada a partir de horarios, un monto multiplicado), envía al analista, aunque el cálculo parezca sencillo. Nunca hagas tú el cálculo.
3. Solo si la respuesta está textualmente en los datos (por ejemplo, "¿está permitido...?", "¿qué horario...?"), no envíes al analista.

Rúbrica de validación (revísala antes de responder FINISH):
1. Datos con fuente: los datos que pide la pregunta fueron entregados por el investigador, con su fuente.
2. Cálculo verificable: si la respuesta es un número que no aparece literalmente en los datos, el analista lo calculó, mostró la operación y usó los datos del investigador.
3. Coherencia: las cifras del analista coinciden con los datos del investigador.
4. Completitud: se responden todas las partes de la pregunta original.
Si un aporte no cumple algún criterio, envía de nuevo al mismo especialista con una instrucción que diga exactamente qué corregir.

Datos no encontrados: si el investigador indica que el reglamento no contiene un dato, envíalo una sola vez más con otros términos de búsqueda. Si tampoco lo encuentra, responde FINISH para que la respuesta final lo informe con honestidad.

Escribe instrucciones concretas y autocontenidas: el especialista solo verá tu instrucción, no la conversación. En tu razón, indica el número del paso del orden de trabajo o del criterio de la rúbrica que aplicaste.
"""


class DecisionSupervisor(BaseModel):
    """Decisión de ruteo del supervisor: quién interviene ahora y con qué tarea."""
    next: Decision = Field(description="Próximo especialista a invocar, o FINISH si la rúbrica se cumple")
    instruccion: str = Field(description="Tarea concreta para el especialista elegido; vacía si next es FINISH")
    razon: str = Field(description="Justificación breve de la decisión, citando el criterio de la rúbrica aplicado")


llm_supervisor = config.get_llm().with_structured_output(DecisionSupervisor)


def _formatear_aportes(state: AgentState) -> str:
    """Convierte las contribuciones en texto para el contexto del supervisor (completas, sin truncar)."""
    contribuciones = state.get("contribuciones", [])
    if not contribuciones:
        return "(ninguno todavía)"
    return "\n".join(
        f"- [{c['agente']}] Instrucción: {c['instruccion']} | Aporte: {c['aporte']}"
        for c in contribuciones
    )


def nodo_supervisor(state: AgentState) -> dict:
    """Decide qué especialista interviene ahora o si la tarea está completa."""
    pasos = state.get("pasos", 0)

    # Guardia determinista: corta aunque el LLM quiera seguir delegando.
    if pasos >= config.MAX_PASOS:
        return {
            "next_agent": "FINISH",
            "instruccion": "",
            "razon": f"Se alcanzó el límite de {config.MAX_PASOS} pasos.",
            "task_completed": True,
        }

    contexto = (
        f"Pregunta original: {state['messages'][0].content}\n\n"
        f"Aportes hasta ahora (paso {pasos} de {config.MAX_PASOS}):\n"
        f"{_formatear_aportes(state)}"
    )
    respuesta = llm_supervisor.invoke([
        SystemMessage(content=PROMPT_SUPERVISOR),
        HumanMessage(content=contexto),
    ])
    decision = DecisionSupervisor.model_validate(respuesta)

    return {
        "next_agent": decision.next,
        "instruccion": decision.instruccion,
        "razon": decision.razon,
        "task_completed": decision.next == "FINISH",
    }


PROMPT_SINTESIS = """Eres el redactor final de un equipo que responde preguntas sobre el Reglamento de Copropiedad del Edificio Riesco 6540.
Redacta una respuesta clara y breve para el usuario usando SOLO los aportes de los especialistas.
- Cada número de tu respuesta debe aparecer literalmente en algún aporte. Si la pregunta pide un número (total, suma, diferencia, porcentaje) que no está en los aportes, no lo calcules: indica que no se alcanzó a calcular.
- Menciona las fuentes (archivo y sección) que citó el investigador.
- Si algún dato no se encontró en el reglamento, dilo con honestidad en vez de suponerlo. No busques información externa.
- Si se te indica que el proceso se cortó por límite de pasos, entrega solo lo que los especialistas alcanzaron a aportar e indica explícitamente qué parte de la pregunta quedó sin resolver.
"""

llm_sintesis = config.get_llm()

_NUMERO = re.compile(r"\d+(?:[.,]\d+)?")


def _numeros(texto: str) -> set[str]:
    """Extrae los números de un texto en forma normalizada: '07' → '7', '57,5' → '57.5'."""
    return {f"{float(n.replace(',', '.')):g}" for n in _NUMERO.findall(texto)}


def _numeros_inventados(respuesta: str, contexto: str) -> set[str]:
    """Números de la respuesta que no aparecen en la pregunta ni en los aportes."""
    return _numeros(respuesta) - _numeros(contexto)


def _respuesta_de_respaldo(state: AgentState) -> str:
    """Respuesta armada en código cuando el LLM no logra redactar sin inventar cifras."""
    return (
        "No fue posible redactar una respuesta verificada. "
        "Estos son los aportes del equipo, tal como los entregaron los especialistas:\n"
        f"{_formatear_aportes(state)}"
    )


def nodo_sintesis(state: AgentState) -> dict:
    """Redacta la respuesta final y valida en código que no introduzca cifras ajenas a los aportes."""
    contexto = (
        f"Pregunta original: {state['messages'][0].content}\n\n"
        f"Aportes del equipo:\n{_formatear_aportes(state)}"
    )
    if state.get("pasos", 0) >= config.MAX_PASOS:
        contexto += f"\n\nAtención: el proceso se cortó al alcanzar el límite de {config.MAX_PASOS} pasos."

    mensajes = [SystemMessage(content=PROMPT_SINTESIS), HumanMessage(content=contexto)]
    respuesta = str(llm_sintesis.invoke(mensajes).content)

    # Validación determinista: cada número de la respuesta debe venir de la pregunta o de un aporte.
    inventados = _numeros_inventados(respuesta, contexto)
    if inventados:
        logger.warning("Síntesis con números ajenos a los aportes %s; se pide reescribir.", sorted(inventados))
        mensajes += [
            AIMessage(content=respuesta),
            HumanMessage(content=(
                f"Tu respuesta incluye números que no aparecen en los aportes: {', '.join(sorted(inventados))}. "
                "Reescríbela sin ellos. Si la pregunta pedía un resultado que no está en los aportes, "
                "indica que no se alcanzó a calcular."
            )),
        ]
        respuesta = str(llm_sintesis.invoke(mensajes).content)
        inventados = _numeros_inventados(respuesta, contexto)
        if inventados:
            logger.warning("El reintento sigue con números ajenos %s; se usa la respuesta de respaldo.", sorted(inventados))
            respuesta = _respuesta_de_respaldo(state)

    return {"messages": [AIMessage(content=respuesta, name="supervisor")]}


if __name__ == "__main__":
    def estado_de_prueba(contribuciones: list, pasos: int) -> AgentState:
        return AgentState(
            messages=[HumanMessage(content="¿Cuántas horas a la semana se permiten mudanzas?")],
            next_agent="FINISH",
            instruccion="",
            razon="",
            contribuciones=contribuciones,
            pasos=pasos,
            task_completed=True,
        )

    aportes = [
        {"agente": "investigador", "instruccion": "Busca los horarios permitidos para mudanzas",
         "aporte": "Lunes a viernes de 07:00 a 17:30; sábados de 09:00 a 14:00; domingos y festivos prohibido. "
                   "Fuente: obras_mudanzas.txt, Protocolo de Mudanzas y Carga en Ascensores."},
        {"agente": "analista", "instruccion": "Calcula las horas semanales permitidas para mudanzas",
         "aporte": "Lunes a viernes: 17.5 - 7 = 10.5 h/día; sábado: 14 - 9 = 5 h. Total: 10.5 * 5 + 5 = 57.5 horas."},
    ]

    print("=== Caso normal (esperado: 57.5 horas, con fuente) ===")
    print(nodo_sintesis(estado_de_prueba(aportes, pasos=2))["messages"][0].content)

    print("\n=== Corte por MAX_PASOS (esperado: advierte respuesta incompleta) ===")
    print(nodo_sintesis(estado_de_prueba(aportes[:1], pasos=config.MAX_PASOS))["messages"][0].content)
