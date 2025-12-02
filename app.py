import os
import streamlit as st
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from loguru import logger
import requests
from dotenv import load_dotenv
import re
import hashlib
import time
from datetime import datetime

load_dotenv(override=True)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL")
OPENAI_MODEL = os.getenv("OPENAI_MODEL")

print(f"Loaded environment variables")
print(f"MODEL: {OPENAI_MODEL}")

if "conversation_history" not in st.session_state:
    st.session_state["conversation_history"] = []
if "response_cache" not in st.session_state:
    st.session_state["response_cache"] = {}
if "metrics_history" not in st.session_state:
    st.session_state["metrics_history"] = []

PLACEHOLDER_OBJECTIVE = "Ej.: Maximizar que todas las viviendas tengan acceso a salud, educación, áreas verdes y supermercado dentro de 15 minutos."
PLACEHOLDER_RULES = "Ej.: No construir en ríos; priorizar compatibilidad entre usos; mantener conectividad con calles existentes; evitar saturar un servicio en una sola zona, etc."
PLACEHOLDER_CALCULATIONS = "Ej.: Contar servicios cercanos por vivienda; medir distancias a pie; aplicar una matriz de compatibilidad sencilla; evitar duplicar el mismo servicio si ya hay cobertura suficiente."
PLACEHOLDER_QUESTION = "Ej.: ¿Por qué construiste un hospital aquí?"

PLACEHOLDER_TECH_OBJECTIVE = r"Maximizar el retorno acumulado \sum_t \gamma^t r_t bajo la política óptima \pi^\*, reduciendo la distancia media a servicios esenciales con umbral N=15 (Manhattan)."
PLACEHOLDER_TECH_RULES = (
    "• Política ε-greedy con ε0=1.0 y decaimiento Gompertz: ε(x)=exp(exp(-c·x + b)), b=-e, c=-0.03883259.\n"
    "• Tasa de aprendizaje α=0.5, descuento γ=0.95.\n"
    "• Compatibilidad espacial según matriz C∈[1,5]; evaluar vecinos a distancia Manhattan 2.\n"
    "• No construir en obstáculos (ríos/zonas no edificables); respetar conectividad vial."
)
PLACEHOLDER_TECH_CALCULATIONS = (
    "• Actualización Q (Bellman): Q(s_t,a_t) ← Q(s_t,a_t) + α [ r_{t+1} + γ max_a Q(s_{t+1},a) − Q(s_t,a_t) ].\n"
    "• Recompensa residencial: sumar compatibilidades de servicios cercanos ponderadas por maxAmount=2 por tipo; decrementar al exceder; total por ciudad R=∑_i R_i.\n"
    "• Cobertura: contar servicios distintos por residencia dentro de N=15.\n"
    "• (Alternativa DQN) MLP [128,64,128], dropout 0.22; exploración ε-greedy idéntica."
)
PLACEHOLDER_TECH_QUESTION = r"¿Por qué la política \pi eligió colocar hospital en la celda (i,j) dadas las Q(s,a) actuales y el maxAmount por servicio?"

PROMPT_FORMATTING_CONFIG = {
    "rules_prefix": "- ",
    "rules_no_data_text": "No se proporcionaron reglas específicas del agente.",
    "calculations_prefix": "- ",
    "calculations_no_data_text": "No se proporcionaron cálculos específicos realizados por el agente.",
    "objective_no_data_text": "No se especificó un objetivo concreto",
    "decision_default_text": "Tomar esta acción en este sitio",
    "calculations_phrase": "Se realizaron los cálculos",
    "decision_phrase": "Es por eso que se decidió:",
}

PRESET_SIMPLE = {
    "objective": PLACEHOLDER_OBJECTIVE,
    "rules": PLACEHOLDER_RULES,
    "calculations": PLACEHOLDER_CALCULATIONS,
    "question": PLACEHOLDER_QUESTION,
}
PRESET_TECHNICAL = {
    "objective": PLACEHOLDER_TECH_OBJECTIVE,
    "rules": PLACEHOLDER_TECH_RULES,
    "calculations": PLACEHOLDER_TECH_CALCULATIONS,
    "question": PLACEHOLDER_TECH_QUESTION,
}

BASE_CRITICAL_RULES = [
    "- NUNCA inventes información que no esté explícitamente en el contexto proporcionado.",
    "- Solo responde 'no sé' si los campos están literalmente vacíos o contienen únicamente texto placeholder.",
    "- Los EJEMPLOS son solo para mostrar el formato, NO uses sus datos. Usa ÚNICAMENTE los datos del contexto actual.",
    "- No repitas ni cites literalmente el mensaje de la persona. No incluyas su texto en la respuesta.",
    "- No inventes datos, números, métricas, cálculos, o decisiones que no estén en el contexto.",
    '- No uses metatexto como "Entendido", "A continuación" o similares.',
    "- Mantén la salida EXACTAMENTE en el formato indicado abajo.",
]

SYSTEM_PROMPT_LEVEL_CONFIG = {
    1: {
        "rol": "un EXPLICADOR URBANO para público general no técnico",
        "tarea": "Tu tarea: explicar en lenguaje sencillo y cotidiano por qué el agente tomó una decisión urbana.",
        "rules_extra": [
            "- Prohibido usar jerga técnica de ningún tipo (ni urbanismo especializado, ni RL).",
            '- Usa palabras cotidianas: "barrio" en vez de "zona", "caminar" en vez de "movilidad peatonal".',
            "- Máximo 200 palabras. Tono cercano, amigable y conversacional.",
        ],
        "format_section": (
            "FORMATO DE SALIDA (EXACTO):\n\n"
            "Dado el objetivo del agente urbano, que es {objective},\n"
            "y las reglas establecidas:\n"
            "{rules_in_simple}\n\n"
            "Se realizaron los cálculos:\n"
            "{calculations_in_simple}\n\n"
            "Es por eso que se decidió: {clear_decision}"
        ),
        "style_guides": [
            '- Explica con palabras muy simples: "vecindarios", "cercanía", "variedad de lugares", "caminos", "no saturar".',
            "- Evita cualquier tecnicismo. Habla como si explicaras a un vecino.",
            "- Estructura mental: objetivo → reglas prácticas → lo que se revisó → decisión final.",
        ],
        "principles_section": (
            "PRINCIPIOS (EXPLICA SIMPLE EN 1–2 FRASES):\n"
            "- Que la gente pueda caminar a los servicios que necesita.\n"
            "- Que haya variedad de servicios sin que se amontonen.\n"
            "- Que los caminos y calles conecten bien todo."
        ),
        "example_section": (
            "EJEMPLO DE FORMATO (NO uses estos datos, son solo para mostrar la estructura):\n"
            "Si tuvieras el objetivo 'acercar servicios a viviendas', reglas sobre 'favorecer cercanía', "
            "y cálculos de 'casas beneficiadas', la respuesta seguiría este patrón:\n\n"
            "Dado el objetivo del agente urbano, que es [objetivo real del contexto],\n"
            "y las reglas establecidas:\n"
            "- [regla 1 del contexto]\n"
            "- [regla 2 del contexto]\n"
            "Se realizaron los cálculos:\n"
            "- [cálculo 1 del contexto]\n"
            "- [cálculo 2 del contexto]\n\n"
            "Es por eso que se decidió: [decisión basada en el contexto real]\n\n"
            "IMPORTANTE: Reemplaza TODO lo que está entre [ ] con información del contexto actual. "
            "Si algo dice 'no sé', responde que falta esa información."
        ),
    },
    2: {
        "rol": "un EXPLICADOR URBANO para profesionales del diseño urbano y arquitectura",
        "tarea": "Tu tarea: explicar desde una perspectiva de planeación urbana por qué el agente tomó una decisión.",
        "rules_extra": [
            "- Usa terminología de urbanismo y diseño urbano profesional.",
            "- Evita jerga específica de RL/ML (no menciones Q-learning, DQN, políticas, Bellman, etc.).",
            "- Términos permitidos: zonificación, morfología urbana, accesibilidad, densidad, uso mixto, red vial, conectividad, equipamiento.",
            "- Máximo 250 palabras. Tono profesional pero accesible.",
            "- IMPORTANTE: Usa EXACTAMENTE la información proporcionada en el contexto. No digas 'no sé' si hay información disponible.",
            "- OBLIGATORIO: Si hay texto en los campos de cálculos y decisión, úsalo directamente sin cuestionar su completitud.",
        ],
        "format_section": (
            "FORMATO DE SALIDA (EXACTO):\n\n"
            "Dado el objetivo del agente urbano, que es {objective},\n"
            "y las reglas establecidas:\n"
            "{rules_in_simple}\n\n"
            "Se realizaron los cálculos:\n"
            "{calculations_in_simple}\n\n"
            "Es por eso que se decidió: {clear_decision}"
        ),
        "style_guides": [
            "- Usa vocabulario de diseño urbano: accesibilidad peatonal, radio de cobertura, compatibilidad de usos, estructura vial, densidad de servicios.",
            "- Conecta con principios de urbanismo sostenible: proximidad, diversidad funcional, permeabilidad.",
            "- Estructura: objetivo de planeación → criterios de diseño → análisis espacial → decisión fundamentada.",
        ],
        "principles_section": (
            "PRINCIPIOS DE DISEÑO URBANO (INCLUIR EN LA CONCLUSIÓN EN 1–2 FRASES):\n"
            "- Proximidad/caminabilidad: optimizar radios de influencia peatonal hacia equipamiento esencial.\n"
            "- Diversidad/compatibilidad: promover uso mixto evitando conflictos funcionales y saturación.\n"
            "- Conectividad: integrar la intervención en la estructura vial y sistema de movilidad."
        ),
        "example_section": "",
    },
    3: {
        "rol": "un EXPLICADOR TÉCNICO de sistemas de Reinforcement Learning aplicados a planeación urbana",
        "tarea": "Tu tarea: explicar desde la perspectiva de RL/DQN por qué el agente tomó una decisión.",
        "rules_extra": [
            "- Usa terminología técnica de RL: Q-learning, DQN, política, función de valor, recompensa, estado, acción, exploración/explotación.",
            "- Términos técnicos permitidos: Q(s,a), policy π, reward function R, state space, action space, Bellman equation, epsilon-greedy, experience replay.",
            "- Si falta información sobre parámetros técnicos, solicítala específicamente.",
            "- Máximo 300 palabras. Tono técnico-académico.",
            "- Puedes referenciar arquitecturas de red, hiperparámetros, funciones de recompensa.",
        ],
        "format_section": (
            "FORMATO DE SALIDA (EXACTO):\n\n"
            "Dado el objetivo del agente de RL, que es {objective},\n"
            "y la política implementada:\n"
            "{rules_in_simple}\n\n"
            "Se evaluaron los estados y acciones:\n"
            "{calculations_in_simple}\n\n"
            "Es por eso que se seleccionó la acción: {clear_decision}"
        ),
        "style_guides": [
            "- Explica en términos de RL: función de valor Q, maximización de recompensa esperada, estado del entorno.",
            "- Usa notación técnica cuando sea apropiado: Q(s,a), R(s,a,s'), γ (discount factor), ε (epsilon).",
            "- Estructura: función objetivo/recompensa → política y reglas de decisión → evaluación de Q-values → selección de acción óptima.",
        ],
        "principles_section": (
            "PRINCIPIOS DEL SISTEMA RL (INCLUIR EN LA CONCLUSIÓN EN 1–2 FRASES):\n"
            "- Optimización: maximizar recompensa acumulada considerando proximidad peatonal, diversidad de servicios y conectividad vial.\n"
            "- Trade-offs: balance entre exploración (nuevas configuraciones) y explotación (estrategias probadas).\n"
            "- Convergencia: cómo esta acción contribuye a la política óptima π* según los Q-values estimados."
        ),
        "example_section": (
            "EJEMPLO DE FORMATO (NO uses estos datos inventados, son solo para mostrar la estructura):\n"
            "Si tuvieras una función de recompensa definida, una política específica y Q-values calculados, "
            "la respuesta seguiría este patrón:\n\n"
            "Dado el objetivo del agente de RL, que es [objetivo real del contexto],\n"
            "y la política implementada:\n"
            "- [política 1 del contexto]\n"
            "- [política 2 del contexto]\n"
            "Se evaluaron los estados y acciones:\n"
            "- [evaluación 1 del contexto]\n"
            "- [evaluación 2 del contexto]\n\n"
            "Es por eso que se seleccionó la acción: [acción basada en el contexto real]\n\n"
            "CRÍTICO: Reemplaza TODO entre [ ] con datos del contexto proporcionado. "
            "NO inventes Q-values, pesos, epsilon, o cualquier parámetro. Si no están en el contexto, di 'no sé'."
        ),
    },
}


def build_system_prompt(level: int) -> str:
    config = SYSTEM_PROMPT_LEVEL_CONFIG.get(level, SYSTEM_PROMPT_LEVEL_CONFIG[1])

    rules = BASE_CRITICAL_RULES + config.get("rules_extra", [])
    rules_block = "REGLAS CRÍTICAS (OBLIGATORIAS):\n" + "\n".join(rules)

    style_guides = config.get("style_guides", [])
    style_block = "GUÍAS DE ESTILO:\n" + "\n".join(style_guides) if style_guides else ""

    sections = [
        f"Eres {config['rol']}",
        config["tarea"],
        rules_block,
        config.get("format_section", "").strip(),
        style_block,
    ]

    principles_section = config.get("principles_section", "").strip()
    if principles_section:
        sections.append(principles_section)

    example_section = config.get("example_section", "").strip()
    if example_section:
        sections.append(example_section)

    return "\n\n".join(section for section in sections if section).strip()


def get_system_prompt_by_level(level: int) -> str:
    return build_system_prompt(level)


st.set_page_config(
    page_title="Explicador del agente urbano", page_icon="🏙️", layout="centered"
)
st.title("Explicador de decisiones del agente urbano")

with st.sidebar:
    st.header("⚙️ Configuración")
    st.subheader("Variables de entorno")

    sidebar_api_key = st.text_input(
        "OPENAI_API_KEY",
        value=OPENAI_API_KEY or "",
        type="password",
        help="Clave API de OpenAI",
    )
    sidebar_base_url = st.text_input(
        "OPENAI_BASE_URL",
        value=OPENAI_BASE_URL or "",
        help="URL base del servicio OpenAI",
    )
    sidebar_model = st.text_input(
        "OPENAI_MODEL", value=OPENAI_MODEL or "", help="Modelo a utilizar"
    )

    if sidebar_api_key:
        OPENAI_API_KEY = sidebar_api_key
    if sidebar_base_url:
        OPENAI_BASE_URL = sidebar_base_url
    if sidebar_model:
        OPENAI_MODEL = sidebar_model

    st.divider()

    st.subheader("Estado de configuración")
    if OPENAI_API_KEY:
        st.success("✓ API Key configurada")
    else:
        st.error("✗ API Key faltante")

    if OPENAI_BASE_URL:
        st.success("✓ Base URL configurada")
    else:
        st.error("✗ Base URL faltante")

    if OPENAI_MODEL:
        st.success("✓ Modelo configurado")
    else:
        st.error("✗ Modelo faltante")

missing_vars = []
if not OPENAI_API_KEY:
    missing_vars.append("OPENAI_API_KEY")
if not OPENAI_BASE_URL:
    missing_vars.append("OPENAI_BASE_URL")
if not OPENAI_MODEL:
    missing_vars.append("OPENAI_MODEL")

if missing_vars:
    st.error(
        f"⚠️ **Variables de entorno faltantes:** {', '.join(missing_vars)}. "
        f"Por favor configúralas en el archivo .env o en la barra lateral."
    )


main_tab, second_tab = st.tabs(["🏙️ Explicador Agente Silogismo Practico", "🤖 Explicacion General RL"])

with main_tab:
    st.subheader("🎚️ Nivel Técnico de Explicación")
    technical_level = st.radio(
        "Selecciona el nivel de tecnicismo en la respuesta:",
        options=[1, 2, 3],
        index=st.session_state.get("technical_level", 1) - 1,
        format_func=lambda x: {
            1: "1️⃣ Lenguaje Común (Público General)",
            2: "2️⃣ Lenguaje Profesional (Arquitecto/Urbanista)",
            3: "3️⃣ Lenguaje Técnico (Deep Q-Learning / RL)",
        }[x],
        horizontal=True,
        help="""💡 Ajusta el vocabulario y complejidad de la explicación:
    
        • Nivel 1: Lenguaje cotidiano sin tecnicismos (ideal para ciudadanos)
        • Nivel 2: Terminología urbanística profesional (para arquitectos/urbanistas)  
        • Nivel 3: Vocabulario técnico de RL/ML (para científicos de datos)
    
        Las respuestas se adaptan completamente al nivel seleccionado.""",
    )
    st.session_state["technical_level"] = technical_level

    level_descriptions = {
        1: "💬 **Lenguaje cotidiano y sencillo** - Perfecto para explicar a vecinos o público general sin conocimientos técnicos.",
        2: "🏗️ **Terminología de urbanismo profesional** - Usa conceptos de diseño urbano, zonificación, y planeación para arquitectos y diseñadores.",
        3: "🤖 **Vocabulario de Reinforcement Learning** - Explicación técnica con Q-learning, políticas, funciones de recompensa y arquitecturas de red.",
    }
    st.info(level_descriptions[technical_level])

    with st.expander("🔧 Personalizar System Prompt (Avanzado)", expanded=False):
        st.caption(
            "Modifica el prompt del sistema para cambiar el comportamiento del agente."
        )
        default_prompt = get_system_prompt_by_level(technical_level)
        prompt_value = st.session_state.get("system_prompt_override", default_prompt)

        if "custom_prompt_level" not in st.session_state:
            st.session_state["custom_prompt_level"] = technical_level
        if "custom_system_prompt" not in st.session_state:
            st.session_state["custom_system_prompt"] = prompt_value

        has_override = "system_prompt_override" in st.session_state
        if (
            has_override
            and st.session_state["custom_system_prompt"]
            != st.session_state["system_prompt_override"]
        ):
            st.session_state["custom_system_prompt"] = st.session_state[
                "system_prompt_override"
            ]
        if not has_override and st.session_state["custom_prompt_level"] != technical_level:
            st.session_state["custom_system_prompt"] = default_prompt

        st.session_state["custom_prompt_level"] = technical_level

        custom_system_prompt = st.text_area(
            "System Prompt",
            height=300,
            help="Este es el prompt que guía el comportamiento del LLM",
            key="custom_system_prompt",
        )
        if st.button("Aplicar prompt personalizado"):
            st.session_state["system_prompt_override"] = custom_system_prompt
            st.success("✓ Prompt personalizado aplicado")
        if st.button("Restaurar prompt por defecto"):
            if "system_prompt_override" in st.session_state:
                del st.session_state["system_prompt_override"]
            if "custom_system_prompt" in st.session_state:
                del st.session_state["custom_system_prompt"]
            st.success("✓ Prompt restaurado al valor por defecto")
            st.rerun()

    col_p1, col_p2 = st.columns([2, 1])
    with col_p1:
        preset_choice = st.selectbox(
            "Preset de ejemplo",
            options=["Sencillo (no técnico)", "Técnico (RL)"],
            index=0,
            help="Elige un ejemplo y presiona 'Cargar ejemplo' para rellenar los campos.",
        )
    with col_p2:
        if st.button("Cargar ejemplo"):
            p = PRESET_SIMPLE if preset_choice.startswith("Sencillo") else PRESET_TECHNICAL
            st.session_state["objective"] = p["objective"]
            st.session_state["rules"] = p["rules"]
            st.session_state["calculations"] = p["calculations"]
            st.session_state["question"] = p["question"]

    if preset_choice.startswith("Sencillo"):
        current_placeholder_objective = PLACEHOLDER_OBJECTIVE
        current_placeholder_rules = PLACEHOLDER_RULES
        current_placeholder_calculations = PLACEHOLDER_CALCULATIONS
        current_placeholder_question = PLACEHOLDER_QUESTION
    else:
        current_placeholder_objective = PLACEHOLDER_TECH_OBJECTIVE
        current_placeholder_rules = PLACEHOLDER_TECH_RULES
        current_placeholder_calculations = PLACEHOLDER_TECH_CALCULATIONS
        current_placeholder_question = PLACEHOLDER_TECH_QUESTION

    st.session_state["placeholder_objective"] = current_placeholder_objective
    st.session_state["placeholder_rules"] = current_placeholder_rules
    st.session_state["placeholder_calculations"] = current_placeholder_calculations
    st.session_state["placeholder_question"] = current_placeholder_question

    objective = st.text_area(
        "1) Objetivo del agente",
        placeholder=current_placeholder_objective,
        height=100,
        key="objective",
        help="🎯 Describe qué busca optimizar el agente. Ejemplo: maximizar accesibilidad a servicios, minimizar distancias caminables.",
    )
    rules = st.text_area(
        "2) Reglas que sigue el agente",
        placeholder=current_placeholder_rules,
        height=140,
        key="rules",
        help="📋 Define las restricciones y políticas del agente. Ejemplo: no construir en zonas protegidas, mantener diversidad de servicios, respetar capacidad máxima.",
    )
    calculations = st.text_area(
        "3) Cálculos realizados",
        placeholder=current_placeholder_calculations,
        height=140,
        key="calculations",
        help="🧮 Especifica las métricas y evaluaciones realizadas. Ejemplo: distancias Manhattan, matriz de compatibilidad, conteo de servicios cercanos.",
    )
    question = st.text_area(
        "4) Pregunta persona",
        placeholder=current_placeholder_question,
        height=80,
        key="question",
        help="❓ Formula la pregunta sobre la decisión del agente. Ejemplo: ¿Por qué colocó el hospital aquí? ¿Por qué no eligió esta otra ubicación?",
    )

    SYSTEM_PROMPT = """
    Eres un EXPLICADOR URBANO para público no técnico.
    Tu tarea: explicar en lenguaje claro por qué el agente tomó una decisión urbana.

    REGLAS CRÍTICAS (OBLIGATORIAS):
    - No repitas ni cites literalmente el mensaje de la persona. No incluyas su texto en la respuesta.
    - Prohibido usar jerga de RL (no digas Q-learning, DQN, política, Bellman, etc.).
    - Si falta información, responde "no sé" y sugiere 1–2 datos concretos que habría que pedir.
    - Máximo 200 palabras. Tono cercano y respetuoso.
    - No inventes datos ni métricas.
    - No uses metatexto como “Entendido”, “A continuación” o similares.
    - Mantén la salida EXACTAMENTE en el formato indicado abajo.

    FORMATO DE SALIDA (EXACTO):

    Dado el objetivo del agente urbano, que es {objetivo},
    y las reglas establecidas:
    {reglas_en_simple}

    Se realizaron los cálculos:
    {calculos_en_simple}

    Es por eso que se decidió: {decision_clara}

    GUÍAS DE ESTILO:
    - Explica reglas y cálculos con palabras sencillas (vecindarios, cercanía, variedad de servicios, conexiones, evitar saturación).
    - Evita tecnicismos, fórmulas o símbolos.
    - Estructura mental tipo silogismo práctico: fin (objetivo) → normas (reglas) → percepción/cálculo (cómputos) → acción (decisión).

    PRINCIPIOS DE PROXIMIDAD (INCLUIR EN LA CONCLUSIÓN EN 1–2 FRASES):
    - Proximidad/caminabilidad: mejorar distancias a pie reales a servicios esenciales.
    - Diversidad/compatibilidad: distribuir distintos servicios sin conflictos de uso.
    - Conectividad: integrar la decisión con calles y transporte para accesos efectivos.
    (Resume explícitamente cómo la decisión favorece proximidad + diversidad/compatibilidad + conectividad.)

    EJEMPLO (MINI few-shot; imita el tono y la estructura, NO COPIES el contenido del usuario):
    Respuesta agente:
    Dado el objetivo del agente de RL, que es acercar educación y áreas verdes a las viviendas,
    y las reglas establecidas:
    - Favorecer que la gente camine poco para llegar a servicios clave.
    - Mantener variedad sin saturar una sola zona.
    - Ubicar usos que se lleven bien entre sí.
    Se realizaron los cálculos:
    - Se contó cuántas casas ganarían acceso a pie.
    - Se verificó que no se sobrecargara la zona y que existieran caminos conectados.
    - Se compararon alternativas cercanas con menos beneficio.

    Es por eso que se decidió: Ubicar una escuela al lado del parque
    """


    def test_llm_connection() -> bool:
        try:
            if not OPENAI_BASE_URL:
                return False
            base = OPENAI_BASE_URL.rstrip("/")
            headers = {}
            if OPENAI_API_KEY:
                headers["Authorization"] = f"Bearer {OPENAI_API_KEY}"
            url = base
            resp = requests.get(url, headers=headers, timeout=8)
            logger.info(f"[LLM] Connection probe {url} → {resp.status_code}")
            return resp.status_code < 400 or resp.status_code in (401, 403, 404)
        except Exception as e:
            logger.warning(f"[LLM] Connection test failed (non-fatal): {e}")
            return False


    def value_or_default(val: str | None, default: str) -> str:
        if val is None:
            return default
        v = val.strip()
        return v if v else default


    def _clean(s: str) -> str:
        if not s:
            return s
        s = s.strip()
        lowers = s.lower()
        if lowers.startswith(("ej.:", "ej:", "ejemplo:", "e.g.")):
            s = s.split(":", 1)[-1].strip()
        return s.strip(' "')


    if not OPENAI_API_KEY or not OPENAI_BASE_URL or not OPENAI_MODEL:
        logger.error("Faltan variables de entorno de OpenAI. El LLM no se inicializará.")
        llm = None
    else:
        try:
            try:
                from langchain.schema import BaseCache

                ChatOpenAI.model_rebuild()
            except (ImportError, AttributeError):
                pass

            base_url = OPENAI_BASE_URL
            llm = ChatOpenAI(
                model=OPENAI_MODEL,
                openai_api_key=OPENAI_API_KEY,
                openai_api_base=base_url,
                timeout=70,
                max_retries=3,
                model_kwargs={},
            )
            logger.info(f"[LLM] Initialized with model: {OPENAI_MODEL}")
            if test_llm_connection():
                logger.info("[LLM] Connection test passed")
            else:
                logger.warning(
                    "[LLM] Connection test failed (puede seguir funcionando si el proveedor no expone /)"
                )
        except Exception as e:
            logger.error(f"[LLM] Failed to initialize: {e}")
            llm = None


    def build_user_prompt(
        objective: str, rules: str, calculations: str, question: str
    ) -> str:
        objective_clean = (objective or "").strip()
        rules_clean = (rules or "").strip()
        calculations_clean = (calculations or "").strip()
        question_clean = (question or "").strip()

        rules_in_simple = (
            PROMPT_FORMATTING_CONFIG["rules_prefix"] + rules_clean
            if rules_clean
            else PROMPT_FORMATTING_CONFIG["rules_prefix"]
            + PROMPT_FORMATTING_CONFIG["rules_no_data_text"]
        )
        calculations_phrase = PROMPT_FORMATTING_CONFIG["calculations_phrase"]
        if calculations_phrase in rules_in_simple and not rules_in_simple.endswith("\n"):
            rules_in_simple = rules_in_simple.replace(
                calculations_phrase, "\n" + calculations_phrase
            )
        if not rules_in_simple.endswith("\n"):
            rules_in_simple += "\n"
        rules_in_simple = re.sub(
            rf"(?<!\n)\s*{re.escape(calculations_phrase)}",
            f"\n\n{calculations_phrase}",
            rules_in_simple,
        )

        calculations_in_simple = (
            PROMPT_FORMATTING_CONFIG["calculations_prefix"] + calculations_clean
            if calculations_clean
            else PROMPT_FORMATTING_CONFIG["calculations_prefix"]
            + PROMPT_FORMATTING_CONFIG["calculations_no_data_text"]
        )
        if (
            calculations_phrase in calculations_in_simple
            and not calculations_in_simple.endswith("\n")
        ):
            calculations_in_simple = calculations_in_simple.replace(
                calculations_phrase, "\n" + calculations_phrase
            )
        if not calculations_in_simple.endswith("\n"):
            calculations_in_simple += "\n"
        calculations_in_simple = re.sub(
            rf"(?<!\n)\s*{re.escape(calculations_phrase)}",
            f"\n\n{calculations_phrase}",
            calculations_in_simple,
        )

        lower_question = question_clean.lower()
        if "hospital" in lower_question:
            clear_decision = "Construir un hospital aquí"
        elif "escuela" in lower_question or "colegio" in lower_question:
            clear_decision = "Ubicar una escuela en este sitio"
        elif "parque" in lower_question or "área verde" in lower_question:
            clear_decision = "Crear un área verde en este punto"
        else:
            clear_decision = PROMPT_FORMATTING_CONFIG["decision_default_text"]

        tech_level = st.session_state.get("technical_level", 1)
        default_prompt = get_system_prompt_by_level(tech_level)
        active_system_prompt = st.session_state.get(
            "system_prompt_override", default_prompt
        )

        format_params = {
            "objective": (
                objective_clean
                if objective_clean
                else PROMPT_FORMATTING_CONFIG["objective_no_data_text"]
            ),
            "rules_in_simple": rules_in_simple,
            "calculations_in_simple": calculations_in_simple,
            "clear_decision": clear_decision,
        }

        try:
            prompt_text = active_system_prompt.format(**format_params)
        except KeyError as e:
            logger.warning(f"Missing format parameter: {e}. Using fallback prompt.")
            prompt_text = active_system_prompt

        prompt_text = re.sub(
            rf"(?<!\n)\s*{re.escape(calculations_phrase)}",
            f"\n\n{calculations_phrase}",
            prompt_text,
        )
        decision_phrase = PROMPT_FORMATTING_CONFIG["decision_phrase"]
        prompt_text = re.sub(
            rf"(?<!\n)\s*{re.escape(decision_phrase)}",
            f"\n\n{decision_phrase}",
            prompt_text,
        )
        return prompt_text


    def generate_response_from_inputs(
        objective_in: str, rules_in: str, calculations_in: str, question_in: str
    ) -> tuple[str | None, dict]:
        metrics = {
            "start_time": time.time(),
            "end_time": None,
            "duration": None,
            "cached": False,
            "technical_level": st.session_state.get("technical_level", 1),
            "timestamp": datetime.now().isoformat(),
        }

        if not llm:
            logger.error("LLM no está inicializado. No se puede generar respuesta.")
            metrics["end_time"] = time.time()
            metrics["duration"] = metrics["end_time"] - metrics["start_time"]
            return None, metrics

        objective_effective = _clean(
            value_or_default(
                objective_in,
                st.session_state.get("placeholder_objective", PLACEHOLDER_OBJECTIVE),
            )
        )
        rules_effective = _clean(
            value_or_default(
                rules_in, st.session_state.get("placeholder_rules", PLACEHOLDER_RULES)
            )
        )
        calculations_effective = _clean(
            value_or_default(
                calculations_in,
                st.session_state.get("placeholder_calculations", PLACEHOLDER_CALCULATIONS),
            )
        )
        question_effective = _clean(
            value_or_default(
                question_in,
                st.session_state.get("placeholder_question", PLACEHOLDER_QUESTION),
            )
        )

        cache_key = hashlib.md5(
            f"{objective_effective}|{rules_effective}|{calculations_effective}|{question_effective}|{metrics['technical_level']}".encode()
        ).hexdigest()

        if cache_key in st.session_state["response_cache"]:
            logger.info(f"[CACHE] Using cached response for key: {cache_key[:8]}...")
            cached_data = st.session_state["response_cache"][cache_key]
            metrics["cached"] = True
            metrics["end_time"] = time.time()
            metrics["duration"] = metrics["end_time"] - metrics["start_time"]
            return cached_data["response"], metrics

        prompt = build_user_prompt(
            objective_effective, rules_effective, calculations_effective, question_effective
        )

        try:
            logger.info(
                f"[LLM] Generating response for pregunta: {question_effective[:80]}..."
            )
            is_custom = "system_prompt_override" in st.session_state
            logger.info(f"[LLM] Using {'custom' if is_custom else 'default'} system prompt")

            result = llm.invoke(
                [
                    SystemMessage(
                        content=(
                            "CRÍTICO: Responde ÚNICAMENTE con información del contexto proporcionado. "
                            "NO inventes datos, números, métricas o decisiones. "
                            "Si el contexto dice 'no sé', debes responder que falta esa información. "
                            "Los ejemplos en el prompt son SOLO para formato, NO uses sus datos. "
                            "Responde en el formato exacto indicado. "
                            "No incluyas prefacios ni metatexto como 'Entendido', 'Estoy listo', 'A continuación', etc."
                        )
                    ),
                    HumanMessage(content=prompt),
                ],
                config={"configurable": {"model_kwargs": {}}},
            )
            response = (result.content or "").strip()

            if hasattr(result, "response_metadata"):
                metrics["tokens"] = result.response_metadata.get("token_usage", {})

            metrics["end_time"] = time.time()
            metrics["duration"] = metrics["end_time"] - metrics["start_time"]

            st.session_state["response_cache"][cache_key] = {
                "response": response,
                "timestamp": metrics["timestamp"],
                "metrics": metrics.copy(),
            }

            logger.info(
                f"[LLM] Response generated in {metrics['duration']:.2f}s: {response[:80]}..."
            )
            return response, metrics
        except requests.exceptions.ConnectionError as e:
            logger.error(f"[ERROR] Connection error to LLM endpoint: {e}")
            metrics["error"] = str(e)
            metrics["end_time"] = time.time()
            metrics["duration"] = metrics["end_time"] - metrics["start_time"]
            return None, metrics
        except requests.exceptions.Timeout as e:
            logger.error(f"[ERROR] Timeout error with LLM endpoint: {e}")
            metrics["error"] = str(e)
            metrics["end_time"] = time.time()
            metrics["duration"] = metrics["end_time"] - metrics["start_time"]
            return None, metrics
        except Exception as e:
            logger.error(f"[ERROR] Falló la generación de respuesta: {e}")
            metrics["error"] = str(e)
            metrics["end_time"] = time.time()
            metrics["duration"] = metrics["end_time"] - metrics["start_time"]
            return None, metrics


    st.divider()

    tab1, tab2 = st.tabs(["💬 Respuesta Individual", "🔄 Modo Comparación (3 Niveles)"])

    with tab1:
        col_btn1, col_btn2 = st.columns([3, 1])
        with col_btn1:
            generate_btn = st.button(
                "🚀 Generar respuesta",
                type="primary",
                disabled=(llm is None),
                key="generate_single",
                help="💡 Genera una explicación usando el nivel técnico seleccionado. Las respuestas se almacenan en caché para consultas repetidas.",
                use_container_width=True,
            )
        with col_btn2:
            if st.session_state.get("conversation_history"):
                total_conversations = len(st.session_state["conversation_history"])
                st.metric(
                    "💬 Total",
                    total_conversations,
                    help="Número total de conversaciones generadas",
                )

        if generate_btn:
            progress_bar = st.progress(0, text="🔄 Inicializando...")
            status_text = st.empty()

            progress_bar.progress(20, text="📝 Construyendo prompt...")
            status_text.info("⚙️ Preparando contexto para el modelo...")
            time.sleep(0.3)

            progress_bar.progress(40, text="🤖 Consultando al modelo...")
            status_text.info(
                f"🎚️ Usando Nivel {st.session_state.get('technical_level', 1)} - {['Lenguaje Común', 'Profesional', 'Técnico RL'][st.session_state.get('technical_level', 1) - 1]}"
            )

            answer, metrics = generate_response_from_inputs(
                objective, rules, calculations, question
            )

            progress_bar.progress(80, text="✅ Procesando respuesta...")
            status_text.success(
                f"{'💾 Respuesta recuperada del caché' if metrics.get('cached') else '🆕 Respuesta generada'} en {metrics['duration']:.2f}s"
            )
            time.sleep(0.5)

            progress_bar.progress(100, text="✨ ¡Completado!")
            time.sleep(0.3)
            progress_bar.empty()
            status_text.empty()

            if answer is None:
                st.error("⚠️ Ocurrió un error al llamar al LLM.")
                if "error" in metrics:
                    with st.expander("🔍 Detalles del error", expanded=True):
                        st.error(f"**Error:** {metrics['error']}")
                        st.info(
                            """💡 **Posibles soluciones:**
                        - Verifica que las variables de entorno estén configuradas correctamente
                        - Revisa tu conexión a internet
                        - Confirma que el modelo esté disponible
                        - Intenta de nuevo en unos momentos"""
                        )
            else:
                question_text = value_or_default(
                    question,
                    st.session_state.get("placeholder_question", PLACEHOLDER_QUESTION),
                )
                st.session_state["conversation_history"].append(
                    {
                        "timestamp": metrics["timestamp"],
                        "question": question_text,
                        "answer": answer,
                        "metrics": metrics,
                        "technical_level": metrics["technical_level"],
                    }
                )
                st.session_state["metrics_history"].append(metrics)

                st.markdown("### 💬 Respuesta")

                col_m1, col_m2, col_m3, col_m4 = st.columns(4)
                with col_m1:
                    st.metric(
                        "⏱️ Tiempo",
                        f"{metrics['duration']:.2f}s",
                        help="⏱️ Tiempo total de generación (incluye llamada al modelo y procesamiento)",
                    )
                with col_m2:
                    level_names = {1: "Común", 2: "Profesional", 3: "Técnico"}
                    st.metric(
                        "🎚️ Nivel",
                        level_names.get(metrics["technical_level"], "N/A"),
                        help=f"🎚️ Nivel técnico usado: {metrics['technical_level']} - Determina el vocabulario y complejidad de la respuesta",
                    )
                with col_m3:
                    cache_icon = "💾" if metrics["cached"] else "🆕"
                    cache_status = "Sí" if metrics["cached"] else "No"
                    cache_delta = "Instantáneo" if metrics["cached"] else None
                    st.metric(
                        f"{cache_icon} Caché",
                        cache_status,
                        delta=cache_delta,
                        help="💾 Indica si la respuesta se recuperó del caché (más rápido) o se generó nuevamente",
                    )
                with col_m4:
                    if "tokens" in metrics:
                        total_tokens = metrics["tokens"].get("total_tokens", "N/A")
                        st.metric(
                            "🔤 Tokens",
                            total_tokens,
                            help="🔤 Número total de tokens procesados (entrada + salida). Afecta el costo de la API.",
                        )
                    else:
                        st.metric(
                            "🔤 Tokens",
                            "N/A",
                            help="🔤 Información de tokens no disponible para este modelo",
                        )

                st.divider()

                with st.chat_message("user"):
                    st.markdown(question_text)
                with st.chat_message("assistant"):
                    st.markdown(answer)

    with tab2:
        st.info(
            """🔄 **Modo Comparación Avanzado**
    
        Este modo genera respuestas simultáneamente en los 3 niveles técnicos:
        - 🗣️ **Nivel 1**: Lenguaje común para público general
        - 🏗️ **Nivel 2**: Terminología profesional de urbanismo
        - 🤖 **Nivel 3**: Vocabulario técnico de RL/ML
    
        Útil para ver cómo cambia la explicación según la audiencia."""
        )

        comparison_btn = st.button(
            "🔄 Generar comparación (3 niveles)",
            type="primary",
            disabled=(llm is None),
            key="generate_comparison",
            help="💡 Genera 3 respuestas simultáneas (una por cada nivel técnico) para comparar vocabularios y enfoques.",
            use_container_width=True,
        )

        if comparison_btn:
            progress_bar = st.progress(0, text="🔄 Inicializando comparación...")
            status_container = st.empty()

            responses = {}
            all_metrics = {}

            level_names = {
                1: "🗣️ Nivel 1: Lenguaje Común",
                2: "🏗️ Nivel 2: Profesional",
                3: "🤖 Nivel 3: Técnico RL",
            }

            for idx, level in enumerate([1, 2, 3], 1):
                progress = int((idx - 1) / 3 * 100)
                progress_bar.progress(progress, text=f"⚙️ Generando {level_names[level]}...")

                with status_container:
                    st.info(f"🔄 Procesando nivel {idx}/3: {level_names[level]}")

                original_level = st.session_state.get("technical_level", 1)
                st.session_state["technical_level"] = level

                answer, metrics = generate_response_from_inputs(
                    objective, rules, calculations, question
                )

                responses[level] = answer
                all_metrics[level] = metrics

                cache_status = "💾 (caché)" if metrics.get("cached") else "🆕 (nueva)"
                with status_container:
                    st.success(
                        f"✅ {level_names[level]} completado {cache_status} - {metrics['duration']:.2f}s"
                    )
                time.sleep(0.3)

                st.session_state["technical_level"] = original_level

            progress_bar.progress(100, text="✨ ¡Comparación completada!")
            time.sleep(0.5)
            progress_bar.empty()
            status_container.empty()

            st.markdown("### 🔄 Comparación de Respuestas")

            st.markdown("#### 📊 Resumen de Métricas")
            col_sum1, col_sum2, col_sum3 = st.columns(3)

            level_names = {
                1: "Nivel 1: Lenguaje Común",
                2: "Nivel 2: Profesional",
                3: "Nivel 3: Técnico RL",
            }

            for idx, level in enumerate([1, 2, 3]):
                with [col_sum1, col_sum2, col_sum3][idx]:
                    st.markdown(f"**{level_names[level]}**")
                    m = all_metrics[level]
                    st.metric("⏱️ Tiempo", f"{m['duration']:.2f}s")
                    cache_text = "💾 Cache" if m["cached"] else "🆕 Nueva"
                    st.caption(cache_text)
                    if "tokens" in m:
                        st.caption(f"🔤 {m['tokens'].get('total_tokens', 'N/A')} tokens")

            st.divider()

            col_r1, col_r2, col_r3 = st.columns(3)

            for idx, level in enumerate([1, 2, 3]):
                with [col_r1, col_r2, col_r3][idx]:
                    st.markdown(f"#### {level_names[level]}")
                    if responses[level]:
                        with st.container(border=True):
                            st.markdown(responses[level])
                    else:
                        st.error("Error generando respuesta")
                        if "error" in all_metrics[level]:
                            st.caption(f"Error: {all_metrics[level]['error']}")

    st.divider()
    with st.expander("📜 Historial de Conversación", expanded=False):
        st.caption(
            "💡 **Tip:** Aquí se guardan todas tus consultas anteriores con sus métricas. Útil para revisar respuestas pasadas o analizar patrones."
        )
        if st.session_state["conversation_history"]:
            col_clear1, col_clear2 = st.columns([3, 1])
            with col_clear2:
                if st.button("🗑️ Limpiar historial"):
                    st.session_state["conversation_history"] = []
                    st.session_state["metrics_history"] = []
                    st.rerun()

            st.markdown(
                f"**Total de conversaciones:** {len(st.session_state['conversation_history'])}"
            )
            st.divider()

            for idx, entry in enumerate(reversed(st.session_state["conversation_history"])):
                with st.container(border=True):
                    col_h1, col_h2, col_h3 = st.columns([2, 1, 1])
                    with col_h1:
                        st.caption(f"🕐 {entry['timestamp']}")
                    with col_h2:
                        st.caption(f"🎚️ Nivel {entry['technical_level']}")
                    with col_h3:
                        cached_text = (
                            "💾 Cache" if entry["metrics"]["cached"] else "🆕 Nueva"
                        )
                        st.caption(cached_text)

                    st.markdown("**Pregunta:**")
                    st.info(entry["question"])
                    st.markdown("**Respuesta:**")
                    st.success(entry["answer"])

                    m = entry["metrics"]
                    st.caption(
                        f"⏱️ Tiempo: {m['duration']:.2f}s | 🔤 Tokens: {m.get('tokens', {}).get('total_tokens', 'N/A')}"
                    )
        else:
            st.info(
                "No hay conversaciones en el historial aún. Genera una respuesta para empezar."
            )

    with st.expander("💾 Estadísticas de Caché", expanded=False):
        st.caption(
            "💡 **¿Qué es el caché?** El sistema guarda respuestas generadas para evitar consultas duplicadas al LLM, ahorrando tiempo y costos. Consultas idénticas retornan instantáneamente."
        )
        cache_size = len(st.session_state["response_cache"])
        st.metric("Respuestas en caché", cache_size)

        if cache_size > 0:
            if st.button("🗑️ Limpiar caché"):
                st.session_state["response_cache"] = {}
                st.success("✓ Caché limpiado")
                st.rerun()

            st.markdown("**Entradas en caché:**")
            for key, value in st.session_state["response_cache"].items():
                with st.container(border=True):
                    st.caption(f"🔑 Key: `{key[:16]}...`")
                    st.caption(f"🕐 Timestamp: {value['timestamp']}")
                    st.caption(f"⏱️ Tiempo original: {value['metrics']['duration']:.2f}s")
        else:
            st.info("No hay respuestas en caché aún.")

with second_tab:
    st.subheader("🤖 Explicación General de Reinforcement Learning")
    st.info("💡 Pregunta sobre conceptos generales de RL: Q-Learning, DQN, políticas, funciones de valor, etc.")
    
    if "rl_history" not in st.session_state:
        st.session_state["rl_history"] = []
    if "rl_cache" not in st.session_state:
        st.session_state["rl_cache"] = {}
    
    rl_question = st.text_area(
        "Tu pregunta sobre Reinforcement Learning",
        placeholder="Ej: ¿Qué es Q-Learning? ¿Cómo funciona DQN? ¿Qué es una política epsilon-greedy?",
        height=120,
        key="rl_question"
    )
    
    if st.button("🚀 Obtener explicación", type="primary", key="rl_submit"):
        if not rl_question.strip():
            st.warning("⚠️ Por favor ingresa una pregunta")
        elif not llm:
            st.error("⚠️ LLM no está inicializado")
        else:
            cache_key = hashlib.md5(rl_question.strip().encode()).hexdigest()
            
            if cache_key in st.session_state["rl_cache"]:
                cached_data = st.session_state["rl_cache"][cache_key]
                st.success("💾 Respuesta recuperada del caché")
                
                with st.chat_message("user"):
                    st.markdown(rl_question)
                with st.chat_message("assistant"):
                    st.markdown(cached_data["response"])
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("⏱️ Tiempo", f"{cached_data['metrics']['duration']:.2f}s")
                with col2:
                    st.metric("💾 Caché", "Sí", delta="Instantáneo")
            else:
                progress_bar = st.progress(0, text="🔄 Generando explicación...")
                
                rl_prompt = f"""Eres un experto profesor de Reinforcement Learning (RL) con amplia experiencia en investigación y aplicaciones prácticas.

OBJETIVO: Explicar ÚNICAMENTE conceptos de Reinforcement Learning. Si la pregunta NO está relacionada con RL, responde: "Esta pregunta no está relacionada con Reinforcement Learning. Por favor pregunta sobre temas de RL como Q-Learning, DQN, políticas, funciones de valor, etc."

TEMAS QUE DOMINAS (SOLO ESTOS):
• Fundamentos: MDP, estados, acciones, recompensas, políticas, funciones de valor
• Algoritmos clásicos: Q-Learning, SARSA, TD-Learning, Monte Carlo
• Deep RL: DQN, Double DQN, Dueling DQN, Rainbow
• Policy-based: REINFORCE, Actor-Critic, A3C, PPO, TRPO
• Exploración: ε-greedy, UCB, Thompson Sampling, curiosity-driven
• Matemáticas: Ecuación de Bellman, convergencia, optimalidad
• Arquitecturas: MLPs, CNNs para RL, experience replay, target networks

TEMAS PROHIBIDOS (RECHAZA ESTAS PREGUNTAS):
• Supervised Learning, clasificación, regresión
• NLP, transformers, LLMs (a menos que se usen EN RL)
• Computer Vision general (a menos que sea para estados en RL)
• Temas no relacionados con ML/RL

ESTILO DE RESPUESTA (SOLO SI ES TEMA DE RL):
1. Definición concisa del concepto
2. Intuición práctica (¿para qué sirve?)
3. Formalización matemática (cuando aplique)
4. Ejemplo concreto o pseudocódigo
5. Ventajas/desventajas o casos de uso

REGLAS:
- PRIMERO verifica si la pregunta es sobre RL. Si NO lo es, recházala educadamente
- Usa notación matemática estándar: π (política), Q(s,a), V(s), γ (discount), α (learning rate)
- Sé preciso pero accesible
- Máximo 400 palabras por respuesta

PREGUNTA DEL USUARIO:
{rl_question}

RESPUESTA:"""
                
                try:
                    start_time = time.time()
                    progress_bar.progress(50, text="🤖 Consultando al modelo...")
                    
                    result = llm.invoke([HumanMessage(content=rl_prompt)])
                    response = (result.content or "").strip()
                    
                    end_time = time.time()
                    duration = end_time - start_time
                    
                    metrics = {
                        "duration": duration,
                        "cached": False,
                        "timestamp": datetime.now().isoformat()
                    }
                    
                    if hasattr(result, "response_metadata"):
                        metrics["tokens"] = result.response_metadata.get("token_usage", {})
                    
                    st.session_state["rl_cache"][cache_key] = {
                        "response": response,
                        "metrics": metrics
                    }
                    st.session_state["rl_history"].append({
                        "question": rl_question,
                        "response": response,
                        "metrics": metrics
                    })
                    
                    progress_bar.progress(100, text="✨ ¡Completado!")
                    time.sleep(0.3)
                    progress_bar.empty()
                    
                    with st.chat_message("user"):
                        st.markdown(rl_question)
                    with st.chat_message("assistant"):
                        st.markdown(response)
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("⏱️ Tiempo", f"{duration:.2f}s")
                    with col2:
                        st.metric("💾 Caché", "No")
                    with col3:
                        if "tokens" in metrics:
                            st.metric("🔤 Tokens", metrics["tokens"].get("total_tokens", "N/A"))
                
                except Exception as e:
                    progress_bar.empty()
                    st.error(f"⚠️ Error: {e}")
    
    st.divider()
    
    with st.expander("📜 Historial de Preguntas RL", expanded=False):
        if st.session_state["rl_history"]:
            if st.button("🗑️ Limpiar historial RL"):
                st.session_state["rl_history"] = []
                st.rerun()
            
            for idx, entry in enumerate(reversed(st.session_state["rl_history"])):
                with st.container(border=True):
                    st.caption(f"🕐 {entry['metrics']['timestamp']}")
                    st.markdown("**Pregunta:**")
                    st.info(entry["question"])
                    st.markdown("**Respuesta:**")
                    st.success(entry["response"])
                    st.caption(f"⏱️ {entry['metrics']['duration']:.2f}s")
        else:
            st.info("No hay preguntas en el historial aún")
