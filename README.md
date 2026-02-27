# 🏙️ Marco de Inteligencia Artificial Explicable para Decisiones de un Agente RL Urbano

------------------------------------------------------------------------

## 📋 Descripción General del Sistema

Esta plataforma web facilita la interpretación de las decisiones de un
agente de Reinforcement Learning (RL) aplicado a planificación urbana a
través de **tres niveles técnicos diferenciados**:

-   **Nivel 1 --- Público General**: Diseñado para audiencias sin
    formación técnica, utilizando lenguaje claro y accesible.
-   **Nivel 2 --- Profesional**: Orientado a arquitectos y urbanistas,
    empleando terminología especializada en diseño urbano.
-   **Nivel 3 --- Técnico RL**: Dirigido a científicos de datos e
    investigadores en ML/RL, utilizando nomenclatura algorítmica y
    computacional avanzada.

------------------------------------------------------------------------

## ✨ Capacidades Principales

### 🎚️ Estratificación Técnica Dinámica

-   Modulación automática del vocabulario según el público objetivo.
-   Tres niveles de complejidad gobernados por ingeniería de prompts
    especializada.
-   Modo comparativo que permite visualizar simultáneamente las tres
    respuestas.

### 💾 Mecanismo Inteligente de Caché

-   Sistema de caché basado en hash MD5 para evitar consultas
    redundantes a la API.
-   Recuperación instantánea de respuestas previamente procesadas.
-   Optimización significativa de latencia y costos operativos.

### 📊 Analítica y Telemetría

-   Medición de latencia por consulta.
-   Métricas de utilización de tokens (entrada/salida).
-   Indicadores de uso de caché.
-   Registro completo de conversaciones y trazabilidad.

### 🔧 Protocolo de Configuración Flexible

-   Configuración de variables de entorno desde la interfaz (UI).
-   Personalización avanzada de prompts del sistema.
-   Presets operativos predefinidos (simplificados y técnicos).
-   Alertas automáticas ante configuraciones incompletas.

### 📜 Trazabilidad y Auditoría

-   Registro exhaustivo de interacciones.
-   Marcas temporales y métricas por sesión.
-   Capacidad administrativa para limpiar historial y caché.

------------------------------------------------------------------------

# 🚀 Despliegue y Flujo Operativo

## 🔹 Requisitos Previos

-   Python 3.11 o superior.
-   Acceso a una API compatible con OpenAI (OpenAI, Azure, etc.).

------------------------------------------------------------------------

## 💻 Configuración en Entorno Local

### 1️⃣ Clonar el repositorio

``` bash
git clone https://github.com/enriquegomeztagle/urban-rl-explainer-es
```

### 2️⃣ Crear entorno virtual

``` bash
python -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate
```

### 3️⃣ Instalar dependencias

``` bash
pip install -r requirements.txt
```

### 4️⃣ Configurar variables de entorno

Crear un archivo `.env` en la raíz del proyecto:

``` env
OPENAI_API_KEY=tu_api_key_aqui
OPENAI_BASE_URL=https://api.openai.com
OPENAI_MODEL=gpt-4
```

### 5️⃣ Ejecutar la aplicación

``` bash
streamlit run app.py
```

La aplicación estará disponible en:

http://localhost:8501

------------------------------------------------------------------------

## 🐳 Contenerización con Docker

### Construir la imagen

``` bash
docker build -t rl-urbanismo-explainer .
```

### Ejecutar el contenedor

``` bash
docker run -p 8501:8501   -e OPENAI_API_KEY=tu_api_key_aqui   -e OPENAI_BASE_URL=https://api.openai.com   -e OPENAI_MODEL=gpt-4   rl-urbanismo-explainer
```

### Alternativamente usando `.env`

``` bash
docker run -p 8501:8501 --env-file .env rl-urbanismo-explainer
```

Acceder desde el navegador:

http://localhost:8501

------------------------------------------------------------------------

# 🧭 Protocolo de Uso

## 🔹 Configuración Inicial

Verificar las variables de entorno en la barra lateral y seleccionar el
nivel técnico mediante el deslizador.

## 🔹 Ingreso de Datos

Utilizar "Cargar Ejemplo" o ingresar manualmente:

-   **Objetivo del Agente**: Meta de optimización.
-   **Restricciones de Política**: Reglas operativas.
-   **Métricas Computacionales**: Cálculos realizados por el agente.
-   **Consulta**: Pregunta específica sobre la decisión.

## 🔹 Generación de Respuesta

-   **Modo Individual**: Genera una respuesta en el nivel seleccionado.
-   **Modo Comparativo**: Genera simultáneamente las tres versiones
    técnicas.

## 🔹 Análisis de Resultados

Revisar latencia, tokens, estado de caché y consultar historial.

------------------------------------------------------------------------

# 🏗️ Arquitectura del Sistema

## Componentes Principales

``` text
app.py
├── Módulo de Configuración
│   ├── Variables de Entorno
│   ├── Gestión de Estado de Sesión
│   └── Presets
│
├── Framework de Ingeniería de Prompts
│   ├── BASE_CRITICAL_RULES
│   ├── SYSTEM_PROMPT_LEVEL_CONFIG
│   └── build_system_prompt()
│
├── Pipeline de Generación
│   ├── generate_response_from_inputs()
│   ├── Integración LangChain + ChatOpenAI
│   └── Manejo de excepciones y métricas
│
└── Interfaz Streamlit
    ├── Sidebar
    ├── Selector de nivel técnico
    ├── Formulario de entrada
    ├── Tabs
    └── Expanders
```

------------------------------------------------------------------------

## 🔐 Arquitectura de Caché

-   **Clave Criptográfica**:
    `MD5(objetivo + restricciones + métricas + consulta + nivel_tecnico)`
-   **Almacenamiento**: `st.session_state`
-   **Beneficio**: Respuestas casi instantáneas y reducción de costos.

------------------------------------------------------------------------

## 🧠 Arquitectura de Síntesis de Prompts

-   Base compartida con reglas críticas.
-   Configuración específica por nivel.
-   Ensamblaje dinámico mediante `build_system_prompt(level)`.

------------------------------------------------------------------------

# 🔍 Mecanismos Avanzados

## Mitigación de Alucinaciones

-   Reglas críticas explícitas en el prompt base.
-   Validación rigurosa del contexto.
-   Instrucciones para responder "desconocido" si falta información.
-   Separación clara entre ejemplos de formato y datos reales.

## Telemetría

-   Indicadores de progreso.
-   Seguimiento en tiempo real.
-   Diferenciación visual entre caché y generación nueva.

## Manejo de Excepciones

-   Gestión robusta de errores de conexión.
-   Manejo de timeouts.
-   Mensajes descriptivos con soluciones sugeridas.

------------------------------------------------------------------------

# 📊 Métricas Disponibles

-   **Latencia de generación**
-   **Nivel técnico utilizado**
-   **Estado de caché**
-   **Uso de tokens**
-   **Timestamp**

------------------------------------------------------------------------

# 🛠️ Stack Tecnológico

-   Streamlit
-   LangChain
-   OpenAI API
-   Loguru
-   Python-dotenv

------------------------------------------------------------------------

# 📦 Dependencias

Consultar `requirements.txt`.

Principales:

-   streamlit
-   langchain-openai
-   loguru

------------------------------------------------------------------------

# 🔒 Seguridad

-   API keys mediante variables de entorno.
-   Campos sensibles con `type="password"`.
-   Sin credenciales hardcodeadas.
-   Uso recomendado de `.env`.

------------------------------------------------------------------------

# 🐛 Resolución de Problemas

### Variables faltantes

Verificar `.env` o configurar desde la barra lateral.

### Timeout de conexión

Revisar conexión y `OPENAI_BASE_URL`.

### API Key inválida

Verificar permisos y validez.

### Inconsistencias en respuesta

Limpiar caché desde el módulo correspondiente.

------------------------------------------------------------------------

# 🚧 Limitaciones

-   Caché volátil.
-   Máximo 1024 tokens (configurable).
-   Requiere conexión a internet.

------------------------------------------------------------------------

# 📄 Licencia y Uso

Proyecto desarrollado exclusivamente para investigación académica en
planificación urbana con RL.

## Aviso de Copyright

Todos los derechos reservados.

## Términos de Uso

-   Uso exclusivo para investigación y evaluación.
-   Prohibido uso comercial sin autorización.
-   Prohibida redistribución sin consentimiento.
-   Atribución obligatoria.

## Propiedad Intelectual

Trabajo original enfocado en planificación urbana y Explainable AI en
RL.

------------------------------------------------------------------------

# 📫 Contacto

Dominio: Investigación en RL aplicado a Planificación Urbana

## Autores

-   Enrique Ulises Baez Gomez Tagle
-   Daniel Adrián Contreras Olivas
-   Francisco Javier Tallabs Utrilla

GitHub: @enriquegomeztagle
