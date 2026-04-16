import os
import requests
import json

# =========================
# CONFIG
# =========================
OLLAMA_MODEL = "llama2:latest"
OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_TIMEOUT = 120  # seconds


def _extract_formatted_response(raw_text: str, expected_result: str) -> str:
    text = (raw_text or "").strip()

    result_line = None
    explanation_line = None

    for line in text.splitlines():
        if line.lower().startswith("result:"):
            result_line = line.split(":", 1)[-1].strip()
        elif line.lower().startswith("explanation:"):
            explanation_line = line.split(":", 1)[-1].strip()
            break

    # Normalize bad or non-sentence explanations from a noisy model.
    invalid_explanation = False
    if explanation_line is None:
        invalid_explanation = True
    else:
        normalized = explanation_line.lower().strip()
        if normalized == "" or normalized == expected_result.lower():
            invalid_explanation = True
        if normalized.startswith("example") or normalized.startswith("explanation"):
            invalid_explanation = True
        if "=" in normalized or "input(" in normalized or "prompt" in normalized:
            invalid_explanation = True

    if invalid_explanation:
        explanation_line = {
            "UNKNOWN": "No prediction.",
            "The model did not provide a valid explanation.": "The model did not provide a valid explanation.",
        }.get(expected_result, expected_result)

    return f"Result: {expected_result}\nExplanation: {explanation_line}"


def _extract_result_from_response(raw_text: str) -> str:
    """
    Extract the result status from LLM response.
    Returns the first valid result found, or UNKNOWN if none found.
    """
    text = (raw_text or "").strip()
    valid_results = {"PASS", "FAIL", "PROCESSING", "UNKNOWN"}
    
    # First, try to find "Result: <status>" format
    for line in text.splitlines():
        line_lower = line.lower().strip()
        if line_lower.startswith("result:"):
            status = line.split(":", 1)[-1].strip().upper()
            # Take only the first word if there are multiple words
            status_first_word = status.split()[0] if status else ""
            if status_first_word in valid_results:
                print(f"🔍 [DEBUG] Found result in line: {line}")
                return status_first_word
    
    # Fallback: search for any valid result keyword in the response
    upper_text = text.upper()
    for result in sorted(valid_results, key=lambda x: -len(x)):  # Check longer words first
        if result in upper_text:
            print(f"🔍 [DEBUG] Found result keyword: {result}")
            return result
    
    # If nothing found, return UNKNOWN
    print(f"🔍 [DEBUG] No result found in response. Text was: {text}")
    return "UNKNOWN"

# =========================
# CHECK OLLAMA CONNECTION
# =========================
def check_ollama_connection():
    """Check if Ollama service is running and accessible."""
    try:
        response = requests.get(f"http://localhost:11434/api/tags", timeout=5)
        response.raise_for_status()
        models = response.json().get("models", [])
        model_names = [m.get("name", "") for m in models]
        
        if OLLAMA_MODEL in model_names:
            print(f"✅ Ollama is running and {OLLAMA_MODEL} model is available.")
            return True
        else:
            print(f"⚠️ Ollama is running but {OLLAMA_MODEL} model not found.")
            print(f"   Available models: {model_names}")
            return False
    except requests.RequestException as e:
        print(f"❌ Cannot connect to Ollama at {OLLAMA_URL}")
        print(f"   Make sure Ollama is running: ollama serve")
        print(f"   Error: {e}")
        return False


# =========================
# GENERATE RESPONSE
# =========================
def generate_text(prompt: str) -> str:
    """Generate text using Ollama API."""
    
    system_prompt = "You are a precise industrial LED inspection assistant. Always follow the format instructions exactly. Answer only with the requested format, nothing else."
    full_prompt = f"{system_prompt}\n\n{prompt}"
    
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": full_prompt,
        "stream": False,
        "options": {
            "temperature": 0.1,  # 🔥 Low temperature for consistent output
            "top_p": 0.9,        # Narrower sampling for focus
            "num_predict": 100,  # More room for structured output
            "top_k": 20,         # Reduced from 40 for more deterministic output
            "repeat_penalty": 1.2,  # Avoid repetition
        }
    }
    
    try:
        print(f"🧠 Calling Ollama ({OLLAMA_MODEL}) to generate response...")
        response = requests.post(OLLAMA_URL, json=payload, timeout=OLLAMA_TIMEOUT)
        response.raise_for_status()
        result = response.json()
        text = result.get("response", "").strip()
        
        if not text:
            print("⚠️ Ollama returned empty response")
            return "UNKNOWN"
        
        return text
        
    except requests.Timeout:
        print(f"❌ Ollama request timed out (>{OLLAMA_TIMEOUT}s)")
        return "UNKNOWN"
    except requests.ConnectionError:
        print(f"❌ Cannot connect to Ollama at {OLLAMA_URL}")
        print("   Make sure Ollama is running: ollama serve")
        return "UNKNOWN"
    except requests.RequestException as e:
        print(f"❌ Ollama API error: {e}")
        return "UNKNOWN"
    except json.JSONDecodeError:
        print(f"❌ Invalid JSON response from Ollama")
        return "UNKNOWN"


# =========================
# MAIN FUNCTION
# =========================
def interpret_led(color: str, lux: float) -> str:
    # =========================
    # LLM INTERPRETS SENSOR DATA
    # =========================
    color = color.lower()

    # =========================
    # FLEXIBLE PROMPT - LET LLM DECIDE BASED ON SENSOR DATA
    # =========================
    prompt = (
        "You are an LED inspection system analyzing live sensor data.\n"
        "YOUR TASK: Based on the LED color AND brightness value, determine the system status.\n\n"
        "INTERPRETATION GUIDELINES:\n"
        "- Green LED: Usually indicates healthy status, but very low lux (< 50) may suggest hardware issue\n"
        "- Red LED: Usually indicates error, but analyze the lux context for severity\n"
        "- Blue LED: Usually indicates processing/initialization state\n"
        "- Consider both color AND lux together for your analysis\n"
        "- Make your own judgment - don't just follow color blindly\n\n"
        "RESPONSE FORMAT (exactly 2 lines, no deviation):\n"
        "Result: <PASS, FAIL, PROCESSING, or UNKNOWN>\n"
        "Explanation: <one sentence analyzing color + lux context>\n\n"
        "EXAMPLES (study the reasoning):\n"
        "Sensor Input: LED=GREEN, Lux=750.5\n"
        "Result: PASS\n"
        "Explanation: Green LED at 750.5 lux indicates healthy system operating at normal brightness.\n\n"
        "Sensor Input: LED=GREEN, Lux=25.0\n"
        "Result: FAIL\n"
        "Explanation: Green LED but critically low brightness (25 lux) suggests hardware malfunction.\n\n"
        "Sensor Input: LED=RED, Lux=200.0\n"
        "Result: FAIL\n"
        "Explanation: Red LED at 200 lux indicates system error with moderate brightness loss.\n\n"
        "Sensor Input: LED=BLUE, Lux=350.0\n"
        "Result: PROCESSING\n"
        "Explanation: Blue LED at 350 lux shows system is initializing or performing diagnostics.\n\n"
        "NOW ANALYZE THIS ACTUAL SENSOR DATA:\n"
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"LED Color: {color.upper()}\n"
        f"Brightness (Lux): {lux:.2f}\n"
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
        "Provide your analysis:\n"
    )

    print(f"🧠 Calling Ollama to interpret: color={color}, lux={lux}...")
    raw_response = generate_text(prompt)
    print(f"🔍 [DEBUG] Raw response: {repr(raw_response)}")
    
    # Extract result from LLM response
    extracted_result = _extract_result_from_response(raw_response)
    print(f"🔍 [DEBUG] Extracted result: {extracted_result}")
    
    response = _extract_formatted_response(raw_response, extracted_result)
    print(f"🔍 [DEBUG] Formatted response: {repr(response)}")
    return response


# =========================
# INTERPRET RESULT FUNCTION
# =========================
def interpret_result(result: str, color: str, lux: float, context: str = None) -> str:
    """
    Given a result and sensor data, ask the LLM to interpret/explain the result.
    
    Args:
        result: The result status (PASS, FAIL, PROCESSING, UNKNOWN)
        color: The LED color observed
        lux: The lux value (brightness)
        context: Optional additional context for interpretation
    
    Returns:
        Formatted response with Result and Explanation
    """
    color = color.lower()
    
    # Build flexible prompt to let LLM provide context-aware analysis
    prompt = (
        "You are an LED inspection system analyzing live sensor data.\n"
        "YOUR TASK: Based on the result status, LED color, and brightness value, provide context-aware explanation.\n\n"
        "RESPONSE FORMAT (exactly 2 lines, no deviation):\n"
        "Result: <restate the result>\n"
        "Explanation: <one sentence analyzing what the sensor combination means>\n\n"
        "EXAMPLES (note how color + lux context matters):\n"
        "Sensor Data: Result=PASS, LED=GREEN, Lux=750.5\n"
        "Result: PASS\n"
        "Explanation: Green LED at 750.5 lux confirms system is healthy and operating at full brightness.\n\n"
        "Sensor Data: Result=FAIL, LED=RED, Lux=45.2\n"
        "Result: FAIL\n"
        "Explanation: Red LED with critically low brightness (45.2 lux) indicates severe system failure.\n\n"
        "Sensor Data: Result=PROCESSING, LED=BLUE, Lux=320.8\n"
        "Result: PROCESSING\n"
        "Explanation: Blue LED at 320.8 lux shows system is in active initialization or diagnostic mode.\n\n"
        "NOW ANALYZE THIS SENSOR RESULT:\n"
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"Result: {result}\n"
        f"LED Color: {color.upper()}\n"
        f"Brightness (Lux): {lux:.2f}\n"
    )
    
    if context:
        prompt += f"Additional Context: {context}\n"
    
    prompt += (
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
        "Provide your interpretation:\n"
    )
    
    print(f"🧠 Calling Ollama to interpret result: {result} (color={color}, lux={lux})...")
    raw_response = generate_text(prompt)
    print(f"🔍 [DEBUG] Raw response: {repr(raw_response)}")
    
    # Extract and format the interpretation
    response = _extract_formatted_response(raw_response, result)
    print(f"🔍 [DEBUG] Formatted response: {repr(response)}")
    return response
