import os
import requests
import json

# =========================
# CONFIG
# =========================
OLLAMA_MODEL = "llama2:latest"
OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_TIMEOUT = 120  # seconds

COLOR_RESULT_MAP = {
    "green": "PASS",
    "red": "FAIL",
    "blue": "PROCESSING",
}

COLOR_REASON_MAP = {
    "green": "Green indicates healthy system status.",
    "red": "Red indicates error or failure condition.",
    "blue": "Blue indicates initialization or processing state.",
}


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
    for result in valid_results:
        if result in upper_text:
            print(f"🔍 [DEBUG] Found result keyword: {result}")
            return result
    
    # If nothing found, return UNKNOWN
    print(f"🔍 [DEBUG] No result found, returning UNKNOWN")
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
    
    system_prompt = "You are a precise industrial LED inspection assistant."
    full_prompt = f"{system_prompt}\n\n{prompt}"
    
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": full_prompt,
        "stream": False,
        "options": {
            "temperature": 0.1,  # 🔥 Deterministic output
            "top_p": 0.95,
            "num_predict": 50,  # Equivalent to max_new_tokens
            "top_k": 40,
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
    # LLM INTERPRETS ALL DATA
    # =========================
    color = color.lower()

    # =========================
    # SIMPLIFIED PROMPT WITH EXAMPLES
    # =========================
    prompt = (
        "You are an LED inspection system. Answer with exactly this format:\n"
        "Result: <PASS, FAIL, PROCESSING, or UNKNOWN>\n"
        "Explanation: <short sentence>\n\n"
        "Important: If the LED color is green, the result must be PASS.\n"
        "If the LED color is red, the result must be FAIL.\n"
        "If the LED color is blue, the result must be PROCESSING.\n"
        "Do not contradict the LED color.\n\n"
        "Examples:\n"
        "Input: green LED, 500 lux\n"
        "Result: PASS\n"
        "Explanation: System healthy and operating normally.\n\n"
        "Input: red LED, 100 lux\n"
        "Result: FAIL\n"
        "Explanation: System error detected.\n\n"
        "Input: blue LED, 300 lux\n"
        "Result: PROCESSING\n"
        "Explanation: System initializing.\n\n"
        f"Input: {color} LED, {lux:.1f} lux\n"
    )

    print(f"🧠 Calling Ollama to interpret: color={color}, lux={lux}...")
    raw_response = generate_text(prompt)
    print(f"🔍 [DEBUG] Raw response: {repr(raw_response)}")
    
    # Extract result from LLM response
    extracted_result = _extract_result_from_response(raw_response)
    print(f"🔍 [DEBUG] Extracted result: {extracted_result}")

    # Enforce the expected result for known LED colors
    if color in COLOR_RESULT_MAP:
        expected_result = COLOR_RESULT_MAP[color]
        if extracted_result != expected_result:
            print(f"🔍 [DEBUG] Overriding {extracted_result} -> {expected_result} for color {color}")
            return f"Result: {expected_result}\nExplanation: {COLOR_REASON_MAP[color]}"

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
    
    # Build simplified prompt with examples
    prompt = (
        "You are an LED inspection system. Answer with exactly this format:\n"
        "Result: <PASS, FAIL, PROCESSING, or UNKNOWN>\n"
        "Explanation: <short sentence>\n\n"
        "Examples:\n"
        "Input: Result PASS, green LED, 500 lux\n"
        "Result: PASS\n"
        "Explanation: System healthy and operating normally.\n\n"
        "Input: Result FAIL, red LED, 100 lux\n"
        "Result: FAIL\n"
        "Explanation: System error detected.\n\n"
        "Input: Result PROCESSING, blue LED, 300 lux\n"
        "Result: PROCESSING\n"
        "Explanation: System initializing.\n\n"
    )
    
    if context:
        prompt += f"Input: Result {result}, {color} LED, {lux:.1f} lux, context: {context}\n"
    else:
        prompt += f"Input: Result {result}, {color} LED, {lux:.1f} lux\n"
    
    print(f"🧠 Calling Ollama to interpret result: {result} (color={color}, lux={lux})...")
    raw_response = generate_text(prompt)
    print(f"🔍 [DEBUG] Raw response: {repr(raw_response)}")
    
    # Extract and format the interpretation
    response = _extract_formatted_response(raw_response, result)
    print(f"🔍 [DEBUG] Formatted response: {repr(response)}")
    return response
