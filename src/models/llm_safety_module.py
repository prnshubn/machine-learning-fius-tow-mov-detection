"""
This module integrates a real Local LLM (via Ollama) to interpret the severity of the 
situation based on Time-To-Collision (TTC) and kinematics. It generates 
human-readable safety protocols and reasoned assessments.

To use this:
1. Install Ollama from https://ollama.com
2. Run 'ollama run llama3'
"""

import requests
import json

def generate_safety_protocol(ttc, velocity, acceleration, model="llama3"):
    """
    Generates a safety protocol and severity assessment using a real Local LLM.
    Args:
        ttc (float): Predicted Time-To-Collision in seconds.
        velocity (float): Current velocity (negative means approaching).
        acceleration (float): Current acceleration.
        model (str): The Ollama model to use.
    Returns:
        dict: A dictionary containing severity, protocol, and message.
    """
    
    # --- Context & Prompt Construction ---
    # Velocity is negative when approaching
    status = "APPROACHING" if velocity < 0 else "STATIONARY/RECEDING"
    
    prompt = f"""
    You are an AI Safety Controller for an autonomous industrial robot.
    Based on the following sensor data, assess the risk and provide a protocol:
    
    - Object Status: {status}
    - Time-To-Collision: {ttc:.2f} seconds
    - Current Velocity: {velocity:.2f} units/s
    - Current Acceleration: {acceleration:.2f} units/s^2
    
    Provide your response in JSON format with exactly these four keys:
    1. "severity_level": (Choose one: LOW, MEDIUM, HIGH, CRITICAL)
    2. "recommended_protocol": (A short, clear action command)
    3. "assessment": (A brief, one-sentence explanation of the physics/risk)
    4. "llm_reasoning": (A concise explanation of why you chose this protocol)
    """

    # --- Ollama API Call ---
    url = "http://localhost:11434/api/generate"
    payload = {
        "model": model,
        "prompt": prompt,
        "format": "json",
        "stream": False
    }

    try:
        # Attempt to reach the local Ollama API
        # Increased timeout to 30s because the first run can be slow on some machines
        response = requests.post(url, json=payload, timeout=30)
        
        if response.status_code == 200:
            # Parse the JSON response from Ollama
            result_json = json.loads(response.json()['response'])
            
            # Add the raw metrics back into the final response
            result_json["raw_metrics"] = {
                "ttc": f"{ttc:.2f}s",
                "velocity": f"{velocity:.2f} units/s",
                "acceleration": f"{acceleration:.2f} units/s^2"
            }
            return result_json
            
    except Exception as e:
        # Fallback logic if Ollama is not running or fails
        print(f"--- Note: Ollama (LLM) not detected or error: {e} ---")
        print("--- Falling back to rule-based safety logic ---")
        
        # Simple rule-based logic (similar to before)
        severity = "LOW"
        action = "Monitor closely."
        explanation = "Object state is stable."
        
        if velocity < 0: # Approaching
            if ttc < 0.5:
                severity = "CRITICAL"
                action = "EMERGENCY BRAKE"
                explanation = f"Impact in {ttc:.2f}s!"
            elif ttc < 2.0:
                severity = "HIGH"
                action = "Hard Braking"
                explanation = f"Collision likely in {ttc:.2f}s."
            elif ttc < 5.0:
                severity = "MEDIUM"
                action = "Slow down"
                explanation = "Object approaching at medium range."
        
        return {
            "severity_level": severity,
            "recommended_protocol": action,
            "assessment": explanation,
            "llm_reasoning": "Ollama fallback (Rule-based).",
            "raw_metrics": {
                "ttc": f"{ttc:.2f}s",
                "velocity": f"{velocity:.2f} units/s"
            }
        }

def print_safety_report(response):
    """
    Prints a formatted safety report to the console.
    """
    print("\n" + "="*50)
    print(f" [AI SAFETY PROTOCOL GENERATOR] ".center(50, "="))
    print(f" SEVERITY:     [{response.get('severity_level', 'UNKNOWN')}]")
    print(f" PROTOCOL:     {response.get('recommended_protocol', 'N/A')}")
    print(f" ASSESSMENT:   {response.get('assessment', 'N/A')}")
    print(f" REASONING:    {response.get('llm_reasoning', 'N/A')}")
    print("-" * 50)
    print(f" METRICS: TTC={response.get('raw_metrics', {}).get('ttc')} | Vel={response.get('raw_metrics', {}).get('velocity')}")
    print("="*50 + "\n")

if __name__ == "__main__":
    # Test with critical scenario
    print("--- Testing AI Safety Module (Local LLM Integration) ---")
    
    # Scenario: Impact in 0.4 seconds
    report = generate_safety_protocol(0.4, -50.0, -2.0)
    print_safety_report(report)
