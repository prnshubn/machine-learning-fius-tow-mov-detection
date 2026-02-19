
"""
This module integrates an LLM (Language Model) to interpret the severity of the situation
based on Time-To-Collision (TTC) and kinematics, and generates human-readable safety protocols.

Main functionalities:
- Generate a severity assessment and recommended protocol based on TTC and kinematics
- Simulate LLM output for safety reporting

Note: In a production environment, this would interface with a local LLM (e.g., Llama 2 via llama.cpp)
or a hosted API. For this implementation, we use a template-based generation that simulates
the output of such a model to ensure reliability and speed without heavy dependencies.
"""

def generate_safety_protocol(ttc, velocity, acceleration):
    """
    Generates a safety protocol and severity assessment using simulated LLM logic.
    Args:
        ttc (float): Predicted Time-To-Collision in seconds.
        velocity (float): Current velocity (negative means approaching).
        acceleration (float): Current acceleration.
    Returns:
        dict: A dictionary containing severity, protocol, and message.
    """
    # --- Context Construction ---
    # Prepare the "prompt" (internal logic here)

    # Default values for severity, action, and explanation
    severity = "LOW"
    action = "Continue monitoring."
    explanation = "Object is at a safe distance or moving away."

    # Velocity is negative when approaching
    is_approaching = velocity < 0

    if not is_approaching:
        # Object is receding or stationary
        severity = "LOW"
        action = "No action required."
        explanation = "Object is receding or stationary."
    else:
        # Object is approaching
        if ttc < 0.5:
            # Imminent collision
            severity = "CRITICAL"
            action = "EMERGENCY BRAKE / EVASIVE MANEUVER"
            explanation = f"Impact imminent in {ttc:.2f}s! Velocity is high ({abs(velocity):.2f})."
        elif ttc < 2.0:
            # High risk
            severity = "HIGH"
            action = "Apply brakes immediately."
            explanation = f"Collision likely in {ttc:.2f}s. Decelerate now."
        elif ttc < 5.0:
            # Medium risk
            severity = "MEDIUM"
            action = "Reduce speed and prepare to stop."
            explanation = f"Object approaching. Impact in {ttc:.2f}s if speed is maintained."
        else:
            # Low risk, object is distant
            severity = "LOW"
            action = "Monitor closely."
            explanation = f"Object approaching but distant (TTC: {ttc:.2f}s)."

    # --- Simulated LLM Output ---
    # This structure mimics what a JSON-mode LLM might return
    response = {
        "severity_level": severity,
        "recommended_protocol": action,
        "assessment": explanation,
        "raw_metrics": {
            "ttc": f"{ttc:.2f}s",
            "velocity": f"{velocity:.2f} units/s",
            "acceleration": f"{acceleration:.2f} units/s^2"
        }
    }
    
    return response

def print_safety_report(response):
    """
    Prints a formatted safety report to the console.
    """
    print("\n=== AI SAFETY PROTOCOL GENERATOR ===")
    print(f"SEVERITY: [{response['severity_level']}]")
    print(f"PROTOCOL: {response['recommended_protocol']}")
    print(f"ASSESSMENT: {response['assessment']}")
    print("------------------------------------")
    print(f"Metrics: TTC={response['raw_metrics']['ttc']} | Vel={response['raw_metrics']['velocity']}")
    print("====================================\n")

if __name__ == "__main__":
    # Test cases
    print("--- Testing LLM Safety Module ---")
    
    print_safety_report(generate_safety_protocol(0.4, -50.0, -2.0))
    print_safety_report(generate_safety_protocol(3.5, -20.0, 0.0))
    print_safety_report(generate_safety_protocol(10.0, 5.0, 0.0))
