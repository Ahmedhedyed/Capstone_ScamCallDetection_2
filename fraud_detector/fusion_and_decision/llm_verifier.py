import os
import json
from openai import OpenAI
from config import LOCAL_LLM_MODEL_NAME # Import the model name from your new config file

# Read LLM server URL from environment variable, fall back to default GPT4All address
LOCAL_LLM_BASE_URL = os.getenv("LOCAL_LLM_BASE_URL", "http://localhost:4891/v1")

class LLMVerifier:
    """
    Uses a local Large Language Model (via GPT4All server) as a reasoning engine
    to verify fraud potential based on context.
    """
    def __init__(self):
        """
        Initializes the client to connect to the local GPT4All API server.
        """
        try:
            # Point the OpenAI client to your local server instead of the internet
            self.client = OpenAI(
                base_url=LOCAL_LLM_BASE_URL,
                api_key="not-needed" # API key is not required for the local server
            )
            print("✓ LLM Verifier initialized (connected to local GPT4All server).")
        except Exception as e:
            raise ConnectionError(
                "Could not connect to the local GPT4All server. "
                "Please ensure the GPT4All application is running with the API Server enabled."
            ) from e
    
    def _create_prompt(self, transcription, features):
        """
        Creates a clear, instruction-based prompt suitable for local models like Mistral.
        """
        prompt = f"""
        [INST]
        You are an expert fraud analyst. Your task is to analyze a phone call transcript and a list of detected signals.
        Based on the context, provide a final fraud probability and a brief reason.
        Respond ONLY with a valid JSON object containing two keys: "probability" (a float from 0.0 to 1.0) and "reasoning" (a string).

        **Pre-Detected Signals:**
        {json.dumps(features, indent=2)}

        **Full Call Transcript:**
        "{transcription}"

        Provide your JSON response now.
        [/INST]
        """
        return prompt

    def verify(self, transcription, features):
        """
        Sends the data to the local LLM for verification and returns the structured response.
        """
        if not transcription.strip():
            return {"probability": 0.0, "reasoning": "No valid text to analyze."}

        prompt = self._create_prompt(transcription, features)

        try:
            response = self.client.chat.completions.create(
                # Use the model name from your config file
                model=LOCAL_LLM_MODEL_NAME, 
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1, # Keep the output consistent and factual
                response_format={"type": "json_object"} # Ask the model to guarantee a JSON output
            )
            
            response_json = json.loads(response.choices[0].message.content)
            
            # Basic validation of the LLM's response
            if 'probability' in response_json and 'reasoning' in response_json:
                return response_json
            else:
                print("Warning: Local LLM response was invalid JSON. Missing required keys.")
                return None

        except Exception as e:
            print(f"An error occurred during local LLM verification: {e}")
            return None