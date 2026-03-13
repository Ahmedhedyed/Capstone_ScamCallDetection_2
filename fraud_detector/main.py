from audio_ingestion.audio_ingester import AudioIngester
from analyzer.word_analyzer.transcriber import Transcriber
from analyzer.word_analyzer.text_feature_extractor import TextFeatureExtractor
from analyzer.audio_analyzer.acoustic_analyzer import AcousticAnalyzer
from fusion_and_decision.master_model import MasterModel
from fusion_and_decision.llm_verifier import LLMVerifier
from output.output_handler import OutputHandler

def main():
    # --- Configuration ---
    AUDIO_FILE_PATH = "samples/sample_negative.mp4"
    LLM_VERIFICATION_THRESHOLD = 0.15  # If initial score is above this, escalate to LLM

    # --- Initialization ---
    print("Initializing components...")
    
    initial_model = MasterModel()
    output_handler = OutputHandler()
    text_extractor = TextFeatureExtractor()
    transcriber = Transcriber(model_size="small", compute_type="int8")

    try:
        audio_ingester = AudioIngester(AUDIO_FILE_PATH)
        llm_verifier = LLMVerifier()
    except (FileNotFoundError, ConnectionError) as e:
        print(f"ERROR: {e}")
        return

    # --- Transcription Stage ---
    English_Fraud_Prompt = "bank, account, OTP, one-time password, transaction, credit card, debit card, CVV, security, verify, reverse, payment, fraud, alert, KYC, customer support, computer, virus."
    print(f"Using English contextual prompt: '{English_Fraud_Prompt}'")
    
    audio_chunks = audio_ingester.get_audio_chunks()
    if not audio_chunks:
        print("Could not find any speech chunks in the audio. Exiting.")
        return

    full_english_transcription = ""
    print(f"\nFound {len(audio_chunks)} speech chunks. Translating each to English...")
    for i, chunk in enumerate(audio_chunks):
        english_text = transcriber.transcribe_and_translate_chunk(
            chunk, 
            language=None,
            initial_prompt=English_Fraud_Prompt
        )
        if english_text:
            print(f'  Chunk {i+1}: Translated to "{english_text}"')
            full_english_transcription += english_text + ". "
        else:
            print(f"  Chunk {i+1}: Translation failed or filtered out.")

    print("\n--- Full English Transcription ---")
    if not full_english_transcription:
        print("Translation resulted in no text. Cannot analyze further.")
        return
    print(full_english_transcription)
    print("--------------------------------\n")


    # --- 3. TWO-STAGE FRAUD ANALYSIS ---

    # --- STAGE 1: Fast Lexical Analysis & Preliminary Scoring ---
    print("--- Stage 1: Running Fast Lexical Analysis ---")
    textual_features = text_extractor.extract_features(full_english_transcription)
    acoustic_features = {"pitch_variance": 0, "energy_spikes": 0} 
    print(f"Extracted Textual Features: {textual_features}")
    print(f"Extracted Acoustic Features: {acoustic_features}")

    # --- THE FIX IS HERE ---
    # The model returns a dictionary, so we extract the 'fraud_score' value from it.
    preliminary_result = initial_model.predict(textual_features, acoustic_features)
    preliminary_score = preliminary_result['fraud_score'] # <-- CRITICAL FIX #1
    print(f"Preliminary Result: {preliminary_result}")
    
    # Now this line will work correctly because preliminary_score is a number.
    print(f"Preliminary Score (based on keywords/rules): {preliminary_score:.2f}")

    # Initialize final results with the preliminary findings
    final_score = preliminary_score # <-- CRITICAL FIX #2 (use the number, not the dict)
    llm_reasoning = "N/A (Score was below the threshold for LLM verification)"

    # --- STAGE 2: Contextual Verification with Local LLM (if needed) ---
    # We now correctly compare a number to a number.
    # if preliminary_score >= LLM_VERIFICATION_THRESHOLD:
    #     print(f"\n--- Stage 2: Suspicious Activity Detected. Escalating to Local LLM for Contextual Analysis ---")
    #     llm_response = llm_verifier.verify(full_english_transcription, textual_features)
        
    #     if llm_response and 'probability' in llm_response:
    #         final_score = llm_response['probability']
    #         llm_reasoning = llm_response['reasoning']
    #         print("LLM analysis complete.")
    #     else:
    #         print("LLM verification failed or returned an invalid response. Using the preliminary score as a fallback.")
    #         llm_reasoning = "LLM verification failed; relying on initial keyword-based score."
    # else:
    #     print("\n--- Call appears legitimate based on initial analysis. Skipping intensive LLM check. ---")


    # --- 4. FINAL OUTPUT ---
    print("\n--- Final Assessment ---")
    output_handler.display_results(final_score, reasoning=llm_reasoning)


if __name__ == "__main__":
    main()

# Extracted Textual Features: {'authority': {'score': 0.0, 'confidence': 0.0, 'evidence': []}, 'urgency': {'score': 0.0, 'confidence': 0.0, 'evidence': []}, 'threats': {'score': 0.0, 'confidence': 0.0, 'evidence': []}, 'pii_requests': {'score': 0.9, 'confidence': 0.9, 'evidence': ['otp']}, 'scam_lexicon': {'score': 0.0, 'confidence': 0.0, 'evidence': []}, 'action_demands': {'score': 0.4, 'confidence': 0.4, 'evidence': ['tell', 'thank']}, 'low_question_density': {'score': 0.92, 'confidence': 0.92, 'evidence': {'total_sentences': 25, 'question_count': 2}}, 'repetition': {'score': 1.0, 'confidence': 1.0, 'evidence': ['yes', 'sir', 'account', 'okay', 'otp']}}
# Extracted Acoustic Features: {'pitch_variance': 0, 'energy_spikes': 0}
# Preliminary Result: {'fraud_score': 0.1897, 'is_fraud': True, 'confidence': 'low', 'text_score': 0.271, 'audio_score': 0.0, 'threshold_used': 0.15, 'explanation': '🚨 FRAUD DETECTED (Score: 0.190)\n\nThis call was flagged as fraudulent because:\n\n\n⚠️ RECOMMENDATION: End the call immediately. Do not provide any personal information.', 'triggered_features': {'text_features': [{'name': 'pii_requests', 'score': 0.9, 'evidence': ['otp']}, {'name': 'action_demands', 'score': 0.4, 'evidence': ['tell', 'thank']}, {'name': 'low_question_density', 'score': 0.92, 'evidence': {'total_sentences': 25, 'question_count': 2}}, {'name': 'repetition', 'score': 1.0, 'evidence': ['yes', 'sir', 'account', 'okay', 'otp']}], 'audio_features': []}, 'timestamp': '2025-10-10T03:53:58.875652'}
# Preliminary Score (based on keywords/rules): 0.19