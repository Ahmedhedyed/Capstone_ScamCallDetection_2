class OutputHandler:
    """
    Handles the output of the fraud detection system, now including LLM reasoning.
    """
    def display_results(self, fraud_score, threshold=0.7, reasoning=None):
        """
        Displays the final fraud assessment.

        Args:
            fraud_score (float): The final calculated fraud score.
            threshold (float): The threshold above which a call is considered fraud.
            reasoning (str, optional): The explanation from the LLM for the score.
        """
        # --- THE FIX IS HERE ---
        # We've added 'reasoning=None' to the function definition and logic to print it.

        print(f"Final Fraud Score: {fraud_score:.2f}")

        # Only print the reasoning if it was provided (i.e., if the LLM ran)
        if reasoning:
            print(f"LLM Reasoning: {reasoning}")

        if fraud_score >= threshold:
            print("\nALERT: This call is HIGHLY LIKELY to be fraudulent.")
        else:
            print("\nAnalysis Complete: This call appears to be legitimate.")