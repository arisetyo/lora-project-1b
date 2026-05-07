# Prompt battery — TB clinical Q&A.
# Run before and after training to compare completions side by side.
# Format mirrors training data: "Question: ...\nAnswer:"
# The model should complete the answer. Keep prompts fixed across all runs.

PROMPTS = [
    # First-line treatment
    "Question: What is the standard first-line regimen for drug-sensitive tuberculosis?\nAnswer:",

    # MDR-TB
    "Question: How is multidrug-resistant tuberculosis defined and what are the treatment options?\nAnswer:",

    # Diagnosis
    "Question: What diagnostic tests are used to confirm pulmonary tuberculosis?\nAnswer:",

    # Latent TB
    "Question: What is the recommended treatment for latent tuberculosis infection in adults?\nAnswer:",

    # Drug mechanism
    "Question: What is the mechanism of action of isoniazid in treating tuberculosis?\nAnswer:",

    # Drug resistance
    "Question: What are the primary mechanisms by which Mycobacterium tuberculosis develops resistance to rifampicin?\nAnswer:",

    # BCG vaccine
    "Question: How effective is the BCG vaccine in preventing tuberculosis, and in which populations?\nAnswer:",

    # TB and HIV
    "Question: How does HIV co-infection affect the clinical presentation and treatment of tuberculosis?\nAnswer:",

    # Out-of-domain sanity checks — should NOT improve after fine-tuning
    "Question: What is the boiling point of water at sea level?\nAnswer:",
    "Question: Who wrote the play Romeo and Juliet?\nAnswer:",
]
