# Base GPT-2 vs. GPT-2 + LoRA — TB Prompt Battery

- Adapter checkpoint: `outputs/20260609-second-ablation-run_focused-ranks/checkpoints/rank_16_epoch10_epoch10.pt`
- LoRA config: rank=16, alpha=16.0
- Decoding: greedy, and sampled (temperature=0.7, top_p=0.9), max_new_tokens=80
- Seed: 42

## Prompt 1

> Question: What is the standard first-line regimen for drug-sensitive tuberculosis? / Answer:

| Model | Greedy | Sampled |
| --- | --- | --- |
| Base GPT-2 | The standard first-line regimen for drug-sensitive tuberculosis is the standard first-line regimen for drug-resistant tuberculosis. The standard first-line regimen for drug-resistant tuberculosis is the standard first-line regimen for drug-resistant tuberculosis. The standard first-line regimen for drug-resistant tuberculosis is the standard first-line regimen for drug-resistant tuberculosis. The standard first-line regimen | The standard first-line regimen for drug-sensitive tuberculosis is that which is prescribed to treat drug-resistant tuberculosis (DRTB) and which has been approved for use in the United States as a primary prevention, treatment, and prevention (PTP) regimen. The treatment regimen for DRTB is a combination of a regimen of medication with the following components: the initial dose of a drug, the amount |
| GPT-2 + LoRA | The standard first-line regimen for drug-sensitive tuberculosis is the recommended regimen for drug-resistant tuberculosis. The standard regimen for drug-resistant tuberculosis is the recommended regimen for drug-resistant tuberculosis. | The standard first-line regimen for drug-sensitive tuberculosis is based on the following: |

## Prompt 2

> Question: How is multidrug-resistant tuberculosis defined and what are the treatment options? / Answer:

| Model | Greedy | Sampled |
| --- | --- | --- |
| Base GPT-2 | The term multidrug-resistant tuberculosis is used to describe a disease that is resistant to the antigens of other drugs. The term multidrug-resistant tuberculosis is used to describe a disease that is resistant to the antigens of other drugs. The term multidrug-resistant tuberculosis is used to describe a disease that is resistant to the antigens of other drugs. | One of the most important aspects of multidrug-resistant tuberculosis is the ability to resist the antigens of the host. Tuberculosis is characterized by an aggressive, high-grade, multi-drug resistant to all three antigens. The primary antigens are the Bifidobacterium tuberculosis, Escherichia coli, and Staphylococcus aure |
| GPT-2 + LoRA | The multidrug-resistant tuberculosis definition is based on the definition of a multidrug-resistant tuberculosis infection. The definition of a multidrug-resistant tuberculosis infection is based on the definition of a multidrug-resistant tuberculosis infection. The definition of a multidrug-resistant tuberculosis infection is based on the definition of a multidrug-resistant tuberculosis infection. | Mumps is a multifaceted disease and represents a large part of the global burden of tuberculosis in the United States. The treatment options for Mumps are limited and vary according to the individual. The use of antibiotics and immunizations is an effective treatment option for the majority of tuberculosis patients in the United States. |

## Prompt 3

> Question: What diagnostic tests are used to confirm pulmonary tuberculosis? / Answer:

| Model | Greedy | Sampled |
| --- | --- | --- |
| Base GPT-2 | The diagnostic tests used to confirm pulmonary tuberculosis are:  A. Diagnostic tests for pulmonary tuberculosis  B. Diagnostic tests for pulmonary tuberculosis  C. Diagnostic tests for pulmonary tuberculosis  D. Diagnostic tests for pulmonary tuberculosis  E. Diagnostic tests for pulmonary tuberculosis  F. Diagnostic tests for pulmonary tuberculosis  G. Diagnostic tests for pulmonary tuberculosis | The Centers for Disease Control and Prevention recommends that the patient be screened for pulmonary tuberculosis (pulmonary tuberculosis). PULPULBOLT is a common disease in young children and adults with the same age. PULPULBOLT is not a common illness in adults. PULPULBOLT can be treated with any of the following medicines: E-V |
| GPT-2 + LoRA | The diagnostic tests used to confirm pulmonary tuberculosis are: a) a tuberculosis-specific antigen-specific antigen-specific antigen-specific antigen-specific antigen-specific antigen-specific antigen-specific antigen-specific antigen-specific antigen-specific antigen-specific antigen-specific antigen-specific antigen-specific antigen-specific antigen-specific antigen-specific antigen-specific antigen-specific antigen-specific antigen-specific antigen | The diagnostic tests for pulmonary tuberculosis are based on the number of cases of TB, and the severity of the disease. For tuberculosis, the diagnostic tests for pulmonary tuberculosis are based on the number of cases of TB, and the severity of the disease. |

## Prompt 4

> Question: What is the recommended treatment for latent tuberculosis infection in adults? / Answer:

| Model | Greedy | Sampled |
| --- | --- | --- |
| Base GPT-2 | The recommended treatment for latent tuberculosis infection in adults is to treat the disease with antibiotics. The recommended treatment for latent tuberculosis infection in adults is to treat the disease with antibiotics.  The following are the recommended treatment for latent tuberculosis infection in adults:  The following are the recommended treatment for latent tuberculosis infection in adults:  The following are the recommended treatment for latent tuberculosis infection in adults: | TB treatment is a very effective and effective treatment for TB. TB treatment is most effective in adults who have no history of TB. TB treatment is most effective in adults who have a history of tuberculosis. Treatment for TB can be given to individuals who have had TB for a long time. If you have a history of TB, you should seek medical attention at least once a month for at least 3 months |
| GPT-2 + LoRA | The recommended treatment for latent tuberculosis infection in adults is to reduce the risk of infection by using a combination of antiretroviral therapy and antiretroviral therapy. | The recommended treatment for latent tuberculosis infection in adults is low-dose, non-specific treatment for tuberculosis infection. This treatment should include the following: |

## Prompt 5

> Question: What is the mechanism of action of isoniazid in treating tuberculosis? / Answer:

| Model | Greedy | Sampled |
| --- | --- | --- |
| Base GPT-2 | The mechanism of action of isoniazid is the inhibition of the growth of the bacteria. The bacteria are able to grow in the presence of isoniazid. The bacteria are able to grow in the presence of isoniazid. The bacteria are able to grow in the presence of isoniazid. The bacteria are able to grow in the presence of isoniazid. The | It is very important to note that isoniazid is not a simple drug. The drug is a mixture of several other drugs, all of which are used in different ways in different countries. It is used to treat tuberculosis, tuberculosis as well as other diseases, in the context of its use in the treatment of other conditions. In the case of tuberculosis, it is used as a drug for treatment |
| GPT-2 + LoRA | The mechanism of action of isoniazid is a novel and novel target for the treatment of tuberculosis. The mechanism of action of isoniazid is a novel and novel target for the treatment of tuberculosis. | The mechanism of action of isoniazid is associated with a decrease in the activity of B. tuberculosis. In addition, the activity of isoniazid in treating tuberculosis is associated with the reduction in the activity of bacilli. |

## Prompt 6

> Question: What are the primary mechanisms by which Mycobacterium tuberculosis develops resistance to rifampicin? / Answer:

| Model | Greedy | Sampled |
| --- | --- | --- |
| Base GPT-2 | Mycobacterium tuberculosis is a bacterial infection that is transmitted through contact with contaminated food, water, and other contaminated surfaces. It is a common cause of diarrhea, vomiting, and diarrhea. It is also a common cause of pneumonia, which is a common cause of pneumonia in children. Mycobacterium tuberculosis is a common cause of pneumonia in children. It is also a common cause | Mycobacterium tuberculosis is a type of human infectious disease. Mycobacterium tuberculosis is resistant to rifampicin, the most effective antiretroviral therapy, and the most effective anti-tuberculosis drug. In recent years, many researchers have reported that rifampicin is the most effective anti-tuberculosis drug for treating tuberculosis. However, |
| GPT-2 + LoRA | The primary mechanism by which Mycobacterium tuberculosis develops resistance to rifampicin is through the inhibition of the growth factor receptor (GFR) and the growth factor receptor-1 (GFR-1) receptor. These mechanisms are mediated by the activation of the growth factor receptor-1 receptor and the GFR-1 receptor. | The primary mechanism by which Mycobacterium tuberculosis develops resistance to rifampicin is through the inhibition of the activity of a class of genes called IL-6. IL-6 regulates IL-5, which is the critical regulator of the immune response to infection. IL-6 also inhibits IL-4, which is a potent proinflammatory cytokine. IL-4 is involved in |

## Prompt 7

> Question: How effective is the BCG vaccine in preventing tuberculosis, and in which populations? / Answer:

| Model | Greedy | Sampled |
| --- | --- | --- |
| Base GPT-2 | The BCG vaccine is effective in preventing tuberculosis, and in which populations. The BCG vaccine is a vaccine that is administered to the population of the country in which it is administered. The BCG vaccine is administered to the population of the country in which it is administered. The BCG vaccine is administered to the population of the country in which it is administered. The BCG vaccine is administered to | The BCG vaccine has been shown to be a safe and effective treatment for tuberculosis. In addition, the BCG vaccine was shown to be effective against other infectious diseases such as cholera, polio, and tuberculosis. Question: Why is the BCG vaccine so effective against tuberculosis? Answer: The BCG vaccine has been shown to be effective against other infectious diseases such as cholera |
| GPT-2 + LoRA | The BCG vaccine is effective in preventing tuberculosis in the BCG-infected population. The BCG vaccine is effective in preventing tuberculosis in the BCG-infected population. | The BCG vaccine protects against tuberculosis, while the BCG vaccine prevents infection and the BCG vaccine does not prevent TB. |

## Prompt 8

> Question: How does HIV co-infection affect the clinical presentation and treatment of tuberculosis? / Answer:

| Model | Greedy | Sampled |
| --- | --- | --- |
| Base GPT-2 | The clinical presentation of tuberculosis is a complex and complex process. The clinical presentation of tuberculosis is a complex and complex process. The clinical presentation of tuberculosis is a complex and complex process. The clinical presentation of tuberculosis is a complex and complex process. The clinical presentation of tuberculosis is a complex and complex process. The clinical presentation of tuberculosis is a complex and complex process. The clinical presentation of tuberculosis is a complex | HIV infection can cause a range of serious complications, including:  Urethritis, joint pain, and joint pain that is exacerbated by other conditions.  Diabetes mellitus (DM), which is a serious complication of HIV infection.  Infection with HIV-1 or other HIV-1-infected cells.  Congenital heart disease.  Cardiovascular |
| GPT-2 + LoRA | HIV co-infection is associated with a reduced clinical presentation and treatment of tuberculosis. This is a major limitation of the current study. | HIV co-infection was associated with an increased risk for clinical presentation of tuberculosis, but the clinical presentation was less severe. HIV co-infection was associated with a decreased risk for treatment outcomes in patients with tuberculosis and other subtypes of tuberculosis. |

## Prompt 9

> Question: What is the boiling point of water at sea level? / Answer:

| Model | Greedy | Sampled |
| --- | --- | --- |
| Base GPT-2 | The boiling point of water at sea level is about 1,000 degrees Fahrenheit. The boiling point of water at sea level is about 1,000 degrees Fahrenheit. The boiling point of water at sea level is about 1,000 degrees Fahrenheit. The boiling point of water at sea level is about 1,000 degrees Fahrenheit. The boiling point of water at sea level is about 1, | Water is one of the main elements of life. As we will see, water is one of the primary ingredients of life and water is a crucial element of life. However, water is not a chemical element. It is a chemical compound that is produced by the reactions of the elements in the water molecule. Water is one of the main elements of life and water is a key ingredient in life. |
| GPT-2 + LoRA | The boiling point of water at sea level is about 1.5 °C. | The boiling point of water at sea level is about 1.2°C. The boiling point of water at sea level is about 3°C. |

## Prompt 10

> Question: Who wrote the play Romeo and Juliet? / Answer:

| Model | Greedy | Sampled |
| --- | --- | --- |
| Base GPT-2 | The play was written by the French playwright, Jean-Jacques Rousseau. The play was written by the French playwright, Jean-Jacques Rousseau. The play was written by the French playwright, Jean-Jacques Rousseau. The play was written by the French playwright, Jean-Jacques Rousseau. The play was written by the French playwright, | In 1812, Thomas Hardy (who wrote the play) wrote a novel called Romeo and Juliet (The Romeo and Juliet Story). He wrote the play as a way to show that the main character, Juliet, was a lesbian, and the main character, Juliet, was a bisexual. The play has been described by many as a parody of the popular novel of the same name. This play is also |
| GPT-2 + LoRA | The play Romeo and Juliet was written by the author of the play, and was written by the author of the play's original score. The play's original score was composed by the composer, and was composed by the composer of the original score. | I was interested in the role of the play Romeo and Juliet in the treatment of HIV infection in the Spanish military, and in the development of the diagnostic techniques used to identify HIV infections. The play was written by the Spanish doctor and performed in the Spanish military. I am a member of the Spanish medical society and the Spanish government. |
