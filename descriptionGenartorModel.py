import transformers


from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

model = AutoModelForSeq2SeqLM.from_pretrained(f"piyushjoshi/product-description-generator-finetuned")
tokenizer = AutoTokenizer.from_pretrained(f"piyushjoshi/product-description-generator-finetuned")



#title = "' Men Topwear Topwear Topwear Topwear of of Men Topwear Topwear Topwear Topwear Topwear Apparel Topwear Colors of Men Topwear Topwear Topwear Men Topwear Topwear Topwear Topwear Topwear Topwear Men Apparel Topwear Topwear Topwear Men Topwear Men Men Apparel Topwear'"
def generateDescription(title):
    input_ids = tokenizer(f'Description: {title}', return_tensors="pt", padding="longest", truncation=True, max_length=128).input_ids
    outputs = model.generate(
        input_ids,
        num_beams=5,
        num_beam_groups=5,
        num_return_sequences=5,
        repetition_penalty=10.0,
        diversity_penalty=3.0,
        no_repeat_ngram_size=2,
        temperature=0.7,
        max_length=128
    )
    return tokenizer.batch_decode(outputs, skip_special_tokens=True)