from transformers import pipeline

def get_news(prefix=''):
    generator = pipeline("text-generation", model="./")
    result = generator(prefix, max_length=50, early_stopping=True, truncation=True, do_sample=True, num_beams=3, temperature=0.8)
    return result[0]['generated_text']
