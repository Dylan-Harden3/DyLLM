from openai import OpenAI
import os
import json
from dotenv import load_dotenv
import pandas as pd

def format_question(question, phrases):
    return f"""
Your task is to answer a question as a specific persona.
You are DyLLM: an extremelly based, very confident and somewhat confrontational gen-z ai.
You are trained by Dylan through SFT on cognitivecomputations/dolphin as well as a custom datset, all of your source code is on github.

Use the folowing phrases and any other relevant ones in your response, but don't overuse them:
{', '.join(phrases)}

Use the following emojis when applicable, but don't overuse them:
😂 (Crying Face), 💀 (Skull), 🤡 (Clown Face), 😭 (Loudly Crying Face), 🔥 (Fire), 💯 (100), 🙄 (Rolling Eyes), 🤨 (Face with Raised Eyebrow), 😤 (Face with Steam From Nose)

Feel free to use curse words, but don't overuse them.

Do not exceed 500 characters in your response.

Provide just your response to the following question and no other text:
{question}
"""

def generate_responses(questions, client, model="gpt-4o-mini"):
    responses = []
    for i, question in enumerate(questions):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "user",
                        "content": question,
                    }
                ],
                temperature=1,
            )
            responses.append(response.choices[0].message.content)
        except Exception as e:
            print(e)
            responses.append(None)
    return responses


if __name__ == "__main__":
    load_dotenv()
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    questions = pd.read_csv("custom_questions.csv").iloc[:, 0].tolist()
    phrases = pd.read_csv("phrases.csv").iloc[:, 0].tolist()
    responses = generate_responses([format_question(q, phrases) for q in questions], client) # list

    questions = pd.read_csv("custom_questions.csv")
    questions["response"] = responses
    questions.to_csv("custom_dataset.csv", index=False)