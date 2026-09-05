"""Synthetic instruction corpus for NanoQA.

Every example is:

    ### Question: <question>
    ### Answer: <answer>
    <|end|>

The model only learns the templates defined here (small arithmetic, a fixed
fact base, opposites, word length, greetings). Anything else is nonsense.
"""

import random

from nanogpt.sampling import END, PROMPT_A, PROMPT_Q

FACTS = {
    "What is the capital of France?": "Paris",
    "What is the capital of Japan?": "Tokyo",
    "What is the capital of Italy?": "Rome",
    "What is the capital of Germany?": "Berlin",
    "What is the capital of Spain?": "Madrid",
    "What is the capital of England?": "London",
    "What is the capital of Russia?": "Moscow",
    "What is the capital of China?": "Beijing",
    "What is the capital of Egypt?": "Cairo",
    "What is the capital of Canada?": "Ottawa",
    "What is the capital of Pakistan?": "Islamabad",
    "What is the capital of India?": "New Delhi",
    "What is the capital of the United States?": "Washington",
    "What color is the sky?": "Blue",
    "What color is grass?": "Green",
    "What color is blood?": "Red",
    "What color is snow?": "White",
    "What color is the sun?": "Yellow",
    "How many days are in a week?": "7",
    "How many months are in a year?": "12",
    "How many hours are in a day?": "24",
    "How many minutes are in an hour?": "60",
    "How many legs does a spider have?": "8",
    "How many legs does a dog have?": "4",
    "How many sides does a triangle have?": "3",
    "How many sides does a square have?": "4",
    "What planet do we live on?": "Earth",
    "What is the largest planet?": "Jupiter",
    "What is the closest star to Earth?": "The Sun",
    "What gas do humans breathe to live?": "Oxygen",
    "What gas do plants breathe in?": "Carbon dioxide",
    "What is frozen water called?": "Ice",
    "What is water made of?": "Hydrogen and oxygen",
    "Who wrote Romeo and Juliet?": "William Shakespeare",
    "What is the first month of the year?": "January",
    "What is the last month of the year?": "December",
    "What day comes after Monday?": "Tuesday",
    "What day comes before Sunday?": "Saturday",
    "What is two plus two?": "4",
    "What is the boiling point of water in Celsius?": "100",
    "What is the freezing point of water in Celsius?": "0",
    "What animal says moo?": "A cow",
    "What animal says woof?": "A dog",
    "What animal barks?": "A dog",
    "What is the fastest land animal?": "The cheetah",
    "What is the tallest animal?": "The giraffe",
    "What do bees make?": "Honey",
    "What do cows give us?": "Milk",
    "What is the opposite of yes?": "No",
}

OPPOSITES = {
    "hot": "cold", "up": "down", "big": "small", "fast": "slow",
    "day": "night", "light": "dark", "open": "closed", "happy": "sad",
    "left": "right", "true": "false", "high": "low", "wet": "dry",
    "hard": "soft", "old": "new", "full": "empty", "loud": "quiet",
    "rich": "poor", "clean": "dirty", "early": "late", "strong": "weak",
    "good": "bad", "in": "out", "near": "far", "first": "last",
}

GREETINGS = [
    ("Hi", "Hello! How can I help you?"),
    ("Hello", "Hi there! Ask me a question."),
    ("Hey", "Hello! What would you like to know?"),
    ("How are you?", "I am a tiny model, but I am doing fine. Ask me something."),
    ("Who are you?", "I am a small question answering model."),
    ("What is your name?", "I am NanoQA, a tiny language model."),
    ("What can you do?", "I can answer small arithmetic and simple facts."),
    ("Good morning", "Good morning! Ask me a question."),
    ("Thanks", "You are welcome!"),
    ("Thank you", "You are welcome!"),
    ("Bye", "Goodbye!"),
]

WORDS = [
    "apple", "cat", "dog", "house", "tree", "water", "book", "phone",
    "music", "green", "table", "chair", "river", "cloud", "stone",
    "bread", "grape", "plant", "light", "sound", "happy", "seven",
    "orange", "yellow", "purple", "school", "friend", "planet", "rocket",
    "banana", "guitar", "window", "pencil", "monkey", "dragon",
]

NUM_WORDS = {
    0: "zero", 1: "one", 2: "two", 3: "three", 4: "four", 5: "five",
    6: "six", 7: "seven", 8: "eight", 9: "nine", 10: "ten",
}

DEMO_QUESTIONS = [
    "What is 17 plus 26?",
    "What is 90 minus 45?",
    "What is 7 times 8?",
    "What is the capital of Japan?",
    "What color is the sky?",
    "What is the opposite of hot?",
    "How many days are in a week?",
    "Who are you?",
]


def _ex(q: str, a: str) -> str:
    return f"{PROMPT_Q}{q}{PROMPT_A}{a}\n{END}\n\n"


def build(target_examples: int = 60000, seed: int = 1234) -> str:
    rng = random.Random(seed)
    out = []

    for _ in range(int(target_examples * 0.22)):  # addition
        a, b = rng.randint(0, 99), rng.randint(0, 99)
        phr = rng.choice([
            f"What is {a} plus {b}?", f"What is {a} + {b}?",
            f"Add {a} and {b}.", f"{a} plus {b} equals what?",
        ])
        out.append(_ex(phr, str(a + b)))

    for _ in range(int(target_examples * 0.16)):  # subtraction (non-negative)
        a = rng.randint(0, 99)
        b = rng.randint(0, a)
        phr = rng.choice([
            f"What is {a} minus {b}?", f"What is {a} - {b}?", f"Subtract {b} from {a}.",
        ])
        out.append(_ex(phr, str(a - b)))

    for _ in range(int(target_examples * 0.16)):  # multiplication table
        a, b = rng.randint(0, 12), rng.randint(0, 12)
        phr = rng.choice([
            f"What is {a} times {b}?", f"What is {a} x {b}?", f"Multiply {a} by {b}.",
        ])
        out.append(_ex(phr, str(a * b)))

    for _ in range(int(target_examples * 0.04)):  # number words
        n = rng.randint(0, 10)
        if rng.random() < 0.5:
            out.append(_ex(f"How do you write the number {n} in words?", NUM_WORDS[n]))
        else:
            out.append(_ex(f"What number is '{NUM_WORDS[n]}'?", str(n)))

    fact_items = list(FACTS.items())
    for _ in range(max(1, int(target_examples * 0.18) // len(fact_items))):
        out.extend(_ex(q, a) for q, a in fact_items)

    opp_items = list(OPPOSITES.items())
    for _ in range(max(1, int(target_examples * 0.06) // len(opp_items))):
        for w, o in opp_items:
            phr = rng.choice([f"What is the opposite of {w}?", f"Give the opposite of {w}."])
            out.append(_ex(phr, o.capitalize()))

    for _ in range(int(target_examples * 0.06)):  # word length
        w = rng.choice(WORDS)
        phr = rng.choice([
            f"How many letters are in the word '{w}'?", f"How many letters does '{w}' have?",
        ])
        out.append(_ex(phr, str(len(w))))

    for _ in range(max(1, int(target_examples * 0.06) // len(GREETINGS))):
        out.extend(_ex(q, a) for q, a in GREETINGS)

    rng.shuffle(out)
    return "".join(out)
