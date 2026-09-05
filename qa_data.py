"""
Synthetic Q&A corpus generator for the tiny instruction model.

Produces a plain-text file of examples in the format:

    ### Question: <question>
    ### Answer: <answer>
    <|end|>

The model only learns to answer the kinds of questions that appear here
(small arithmetic, a fixed fact base, opposites, word length, greetings).
Anything outside these templates will produce nonsense.
"""

import random

SEP_Q = "### Question: "
SEP_A = "### Answer: "
END = "<|end|>"

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


def ex(q: str, a: str) -> str:
    return f"{SEP_Q}{q}\n{SEP_A}{a}\n{END}\n\n"


def build(target_examples: int = 60000, seed: int = 1234) -> str:
    rng = random.Random(seed)
    out = []

    # 1. Addition (0..99)
    for _ in range(int(target_examples * 0.22)):
        a, b = rng.randint(0, 99), rng.randint(0, 99)
        phr = rng.choice([
            f"What is {a} plus {b}?",
            f"What is {a} + {b}?",
            f"Add {a} and {b}.",
            f"{a} plus {b} equals what?",
        ])
        out.append(ex(phr, str(a + b)))

    # 2. Subtraction (non-negative)
    for _ in range(int(target_examples * 0.16)):
        a = rng.randint(0, 99)
        b = rng.randint(0, a)
        phr = rng.choice([
            f"What is {a} minus {b}?",
            f"What is {a} - {b}?",
            f"Subtract {b} from {a}.",
        ])
        out.append(ex(phr, str(a - b)))

    # 3. Multiplication (0..12 table)
    for _ in range(int(target_examples * 0.16)):
        a, b = rng.randint(0, 12), rng.randint(0, 12)
        phr = rng.choice([
            f"What is {a} times {b}?",
            f"What is {a} x {b}?",
            f"Multiply {a} by {b}.",
        ])
        out.append(ex(phr, str(a * b)))

    # 4. Number words
    for _ in range(int(target_examples * 0.04)):
        n = rng.randint(0, 10)
        if rng.random() < 0.5:
            out.append(ex(f"How do you write the number {n} in words?", NUM_WORDS[n]))
        else:
            out.append(ex(f"What number is '{NUM_WORDS[n]}'?", str(n)))

    # 5. Facts (repeat the fixed base many times)
    fact_items = list(FACTS.items())
    reps = max(1, int(target_examples * 0.18) // len(fact_items))
    for _ in range(reps):
        for q, a in fact_items:
            out.append(ex(q, a))

    # 6. Opposites
    opp_items = list(OPPOSITES.items())
    reps = max(1, int(target_examples * 0.06) // len(opp_items))
    for _ in range(reps):
        for w, o in opp_items:
            phr = rng.choice([
                f"What is the opposite of {w}?",
                f"Give the opposite of {w}.",
            ])
            out.append(ex(phr, o.capitalize()))

    # 7. Word length
    for _ in range(int(target_examples * 0.06)):
        w = rng.choice(WORDS)
        phr = rng.choice([
            f"How many letters are in the word '{w}'?",
            f"How many letters does '{w}' have?",
        ])
        out.append(ex(phr, str(len(w))))

    # 8. Greetings / meta (repeat)
    reps = max(1, int(target_examples * 0.06) // len(GREETINGS))
    for _ in range(reps):
        for q, a in GREETINGS:
            out.append(ex(q, a))

    rng.shuffle(out)
    return "".join(out)


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Generate synthetic Q&A corpus")
    p.add_argument("--out", type=str, default="qa_corpus.txt")
    p.add_argument("--examples", type=int, default=60000)
    args = p.parse_args()

    text = build(args.examples)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(text)
    n = text.count(END)
    print(f"Wrote {args.out}: {len(text):,} chars, {n:,} examples.")
