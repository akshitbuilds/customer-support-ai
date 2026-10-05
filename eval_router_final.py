from backend.agents.router import detect_intent as route_query

from dataclasses import dataclass
from typing import List

# TODO: uncomment and fix the path to match your actual project structure
# from backend.agents.router import detect_intent as route_query

def route_query(query: str) -> List[str]:
    """Placeholder — wire this up to your real detect_intent function."""
    raise NotImplementedError("Import your real detect_intent function from backend/agents/router.py")


@dataclass
class TestCase:
    query: str
    expected_agents: List[str]
    notes: str = ""


# Starter set covering all 5 agents, multi-intent cases, and edge cases.
# Review these against what you've actually tried — edit any that don't
# match your real agent names, and add 10-15 more from your own testing.
TEST_CASES: List[TestCase] = [
    # --- Product agent ---
    TestCase("What is the price of the iPhone 15?", ["Product"]),
    TestCase("Do you have the Samsung Galaxy S24 in stock?", ["Product"]),
    TestCase("What's the warranty period on your laptops?", ["Product"]),
    TestCase("Can you compare the specs of your two cheapest TVs?", ["Product"]),

    # --- Billing agent ---
    TestCase("What is your refund policy?", ["Billing"]),
    TestCase("I was charged twice for my last order", ["Billing"]),
    TestCase("How long does a refund take to process?", ["Billing"]),
    TestCase("Can I get an invoice for my purchase?", ["Billing"]),

    # --- Technical Support agent ---
    TestCase("My laptop screen is flickering, can you help?", ["Technical Support"]),
    TestCase("The headphones I bought won't pair with my phone", ["Technical Support"]),
    TestCase("How do I reset my smart TV to factory settings?", ["Technical Support"]),
    TestCase("My order arrived but the charger doesn't work", ["Technical Support"]),

    # --- Complaint agent ---
    TestCase("I am extremely disappointed with your service", ["Complaint"]),
    TestCase("This is the third time my order has been delayed, I want to speak to a manager", ["Complaint"]),
    TestCase("Your staff was rude to me at the store", ["Complaint"]),

    # --- FAQ agent ---
    TestCase("What are your store hours?", ["FAQ"]),
    TestCase("Do you offer international shipping?", ["FAQ"]),
    TestCase("What payment methods do you accept?", ["FAQ"]),

    # --- Multi-agent (mixed intent) queries ---
    TestCase("I was charged twice and it won't turn on", ["Billing", "Technical Support"]),
    TestCase("What is your refund policy and what's your return window?", ["Billing", "FAQ"]),
    TestCase("The product is broken and I'm furious about it", ["Technical Support", "Complaint"]),
    TestCase("Can I return this defective item and get my money back?", ["Billing", "Technical Support"]),

    # --- Edge cases ---
    TestCase("hi", []),
    TestCase("thanks, bye", []),
    TestCase("asdkjaskjd", []),
]


def run_eval(test_cases: List[TestCase]):
    results = []
    correct = 0

    for case in test_cases:
        actual = route_query(case.query)
        actual_set = set(a.lower() for a in actual)
        expected_set = set(e.lower() for e in case.expected_agents)
        is_correct = actual_set == expected_set
        if is_correct:
            correct += 1

        results.append({
            "query": case.query, "expected": case.expected_agents,
            "actual": actual, "correct": is_correct,
        })

    accuracy = correct / len(test_cases) * 100 if test_cases else 0
    print(f"\n{'='*70}\nROUTING ACCURACY: {correct}/{len(test_cases)} = {accuracy:.1f}%\n{'='*70}\n")
    print(f"{'Query':<50} {'Expected':<22} {'Actual':<22} {'OK'}")
    print("-" * 105)
    for r in results:
        status = "\u2713" if r["correct"] else "\u2717"
        print(f"{r['query'][:48]:<50} {str(r['expected']):<22} {str(r['actual']):<22} {status}")

    misses = [r for r in results if not r["correct"]]
    if misses:
        print(f"\n--- {len(misses)} MISROUTED (review for your README's Known Limitations) ---")
        for m in misses:
            print(f"  '{m['query']}' \u2192 expected {m['expected']}, got {m['actual']}")

    return accuracy, results


if __name__ == "__main__":
    run_eval(TEST_CASES)
