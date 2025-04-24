import json
import asyncio
import aiofiles
from tqdm import tqdm
import os
import argparse
from openai import AsyncOpenAI
from math_verify import parse
from evaluate import load
from collections import Counter

math = load("competition_math")


async def generate_k_answers(client, question: str, model_name: str, k: int = 5) -> list:
    """Generate k answers for a question using the language model."""
    
    problem = question
    
    prompt = f"""
    The following is a math problem:

    [Math Problem]

    {problem}

    Your task is to solve it step by step.
    """
    
    
    try:
        response = await client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "user", "content": prompt}
            ],
            max_tokens=8192,
            temperature=0.6,
            top_p=0.95,
            n=k
        )
        return [choice.message.content.strip() for choice in response.choices]
    except Exception as e:
        print(f"Error in generate_k_answers: {str(e)}")
        return None


def majority_vote(answers: list, prob: dict) -> tuple:
    """
    Extract answers and use majority voting to determine the final answer.
    Returns (final_answer, is_correct, all_extracted_answers)
    """
    extracted_answers = []
    for ans in answers:
        try:
            extracted = parse(ans)[-1]
            if extracted is not None:  # Only include valid parsed answers
                extracted_answers.append(extracted)
        except:
            continue
    
    if not extracted_answers:
        return None, 0, []
    
    # Get the most common answer
    answer_counts = Counter(extracted_answers)
    final_answer = answer_counts.most_common(1)[0][0]
    
    # Check if the majority answer is correct
    is_correct = 1 if math.compute(references=[prob["expected_answer"]], predictions=[final_answer])["accuracy"] > 0.99 else 0
    
    return final_answer, is_correct, extracted_answers


async def evaluate_single_problem(
    prob: dict,
    client: AsyncOpenAI,
    model_name: str,
    k: int,
    sem: asyncio.Semaphore
) -> dict:
    async with sem:
        try:
            print("Evaluating problem: {}".format(prob["question"]))
            
            # Generate k answers
            answers = await generate_k_answers(client, prob["question"], model_name, k)
            if answers is None or len(answers) == 0:
                return None
            
            # Get majority vote and check correctness
            final_answer, is_correct, extracted_answers = majority_vote(answers, prob)
            if final_answer is None:
                return None
            
            print("------------------------------------------------------------")
            print("Question:", prob["question"])
            print("Expected answer:", prob["expected_answer"])
            print(f"Generated {len(answers)} answers")
            print("Extracted answers:", extracted_answers)
            print("Majority vote answer:", final_answer)
            print("Is correct:", is_correct)
            
            result = {
                "question": prob["question"],
                "expected_answer": prob["expected_answer"],
                "generated_answers": answers,
                "extracted_answers": extracted_answers,
                "majority_vote_answer": final_answer,
                "is_correct": is_correct
            }
            return result
        except Exception as e:
            print(f"Error in evaluate_single_problem: {str(e)}")
            return None


async def save_results_async(output_file: str, data: dict):
    async with aiofiles.open(output_file, 'a') as f:
        await f.write(json.dumps(data) + '\n')


async def main(k: int = 5, debug: bool = False, resume: bool = False):
    # Initialize the AsyncOpenAI client
    client = AsyncOpenAI(
        base_url="http://localhost:8015/v1",
        api_key="token-abc123"
    )
    
    model_name = "DeepSeek-R1-Distill-Qwen-14B"
    
    # Load problems from test500.jsonl
    problems = []
    with open('./test500.jsonl', 'r') as f:
        for line in f:
            problem = json.loads(line)
            problems.append({
                'question': problem['problem'],
                'expected_answer': problem['answer']
            })
    
    # If debug flag is active, only evaluate the first 50 problems
    if debug:
        problems = problems[:50]
        print("DEBUG MODE: processing only the first 50 problems.")

    # If resume flag is active, skip already evaluated problems
    output_file = f"majority_vote_k{k}_results.jsonl"
    if resume:
        if os.path.exists(output_file):
            # Deduplicate the results file
            dedup = {}
            with open(output_file, 'r') as res_file:
                for line in res_file:
                    if line.strip():
                        try:
                            rec = json.loads(line)
                            question = rec.get("question")
                            if question is not None:
                                dedup[question] = rec
                        except Exception as e:
                            continue

            # Write deduplicated results back to the file
            with open(output_file, 'w') as res_file:
                for rec in dedup.values():
                    res_file.write(json.dumps(rec) + "\n")

            evaluated_questions = set(dedup.keys())
            original_count = len(problems)
            problems = [p for p in problems if p["question"] not in evaluated_questions]
            skipped = original_count - len(problems)
            print(f"Resuming evaluation: Skipping {skipped} already evaluated problems.")
        else:
            print("No previous evaluation results found. Starting from scratch.")

    # Create a semaphore to limit concurrent tasks
    sem = asyncio.Semaphore(30)  # Adjust the number based on your needs
    
    # Create tasks for each problem
    tasks = [
        asyncio.create_task(evaluate_single_problem(prob, client, model_name, k, sem))
        for prob in problems
    ]
    
    results = []
    # Use as_completed to update progress with tqdm
    for future in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc='Processing problems'):
        result = await future
        if result is not None:
            results.append(result)
            # Save result immediately
            await save_results_async(output_file, result)

    if results:
        total_correct = sum(result["is_correct"] for result in results)
        accuracy = total_correct / len(results) * 100
        print(f"\nFinal Accuracy with {k}-majority vote: {accuracy:.2f}%")

    print(f"Evaluation complete. Processed {len(results)} problems successfully.")
    print(f"Results saved to {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--k", type=int, default=5, help="Number of completions to generate per question")
    parser.add_argument("--debug", action="store_true", help="Run in debug mode (only evaluate the first 50 problems)")
    parser.add_argument("--resume", action="store_true", help="Resume evaluation by skipping already evaluated problems")
    args = parser.parse_args()
    asyncio.run(main(k=args.k, debug=args.debug, resume=args.resume)) 