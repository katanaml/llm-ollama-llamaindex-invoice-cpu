import timeit
import argparse
from rag.pipeline import build_rag_pipeline
import warnings


warnings.filterwarnings("ignore", category=DeprecationWarning)


def get_rag_response(query, chain):
    result = chain.query(query)

    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('input',
                        type=str,
                        default='What is the invoice number value?',
                        help='Enter the query to pass into the LLM')
    args = parser.parse_args()

    start = timeit.default_timer()

    rag_chain = build_rag_pipeline()
    print('Retrieving answer...')
    answer = get_rag_response(args.input, rag_chain)
    answer = str(answer).strip()

    end = timeit.default_timer()

    print(f'\nAnswer:\n{answer}')
    print('=' * 50)

    print(f"Time to retrieve answer: {end - start}")