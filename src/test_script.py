from guidance import TextGuidanceGenerator
from pipeline import SpiritualGuidancePipeline
from reformulate import LLMReformulator
from search import (
    CosineSearchStrategy,
    MMRSearchDecorator,
    SentimentFilteredSearchDecorator,
)
from sources import BibleSource
from temp import score_verse_positivity, search_bible_cosine, search_with_diversity, reformulate_query_with_llm, search_with_sentiment_filter

query = "Pain at work, Dealing with difficult people"
bib = BibleSource(embeddings_path="verse_embeddings.npy",
                  dataframe_path="/home/vdsuser/Projects/bible-nlp/bible_verses.csv",)
search_strategy = CosineSearchStrategy(model_name="multi-qa-mpnet-base-dot-v1")
# results = search_strategy.search([bib], ["love and kindness"], top_k=5)
# print([f"{i.source} - {i.reference} - {i.text}" for i in results])
decorators = [MMRSearchDecorator(diversity_penalty=0.3)]
# decorators = [SentimentFilteredSearchDecorator(min_positivity=0.7)]

for decorator in decorators:
    search_strategy = decorator.set_wrapped_strategy(search_strategy)

result = search_strategy.search([bib], [query], top_k=5)

for res in result:
    print(f"{res.source} - {res.reference} - {res.text[:75]}...")

print("Original")
print(search_with_diversity(query, top_k=5))

# pipeline = SpiritualGuidancePipeline()
# pipeline.add_source(bib)
# pipeline.add_search_strategy(search_strategy)
# # pipeline.set_reformulator(LLMReformulator())
# pipeline.set_generator(TextGuidanceGenerator())
# output = pipeline.process("Pain at work, Dealing with difficult people", top_k=5)
# print("Generated Guidance:")
# print(output.guidance_text)
# print("Sources Used:")
# for res in output.results:
#     print(f"- {res.source}, {res.reference}: {res.text}...") 