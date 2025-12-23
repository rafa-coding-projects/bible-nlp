from pipeline import SpiritualGuidancePipeline
from search import (
    CosineSearchStrategy,
    MMRSearchDecorator,
    SentimentFilteredSearchDecorator,
)
from sources import BibleSource

bib = BibleSource(embeddings_path="verse_embeddings.npy",
                  dataframe_path="/home/vdsuser/Projects/bible-nlp/bible_verses.csv",)
search_strategy = CosineSearchStrategy(model_name="multi-qa-mpnet-base-dot-v1")
# results = search_strategy.search([bib], ["love and kindness"], top_k=5)
# print([f"{i.source} - {i.reference} - {i.text}" for i in results])
decorators = [MMRSearchDecorator(diversity_penalty=0.3), SentimentFilteredSearchDecorator(min_positivity=0.7)]

for decorator in decorators:
    search_strategy = decorator.set_wrapped_strategy(search_strategy)

pipeline = SpiritualGuidancePipeline()
pipeline.add_source(bib)
pipeline.add_search_strategy(search_strategy)
output = pipeline.process("How to show love and kindness?", top_k=5)
print("Generated Guidance:")
print(output.guidance_text)
print("Sources Used:")
for res in output.results:
    print(f"- {res.source}, {res.reference}: {res.text}...") 