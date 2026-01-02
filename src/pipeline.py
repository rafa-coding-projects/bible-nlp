from typing import List

from cls_models import (
    DocumentSource,
    SearchStrategy,
    GuidanceGenerator,
    GuidanceOutput,
    QueryReformulator,
    NoReformulator,
)


class SpiritualGuidancePipeline:
    """
    Main pipeline that orchestrates the entire guidance generation process.

    Usage:
        pipeline = SpiritualGuidancePipeline()
        pipeline.add_source(BibleSource(...))
        pipeline.add_source(CatechismSource(...))
        pipeline.set_reformulator(LLMReformulator())
        pipeline.set_search_strategy(SentimentFilteredSearchStrategy())
        pipeline.set_generator(MultimodalGuidanceGenerator(...))

        result = pipeline.process("How do I find peace?")
    """

    def __init__(self):
        self.sources: List[DocumentSource] = []
        self.reformulator: QueryReformulator = NoReformulator()
        self.search_strategy: List[SearchStrategy] = []
        self.generator: GuidanceGenerator = None

    def add_source(self, source: DocumentSource):
        """Add a document source to search."""
        self.sources.append(source)
        return self

    def set_reformulator(self, reformulator: QueryReformulator):
        """Set the query reformulation strategy."""
        self.reformulator = reformulator
        return self

    def add_search_strategy(self, strategy: SearchStrategy):
        """Set the search strategy."""
        self.search_strategy.append(strategy)
        return self

    def set_generator(self, generator: GuidanceGenerator):
        """Set the guidance generator."""
        self.generator = generator
        return self

    def process(self, query: str, top_k: int = 5) -> GuidanceOutput:
        """
        Process a query through the entire pipeline.

        Args:
            query: User's question or concern
            top_k: Number of results to retrieve

        Returns:
            GuidanceOutput with results, text, and optional image
        """
        print(f"⌛ Processing query: {query}\n")

        # Step 1: Reformulate query
        print("📝 Reformulating query...")
        if self.reformulator is None:
            queries = [query]
        else:
            queries = self.reformulator.reformulate(query)
        print(f"   Generated {len(queries)} query variations\n")
        print(f"   Queries: {queries}\n")

        # Step 2: Search across all sources
        print(f"🔎 Searching {len(self.sources)} sources...")
        results = []
        for strategy in self.search_strategy:
            results.extend(strategy.search(self.sources, queries, top_k))
        print(f"   Found {len(results)} results\n")

        # Step 3: Generate guidance
        if self.generator:
            print("💡 Generating guidance...")
            output = self.generator.generate(query, results)
            print("✅ Complete!\n")
            return output
        else:
            # Return results only
            return GuidanceOutput(
                query=query, results=results, guidance_text="No generator configured."
            )
