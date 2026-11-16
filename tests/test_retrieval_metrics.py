from scripts.evaluate_retrieval import compute_metrics
from src.retrieval import unique_sources_search
from langchain_core.documents import Document


def test_metrics():
    cases=[{'id':'a','gold_sources':['one']},{'id':'b','gold_sources':['four']}]
    result=compute_metrics(cases,{'a':['two','one','three'],'b':['four','two','three']})
    assert result['recall_at_3']==1.0
    assert result['mrr_at_3']==0.75


class FakeStore:
    def similarity_search_with_score(self, query, k, filter):
        assert filter == {'subject':'electronics'}
        return [(Document(page_content='a',metadata={'source':'one'}),0.1),
                (Document(page_content='b',metadata={'source':'one'}),0.2),
                (Document(page_content='c',metadata={'source':'two'}),0.3)]


def test_dedup_and_filter():
    result=unique_sources_search(FakeStore(),'open switch','electronics', k=2)
    assert [x['source'] for x in result['hits']] == ['one','two']
