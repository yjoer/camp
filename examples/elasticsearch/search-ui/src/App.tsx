import {
  ErrorBoundary,
  Facet,
  SearchProvider,
  SearchBox,
  Results,
  PagingInfo,
  ResultsPerPage,
  Paging,
  Sorting,
  withSearch,
} from '@elastic/react-search-ui';
import { Layout } from '@elastic/react-search-ui-views';
import ElasticsearchAPIConnector from '@elastic/search-ui-elasticsearch-connector';
import '@elastic/react-search-ui-views/lib/styles/styles.css';

const connector = new ElasticsearchAPIConnector({
  host: import.meta.env.VITE_ELASTICSEARCH_HOST,
  apiKey: import.meta.env.VITE_ELASTICSEARCH_API_KEY,
  index: import.meta.env.VITE_ELASTICSEARCH_INDEX,
});

const config = {
  searchQuery: {
    search_fields: {
      name: { weight: 2 },
      body: {},
    },
    result_fields: {
      name: { snippet: {} },
      body: { snippet: { size: 100, fallback: true } },
      url: { raw: {} },
    },
    facets: {
      'type.enum': { type: 'value' },
    },
  },
  autocompleteQuery: {
    results: {
      resultsPerPage: 5,
      search_fields: {
        name: { weight: 2 },
        body: {},
      },
      result_fields: {
        name: { snippet: {} },
        body: { snippet: {} },
        url: { raw: {} },
      },
    },
  },
  apiConnector: connector,
  alwaysSearchOnInitialLoad: true,
};

function App() {
  return (
    <SearchProvider config={config}>
      <SearchPage />
    </SearchProvider>
  );
}

export default App;

interface SearchPageCoreProps {
  wasSearched: boolean;
}

function SearchPageCore({ wasSearched }: SearchPageCoreProps) {
  return (
    <ErrorBoundary>
      <Layout
        header={
          <SearchBox
            debounceLength={0}
            autocompleteResults={{
              linkTarget: '_blank',
              titleField: 'name',
              urlField: 'url',
            }}
            autocompleteMinimumCharacters={3}
          />
        }
        sideContent={
          <div>
            {wasSearched && (
              <Sorting
                label="Sort by"
                sortOptions={[
                  { name: 'Relevance', value: '', direction: '' },
                  { name: 'Date', value: '_timestamp', direction: 'asc' },
                ]}
              />
            )}
            <Facet key="1" field="type.enum" label="Type" />
          </div>
        }
        bodyContent={<Results titleField="name" urlField="url" />}
        bodyHeader={
          <>
            {wasSearched && <PagingInfo />}
            {wasSearched && <ResultsPerPage />}
          </>
        }
        bodyFooter={<Paging />}
      />
    </ErrorBoundary>
  );
}

const SearchPage = withSearch(({ wasSearched }) => ({
  wasSearched,
}))(SearchPageCore);
