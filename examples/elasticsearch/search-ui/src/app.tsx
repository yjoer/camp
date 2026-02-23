import {
  ErrorBoundary,
  Facet,
  Paging,
  PagingInfo,
  Results,
  ResultsPerPage,
  SearchBox,
  SearchProvider,
  Sorting,
  withSearch,
} from '@elastic/react-search-ui';
import { Layout } from '@elastic/react-search-ui-views';
import ElasticsearchAPIConnector from '@elastic/search-ui-elasticsearch-connector';
import '@elastic/react-search-ui-views/lib/styles/styles.css';

import type { SearchDriverOptions } from '@elastic/search-ui';

const connector = new ElasticsearchAPIConnector({
  host: import.meta.env.VITE_ELASTICSEARCH_HOST,
  apiKey: import.meta.env.VITE_ELASTICSEARCH_API_KEY,
  index: import.meta.env.VITE_ELASTICSEARCH_INDEX,
});

const config: SearchDriverOptions = {
  searchQuery: {
    search_fields: {
      name: { weight: 2 },
      body: {},
    },
    result_fields: {
      name: { snippet: {} },
      body: { snippet: { size: 100, fallback: true } },
      revision: { raw: {} },
      updated_at: { raw: {} },
      url: { raw: {} },
    },
    facets: {
      type: { type: 'value' },
    },
    fuzziness: true,
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

export function App() {
  return (
    <SearchProvider config={config}>
      <SearchPage />
    </SearchProvider>
  );
}

interface SearchPageCoreProps {
  wasSearched: boolean;
}

function SearchPageCore({ wasSearched }: SearchPageCoreProps) {
  return (
    <ErrorBoundary>
      <Layout
        bodyContent={<Results titleField="name" urlField="url" />}
        bodyFooter={<Paging />}
        bodyHeader={
          <>
            {!!wasSearched && <PagingInfo />}
            {!!wasSearched && <ResultsPerPage />}
          </>
        }
        header={
          <SearchBox
            debounceLength={50}
            searchAsYouType
            // autocompleteResults={{
            //   linkTarget: '_blank',
            //   titleField: 'name',
            //   urlField: 'url',
            // }}
            // autocompleteMinimumCharacters={3}
          />
        }
        sideContent={
          <div>
            {!!wasSearched && (
              <Sorting
                label="Sort by"
                sortOptions={[
                  { name: 'Relevance', value: '', direction: '' },
                  { name: 'Oldest', value: 'updated_at', direction: 'asc' },
                  { name: 'Newest', value: 'updated_at', direction: 'desc' },
                ]}
              />
            )}
            <Facet key="1" field="type" label="Type" />
          </div>
        }
      />
    </ErrorBoundary>
  );
}

const SearchPage = withSearch(({ wasSearched }) => ({
  wasSearched,
}))(SearchPageCore);
