import { createSlice } from '@reduxjs/toolkit';

const TransitionReduxSlice = createSlice({
  name: 'transition-redux',
  initialState: {
    page: 1,
    pageSlow: 1,
  },
  reducers: {
    setPage: (state) => {
      state.page += 1;
    },
    setPageSlow: (state) => {
      state.pageSlow += 1;
    },
  },
});

export const { setPage, setPageSlow } = TransitionReduxSlice.actions;

export default TransitionReduxSlice.reducer;
