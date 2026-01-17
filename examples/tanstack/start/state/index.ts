import { configureStore } from '@reduxjs/toolkit';
import { useDispatch, useSelector, useStore } from 'react-redux';

import TransitionReduxReducer from './TransitionReduxSlice';

import type { Action, ThunkAction } from '@reduxjs/toolkit';
import type { TypedUseSelectorHook } from 'react-redux';

export function create_store() {
  const reducer = {
    'transition-redux': TransitionReduxReducer,
  };

  return configureStore({
    reducer,
  });
}

export type AppStore = ReturnType<typeof create_store>;
export type RootState = ReturnType<AppStore['getState']>;
export type AppDispatch = AppStore['dispatch'];
export type AppThunk<ReturnType = void> = ThunkAction<ReturnType, RootState, unknown, Action>;

export const useAppDispatch: () => AppDispatch = useDispatch;
export const useAppSelector: TypedUseSelectorHook<RootState> = useSelector;
export const useAppStore: () => AppStore = useStore;
